"""
Processing pipeline — main orchestration of separation, transcription, and post-processing.

Contains process_audio (the main pipeline) and process_url (URL wrapper).
"""

import os
import logging
import threading
import librosa
from pathlib import Path

from models.job import (
    ProcessingJob, save_job_to_disk, save_job_checkpoint,
    OUTPUT_DIR, UPLOAD_DIR,
)
from processing.separation import (
    separate_stems, separate_stems_roformer, separate_stems_mdx, separate_stems_ensemble,
    separate_stems_modal, is_modal_enabled, MODAL_AVAILABLE,
    ENHANCED_SEPARATOR_AVAILABLE, ENSEMBLE_SEPARATOR_AVAILABLE,
)
from processing.smart_extract import smart_separate
from processing.transcription import (
    transcribe_to_midi, detect_chords_for_job, convert_midi_to_musicxml, CHORD_DETECTOR_AVAILABLE,
)
from services.downloader import download_from_url

logger = logging.getLogger(__name__)

# ============ CONDITIONAL IMPORTS (single source of truth: dependencies.py) ============

from dependencies import (
    GUITAR_SEPARATOR_AVAILABLE, ENHANCER_AVAILABLE, SKILLS_AVAILABLE,
    TRACK_INFO_AVAILABLE, DRIVE_AVAILABLE, GP_CONVERTER_AVAILABLE,
    ENHANCED_SEPARATOR_AVAILABLE as _ESA,
)

if _ESA:
    from enhanced_separator import EnhancedSeparator  # noqa: F401
if GUITAR_SEPARATOR_AVAILABLE:
    from guitar_separator import GuitarSeparator  # noqa: F401
if ENHANCER_AVAILABLE:
    from enhancer import enhance_all_stems  # noqa: F401
if SKILLS_AVAILABLE:
    from skills import get_skill, get_all_skills, apply_skill, SKILL_REGISTRY, analyze_stems_for_skills  # noqa: F401
if TRACK_INFO_AVAILABLE:
    from track_info import fetch_track_info, extract_artist_from_title, should_stereo_split  # noqa: F401
if DRIVE_AVAILABLE:
    from drive_service import upload_job_to_drive  # noqa: F401
if GP_CONVERTER_AVAILABLE:
    from midi_to_gp import convert_job_midis_to_gp  # noqa: F401


# ============ DURATION LIMITS (seconds) ============

PLAN_DURATION_LIMITS = {
    'free': 30 * 60,      # 30 minutes (no restrictions during beta)
    'premium': 30 * 60,   # 30 minutes
    'pro': 30 * 60,       # 30 minutes
    'beta': 30 * 60,      # 30 minutes
}

PLAN_UPGRADE_MSG = {
    'free': 'Upgrade to Pro ($10/mo) for songs up to 10 minutes, or Premium ($20/mo) for up to 20 minutes.',
    'pro': 'Upgrade to Premium ($20/mo) for songs up to 20 minutes.',
    'beta': 'Beta plan supports songs up to 10 minutes.',
    'pro': 'Pro plan supports songs up to 20 minutes.',
}


class DurationLimitExceeded(Exception):
    """Raised when audio exceeds the plan's duration limit."""
    def __init__(self, duration_sec, limit_sec, plan):
        self.duration_sec = duration_sec
        self.limit_sec = limit_sec
        self.plan = plan
        dur_min = duration_sec / 60
        lim_min = limit_sec / 60
        upgrade_msg = PLAN_UPGRADE_MSG.get(plan, '')
        super().__init__(
            f"This song is {dur_min:.1f} minutes long, but your {plan} plan "
            f"supports up to {lim_min:.0f} minutes. {upgrade_msg}"
        )


def check_duration_limit(audio_path, plan='free'):
    """Check if audio duration is within the plan's limit. Raises DurationLimitExceeded if not."""
    limit = PLAN_DURATION_LIMITS.get(plan, PLAN_DURATION_LIMITS['free'])
    try:
        duration = librosa.get_duration(path=str(audio_path))
    except Exception:
        return  # If we can't determine duration, let it through
    if duration > limit:
        raise DurationLimitExceeded(duration, limit, plan)


# ============ PIPELINE FUNCTIONS ============

def apply_skills_to_job(job: ProcessingJob):
    """Analyze audio and apply only relevant skills based on content detection"""
    if not SKILLS_AVAILABLE:
        return

    job.stage = 'Analyzing audio content...'
    logger.info("🎧 Listening to stems to detect instruments...")

    # Smart detection - analyze stems and decide which skills to run
    skills_to_run = analyze_stems_for_skills(job.stems)

    if not skills_to_run:
        logger.info("No additional instrument content detected - skipping skill extraction")
        return

    job.selected_skills = skills_to_run  # Store for reference
    job.stage = f'Extracting {len(skills_to_run)} detected instruments'
    logger.info(f"🎯 Detected instruments, running skills: {skills_to_run}")

    output_dir = OUTPUT_DIR / job.job_id / 'stems'

    for skill_id in skills_to_run:
        try:
            skill = get_skill(skill_id)
            if not skill:
                logger.warning(f"Unknown skill: {skill_id}")
                continue

            job.stage = f'Applying {skill.emoji} {skill.name}'
            logger.info(f"Applying skill: {skill.name}")

            # Apply the skill
            sub_stem_paths = apply_skill(skill_id, job.stems, str(output_dir))

            if sub_stem_paths:
                # Convert absolute paths to relative paths for API
                job.sub_stems[skill_id] = {}
                for sub_stem_name, abs_path in sub_stem_paths.items():
                    # Store just the filename for the API
                    rel_path = f"skill_{skill_id}/{os.path.basename(abs_path)}"
                    job.sub_stems[skill_id][sub_stem_name] = rel_path
                    logger.info(f"  Generated sub-stem: {sub_stem_name}")

        except Exception as e:
            logger.error(f"Skill {skill_id} failed: {e}")


def process_audio(job: ProcessingJob, audio_path: Path, enhance_stems: bool = False, stereo_split: bool = False, gp_tabs: bool = True, chord_detection: bool = True, mdx_model: bool = False, ensemble_mode: bool = False):
    """Main processing pipeline - simplified: separate → (optional stereo split) → transcribe → done"""
    try:
        job.status = 'processing'
        save_job_checkpoint(job)

        # Check duration against plan limits
        plan = job.metadata.get('plan', 'free')
        try:
            check_duration_limit(audio_path, plan)
        except DurationLimitExceeded as e:
            logger.warning(f"Duration limit exceeded for job {job.job_id}: {e}")
            job.status = 'failed'
            job.error = str(e)
            job.metadata['duration_limited'] = True
            save_job_checkpoint(job)
            try:
                from error_tracker import log_error
                log_error(job.job_id, 'duration_limit', str(e),
                          song_duration=e.duration_sec, processing_stage='pre-check')
            except Exception:
                pass
            return

        # Step 1: Separate stems
        # Mode selection: Modal cloud GPU > explicit user choice > default
        separation_success = False

        print(f"[MODAL DEBUG] MODAL_AVAILABLE={MODAL_AVAILABLE}, ensemble={ensemble_mode}, mdx={mdx_model}", flush=True)
        if MODAL_AVAILABLE and not ensemble_mode and not mdx_model:
            # Modal cloud GPU — default when enabled, skips local GPU entirely
            logger.info("☁️ Using Modal cloud GPU (T4) for separation")
            separation_success = separate_stems_modal(job, audio_path)
        elif ensemble_mode and ENSEMBLE_SEPARATOR_AVAILABLE:
            logger.info("🎯 Using ENSEMBLE mode (multi-model voting)")
            separation_success = separate_stems_ensemble(job, audio_path)
        elif mdx_model and ENHANCED_SEPARATOR_AVAILABLE:
            logger.info("🎹 Using MDX HYBRID mode (MDX23C vocals + Demucs instruments)")
            separation_success = separate_stems_mdx(job, audio_path, stereo_split_guitar=stereo_split)
        elif ENHANCED_SEPARATOR_AVAILABLE:
            # Default: RoFormer+Demucs ensemble (best quality)
            logger.info("🧠 Using RoFormer+Demucs ensemble (state-of-the-art)")
            separation_success = separate_stems_roformer(job, audio_path)
        else:
            # Fallback: pure Demucs if audio-separator not installed
            logger.info("🎸 Falling back to standard Demucs (audio-separator not available)")
            separation_success = separate_stems(job, audio_path)

        if not separation_success:
            job.status = 'failed'
            try:
                from error_tracker import log_error
                log_error(job.job_id, 'separation_failed', 'Stem separation returned failure',
                          song_duration=job.metadata.get('duration'),
                          source='upload' if not job.source_url else 'url',
                          processing_stage='separation')
            except Exception:
                pass
            return

        # Optional: Enhance stems (off by default)
        if enhance_stems and ENHANCER_AVAILABLE:
            job.stage = 'Enhancing stems'
            job.progress = 52
            try:
                enhanced_output_dir = OUTPUT_DIR / job.job_id / 'stems' / 'enhanced'
                enhanced_output_dir.mkdir(parents=True, exist_ok=True)
                job.enhanced_stems = enhance_all_stems(job.stems, str(enhanced_output_dir))
                logger.info(f"✅ Enhanced {len(job.enhanced_stems)} stems")
            except Exception as e:
                logger.warning(f"Stem enhancement failed: {e}")

        # Auto-split vocals into lead/backing using Karaoke model
        # RoFormer gives clean combined vocals → Karaoke model separates lead vs backing
        if ENHANCED_SEPARATOR_AVAILABLE and 'vocals' in job.stems:
            try:
                job.stage = 'Splitting lead/backing vocals (BS-Roformer)'
                job.progress = 45
                save_job_checkpoint(job)
                logger.info("🎤 Auto-splitting vocals with BS-Roformer for better quality...")

                vocal_split_dir = OUTPUT_DIR / job.job_id / 'stems' / 'vocal_split'
                vocal_split_dir.mkdir(parents=True, exist_ok=True)

                separator = EnhancedSeparator(output_dir=str(vocal_split_dir))
                lead_path, backing_path = separator.split_lead_backing_vocals(job.stems['vocals'])

                if lead_path and backing_path:
                    # Ensure absolute paths (audio-separator may return relative)
                    lead_abs = str(vocal_split_dir / Path(lead_path).name) if not os.path.isabs(lead_path) else lead_path
                    backing_abs = str(vocal_split_dir / Path(backing_path).name) if not os.path.isabs(backing_path) else backing_path
                    job.stems['vocals_lead'] = lead_abs
                    job.stems['vocals_backing'] = backing_abs
                    logger.info(f"✅ Vocals split: lead={Path(lead_abs).name}, backing={Path(backing_abs).name}")
                else:
                    logger.warning("Vocal split returned empty paths")
            except Exception as e:
                logger.warning(f"Auto vocal split failed (non-fatal): {e}")

        # Auto-split guitar into lead/rhythm using trained MelBand-RoFormer
        if GUITAR_SEPARATOR_AVAILABLE and 'guitar' in job.stems:
            try:
                job.stage = 'Splitting lead/rhythm guitar (MelBand-RoFormer)'
                job.progress = 48
                save_job_checkpoint(job)
                logger.info("🎸 Auto-splitting guitar into lead/rhythm with trained MelBand-RoFormer...")

                guitar_split_dir = OUTPUT_DIR / job.job_id / 'stems' / 'guitar_split'
                guitar_split_dir.mkdir(parents=True, exist_ok=True)

                guitar_sep = GuitarSeparator(output_dir=str(guitar_split_dir))
                lead_guitar_path, rhythm_guitar_path = guitar_sep.separate(job.stems['guitar'])

                if lead_guitar_path and rhythm_guitar_path:
                    # Convert to MP3 for consistency with other stems
                    lead_mp3 = guitar_split_dir / 'guitar_lead.mp3'
                    rhythm_mp3 = guitar_split_dir / 'guitar_rhythm.mp3'

                    import subprocess
                    for src, dst in [(lead_guitar_path, lead_mp3), (rhythm_guitar_path, rhythm_mp3)]:
                        try:
                            subprocess.run([
                                'ffmpeg', '-y', '-i', str(src),
                                '-codec:a', 'libmp3lame', '-b:a', '320k',
                                str(dst)
                            ], capture_output=True, timeout=120)
                        except Exception:
                            pass

                    if lead_mp3.exists():
                        job.stems['guitar_lead'] = str(lead_mp3)
                    else:
                        job.stems['guitar_lead'] = lead_guitar_path

                    if rhythm_mp3.exists():
                        job.stems['guitar_rhythm'] = str(rhythm_mp3)
                    else:
                        job.stems['guitar_rhythm'] = rhythm_guitar_path

                    logger.info(f"✅ Guitar split: lead={Path(job.stems['guitar_lead']).name}, "
                                f"rhythm={Path(job.stems['guitar_rhythm']).name}")
                else:
                    logger.warning("Guitar split returned empty paths")
            except Exception as e:
                logger.warning(f"Auto guitar lead/rhythm split failed (non-fatal): {e}")

        # Smart Deep Extraction: Automatically analyze and extract all instruments
        # Replaces manual cascade separation — one button, best result every time
        job.stage = 'Analyzing stems for deep extraction'
        job.progress = 52
        save_job_checkpoint(job)
        try:
            smart_result = smart_separate(job)
            if smart_result:
                logger.info(f"🧠 Smart extraction found additional instruments ({len(job.stems)} total stems)")
            else:
                logger.info("🧠 Track is cleanly separated — no deep extraction needed")
        except Exception as e:
            logger.warning(f"Smart separation failed (non-fatal): {e}")

        # Step 1b: Detect vocal onset time (for karaoke sync)
        # Analyzes the vocal stem to find when singing actually starts
        vocals_stem_path = job.stems.get('vocals_lead') or job.stems.get('vocals')
        if vocals_stem_path and Path(vocals_stem_path).exists():
            try:
                import numpy as np
                import subprocess
                logger.info("🎤 Detecting vocal onset time for karaoke sync...")

                # Decode vocal stem to raw PCM using ffmpeg
                result = subprocess.run([
                    'ffmpeg', '-i', str(vocals_stem_path),
                    '-f', 'f32le', '-ac', '1', '-ar', '22050', '-'
                ], capture_output=True, timeout=30)

                if result.returncode == 0 and len(result.stdout) > 0:
                    samples = np.frombuffer(result.stdout, dtype=np.float32)
                    sr = 22050

                    # Compute RMS energy in 50ms windows
                    window_size = int(sr * 0.05)
                    num_windows = len(samples) // window_size
                    rms = np.array([
                        np.sqrt(np.mean(samples[i*window_size:(i+1)*window_size]**2))
                        for i in range(num_windows)
                    ])

                    # Find first window where RMS exceeds threshold (vocal activity)
                    # Use adaptive threshold: 5x the median noise floor
                    noise_floor = np.median(rms[:int(sr / window_size * 2)])  # first 2 seconds
                    threshold = max(noise_floor * 5, 0.005)

                    onset_window = None
                    # Require 3 consecutive windows above threshold (not just a blip)
                    for i in range(len(rms) - 2):
                        if rms[i] > threshold and rms[i+1] > threshold and rms[i+2] > threshold:
                            onset_window = i
                            break

                    if onset_window is not None:
                        vocals_onset = onset_window * 0.05  # convert to seconds
                        job.metadata['vocals_onset_time'] = round(vocals_onset, 2)
                        logger.info(f"✓ Vocal onset detected at {vocals_onset:.2f}s (threshold={threshold:.4f})")
                    else:
                        logger.info("ℹ No clear vocal onset detected — may be instrumental")
            except Exception as e:
                logger.warning(f"Vocal onset detection failed (non-fatal): {e}")

        # Step 2: Detect chord progression (optional - before transcription)
        if chord_detection and CHORD_DETECTOR_AVAILABLE:
            job.stage = 'Detecting chords and key'
            job.progress = 58
            try:
                detect_chords_for_job(job, audio_path)
            except Exception as e:
                logger.warning(f"Chord detection failed (non-fatal): {e}")
        elif not chord_detection:
            logger.info("⏭️ Skipping chord detection (disabled)")

        # Step 2b: Generate chord chart (chords + lyrics + structure)
        # Wrapped in a timeout thread to prevent stalls from Whisper/lyrics extraction
        if job.chord_progression:
            CHART_TIMEOUT = 180  # 3 minutes max for chord chart generation
            chart_result = [None]
            chart_error = [None]

            def _generate_chart():
                try:
                    from chart_assembler import generate_chord_chart_for_job
                    vocals_stem = job.stems.get('vocals_lead') or job.stems.get('vocals')
                    title = job.metadata.get('title', job.filename or 'Untitled') if job.metadata else (job.filename or 'Untitled')
                    artist = job.metadata.get('artist', '') if job.metadata else ''
                    chart_result[0] = generate_chord_chart_for_job(
                        job_id=job.job_id,
                        chords=job.chord_progression,
                        vocals_path=vocals_stem,
                        audio_path=str(audio_path),
                        title=title,
                        artist=artist,
                    )
                except Exception as e:
                    chart_error[0] = e

            try:
                # Chord chart generation SKIPPED — using chord library + Songsterr instead of AI
                # AI chord detection produces wrong chords for guitar/bass
                logger.info("Skipping AI chord chart generation (using chord library + Songsterr)")
                job.progress = 59
                save_job_checkpoint(job)

                if chart_thread.is_alive():
                    logger.warning(f"Chord chart generation timed out after {CHART_TIMEOUT}s — skipping")
                elif chart_error[0]:
                    logger.warning(f"Chord chart generation failed (non-fatal): {chart_error[0]}")
                elif chart_result[0]:
                    chart = chart_result[0]
                    num_sections = len(chart.get('sections', []))
                    num_lines = sum(len(s.get('lines', [])) for s in chart.get('sections', []))
                    logger.info(f"✓ Auto-generated chord chart: {num_sections} sections, {num_lines} lines")
                else:
                    logger.info("Chord chart generation returned no result")
            except Exception as e:
                logger.warning(f"Chord chart generation failed (non-fatal): {e}")

        # Step 3: Transcribe to MIDI
        job.progress = 60
        save_job_checkpoint(job)
        midi_success = transcribe_to_midi(job)
        if midi_success:
            logger.info(f"✓ MIDI transcription succeeded for {len(job.midi_files)} stems")

            # Step 3: Convert MIDI to MusicXML for notation
            xml_success = convert_midi_to_musicxml(job)
            if xml_success:
                logger.info(f"✓ MusicXML conversion succeeded for {len(job.musicxml_files)} stems")

            # Step 4: Convert MIDI to Guitar Pro for tablature (optional)
            if gp_tabs and GP_CONVERTER_AVAILABLE:
                job.stage = 'Creating Guitar Pro tabs'
                job.progress = 97
                try:
                    gp_output_dir = OUTPUT_DIR / job.job_id / 'guitarpro'
                    gp_files = convert_job_midis_to_gp(job, gp_output_dir)
                    if gp_files:
                        job.gp_files = gp_files
                        logger.info(f"✓ Guitar Pro conversion succeeded for {len(gp_files)} stems")
                except Exception as e:
                    logger.warning(f"Guitar Pro conversion failed (non-fatal): {e}")
            elif not gp_tabs:
                logger.info("⏭️ Skipping Guitar Pro tabs (disabled)")

            # Step 5: Try to fetch professional tabs from Songsterr (background, non-blocking)
            try:
                title = job.metadata.get('title', '') if job.metadata else ''
                artist = job.metadata.get('artist', '') if job.metadata else ''
                if title:
                    job.stage = 'Fetching professional tabs'
                    job.progress = 98
                    from songsterr import SongsterrAPI
                    api = SongsterrAPI()
                    tab = api.search(f"{title} {artist}".strip())
                    if tab:
                        output_dir = OUTPUT_DIR / job.job_id
                        gp5_path = api.download_gp5(tab, output_dir)
                        if gp5_path:
                            if not hasattr(job, 'pro_tabs') or job.pro_tabs is None:
                                job.pro_tabs = {}
                            job.pro_tabs['songsterr'] = {
                                'path': str(gp5_path),
                                'title': tab.title,
                                'artist': tab.artist,
                                'song_id': tab.song_id,
                                'tracks': [t.get('name', '') for t in tab.tracks[:5]]
                            }
                            logger.info(f"✓ Downloaded professional tabs from Songsterr: {tab.title}")
                        else:
                            logger.info("⚠ Songsterr tab found but download failed")
                    else:
                        logger.info("ℹ No Songsterr tab found for this song")
            except Exception as e:
                logger.debug(f"Pro tabs fetch skipped: {e}")
        else:
            logger.warning("⚠ MIDI transcription failed - check logs above for details")

        job.progress = 100
        job.stage = 'Complete'
        job.status = 'completed'
        logger.info(f"Job {job.job_id} completed successfully with {len(job.stems)} stems, {len(job.sub_stems)} skill outputs")

        # Auto-upload to Google Drive (transcriptions only, not stems)
        if DRIVE_AVAILABLE:
            try:
                job.stage = 'Uploading to Google Drive'
                drive_result = upload_job_to_drive(job, keep_stems=False)
                if drive_result:
                    job.metadata['drive_upload'] = drive_result
                    logger.info(f"Uploaded job {job.job_id} to Google Drive")
            except Exception as e:
                logger.warning(f"Drive upload failed (non-fatal): {e}")

        # Cache URL results for future lookups
        if job.source_url:
            try:
                from url_cache import add_to_cache
                add_to_cache(
                    url=job.source_url,
                    job_id=job.job_id,
                    title=job.metadata.get('title', ''),
                    artist=job.metadata.get('artist', ''),
                    stem_count=len(job.stems),
                    has_chords=bool(job.chord_progression),
                    has_gp=bool(job.gp_files),
                    has_midi=bool(job.midi_files),
                )
            except Exception as e:
                logger.warning(f"Failed to cache URL result: {e}")

        # Save job to disk for library persistence
        save_job_to_disk(job)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        job.status = 'failed'
        job.error = str(e)
        # Log to error tracker
        try:
            from error_tracker import log_error
            source = 'upload'
            if job.source_url:
                if 'archive.org' in (job.source_url or ''):
                    source = 'archive'
                elif 'youtu' in (job.source_url or ''):
                    source = 'youtube'
                elif 'spotify' in (job.source_url or ''):
                    source = 'spotify'
                else:
                    source = 'url'
            log_error(
                job_id=job.job_id,
                error_type=type(e).__name__,
                error_message=str(e),
                song_duration=job.metadata.get('duration'),
                source=source,
                processing_stage=job.stage,
                extra={'filename': job.filename},
            )
        except Exception:
            pass  # Error tracking itself must never break the pipeline


def process_url(job: ProcessingJob, url: str, enhance_stems: bool = False, stereo_split: bool = False, gp_tabs: bool = True, chord_detection: bool = True, mdx_model: bool = False, ensemble_mode: bool = False):
    """Download from URL then process"""
    try:
        job.status = 'processing'

        # Create upload directory for this job
        job_upload_dir = UPLOAD_DIR / job.job_id
        job_upload_dir.mkdir(exist_ok=True)

        # Step 0: Download audio
        audio_path = download_from_url(job, url, job_upload_dir)

        # DISABLED: Stereo splitting creates too many overlapping stems
        # Instead we use cascade separation on the 'other' stem
        # if not stereo_split and TRACK_INFO_AVAILABLE:
        #     artist = job.metadata.get('artist', '').lower()
        #     title = job.metadata.get('title', '').lower()
        #     if should_stereo_split(artist) or should_stereo_split(title):
        #         stereo_split = True
        #         logger.info(f"🎸 Auto-enabled stereo split for dual-guitar band: {artist}")

        # Steps 1-2: Separate and transcribe
        process_audio(job, audio_path, enhance_stems=enhance_stems, stereo_split=stereo_split, gp_tabs=gp_tabs, chord_detection=chord_detection, mdx_model=mdx_model, ensemble_mode=ensemble_mode)

    except Exception as e:
        logger.error(f"URL processing failed: {e}")
        job.status = 'failed'
        job.error = str(e)
        # Log to error tracker
        try:
            from error_tracker import log_error
            source = 'unknown'
            if 'archive.org' in (url or ''):
                source = 'archive'
            elif 'youtu' in (url or ''):
                source = 'youtube'
            elif 'spotify' in (url or ''):
                source = 'spotify'
            elif 'soundcloud' in (url or ''):
                source = 'soundcloud'
            else:
                source = 'url'
            log_error(
                job_id=job.job_id,
                error_type=type(e).__name__,
                error_message=str(e),
                song_duration=job.metadata.get('duration'),
                source=source,
                processing_stage=job.stage,
                extra={'url': url, 'filename': job.filename},
            )
        except Exception:
            pass  # Error tracking itself must never break the pipeline


