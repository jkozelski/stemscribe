"""
StemScribe Enhancement Integration Guide
=========================================
This file shows how to integrate the new modules into app.py.

New Modules Created:
1. transcriber_enhanced.py - Better MIDI transcription with articulations
2. drum_transcriber_v2.py - Ghost notes, cymbal differentiation
3. chord_detector.py - Chord progression detection
4. band_database_extended.py - Phish, Allman Brothers deep coverage

To integrate, add the imports and modify the relevant functions in app.py.
"""

# ============================================================================
# STEP 1: Add imports to app.py (after existing imports)
# ============================================================================

IMPORTS_TO_ADD = '''
# Enhanced transcription (NEW)
try:
    from transcriber_enhanced import EnhancedTranscriber, transcribe_with_enhanced
    ENHANCED_TRANSCRIBER_AVAILABLE = True
    logger.info("Enhanced transcriber available")
except ImportError as e:
    ENHANCED_TRANSCRIBER_AVAILABLE = False
    logger.warning(f"Enhanced transcriber not available: {e}")

# Improved drum transcription (NEW)
try:
    from drum_transcriber_v2 import EnhancedDrumTranscriber, transcribe_drums_to_midi as transcribe_drums_v2
    DRUM_TRANSCRIBER_V2_AVAILABLE = True
    logger.info("Enhanced drum transcriber v2 available")
except ImportError as e:
    DRUM_TRANSCRIBER_V2_AVAILABLE = False
    logger.warning(f"Drum transcriber v2 not available: {e}")

# Chord detection (NEW)
try:
    from chord_detector import ChordDetector, detect_chords
    CHORD_DETECTOR_AVAILABLE = True
    logger.info("Chord detector available")
except ImportError as e:
    CHORD_DETECTOR_AVAILABLE = False
    logger.warning(f"Chord detector not available: {e}")

# Extended band database (NEW)
try:
    from band_database_extended import (
        get_player_positions as get_extended_positions,
        get_era_info as get_extended_era,
        get_album_info as get_extended_album,
        get_knowledge as get_extended_knowledge,
        should_stereo_split as extended_should_stereo_split,
        PHISH_KNOWLEDGE, ALLMAN_KNOWLEDGE
    )
    EXTENDED_BAND_DB_AVAILABLE = True
    logger.info("Extended band database available (Phish, Allman Brothers)")
except ImportError as e:
    EXTENDED_BAND_DB_AVAILABLE = False
    logger.warning(f"Extended band database not available: {e}")
'''


# ============================================================================
# STEP 2: Update ProcessingJob class to include new fields
# ============================================================================

PROCESSINGJOB_ADDITIONS = '''
class ProcessingJob:
    def __init__(self, job_id, filename, source_url=None, skills=None):
        # ... existing fields ...
        self.chord_progression = []  # NEW: Detected chords
        self.detected_key = None     # NEW: Detected key
        self.articulations = {}      # NEW: Per-stem articulations
        self.transcription_quality = {}  # NEW: Quality scores per stem

    def to_dict(self):
        return {
            # ... existing fields ...
            'chord_progression': self.chord_progression,  # NEW
            'detected_key': self.detected_key,            # NEW
            'transcription_quality': self.transcription_quality,  # NEW
        }
'''


# ============================================================================
# STEP 3: Replace transcribe_to_midi function with enhanced version
# ============================================================================

ENHANCED_TRANSCRIBE_FUNCTION = '''
def transcribe_to_midi_enhanced(job: ProcessingJob, quantize: bool = True,
                                 grid_size: float = 0.125,
                                 detect_articulations: bool = True):
    """
    Enhanced MIDI transcription with articulation detection.
    Falls back to Basic Pitch if enhanced transcriber unavailable.
    """
    import traceback

    job.stage = 'Transcribing to MIDI (enhanced)'
    job.progress = 60
    logger.info("Starting enhanced MIDI transcription")

    midi_output_dir = OUTPUT_DIR / job.job_id / 'midi'
    midi_output_dir.mkdir(parents=True, exist_ok=True)

    total_stems = len(job.stems)
    successful = 0
    SKIP_STEMS = {'other', 'instrumental'}

    # Use enhanced transcriber if available
    use_enhanced = ENHANCED_TRANSCRIBER_AVAILABLE
    if use_enhanced:
        transcriber = EnhancedTranscriber()
        logger.info("Using enhanced transcriber with articulation detection")

    for idx, (stem_name, stem_path) in enumerate(job.stems.items()):
        try:
            if stem_name.lower() in SKIP_STEMS:
                logger.info(f"⏭️ Skipping {stem_name} (mixed content)")
                continue

            if not Path(stem_path).exists():
                logger.error(f"Stem file not found: {stem_path}")
                continue

            # Use v2 drum transcriber for drums
            if stem_name.lower() == 'drums' or '_drums' in stem_name.lower():
                if DRUM_TRANSCRIBER_V2_AVAILABLE:
                    logger.info(f"🥁 Transcribing {stem_name} with enhanced drum transcriber v2")
                    job.stage = f'Transcribing {stem_name} (v2)'
                    drum_midi_path = midi_output_dir / f"{stem_name}_transcribed.mid"

                    drum_transcriber = EnhancedDrumTranscriber()
                    stats = drum_transcriber.transcribe(
                        audio_path=stem_path,
                        output_path=str(drum_midi_path),
                        detect_ghost_notes=True,
                        preserve_dynamics=True
                    )

                    if stats.total_hits > 0:
                        job.midi_files[stem_name] = str(drum_midi_path)
                        job.transcription_quality[stem_name] = stats.quality_score
                        logger.info(f"✓ Drum MIDI: {stats.total_hits} hits, "
                                   f"ghost: {stats.ghost_notes}, quality: {stats.quality_score:.2f}")
                        successful += 1
                    continue
                elif DRUM_TRANSCRIBER_AVAILABLE:
                    # Fall back to v1
                    logger.info(f"🥁 Transcribing {stem_name} with drum transcriber v1")
                    job.stage = f'Transcribing {stem_name}'
                    drum_midi_path = midi_output_dir / f"{stem_name}_transcribed.mid"
                    if transcribe_drums_to_midi(stem_path, str(drum_midi_path)):
                        job.midi_files[stem_name] = str(drum_midi_path)
                        successful += 1
                    continue

            # Use enhanced transcriber for melodic instruments
            if use_enhanced and stem_name.lower() in ['guitar', 'bass', 'vocals', 'piano']:
                logger.info(f"🎼 Transcribing {stem_name} with enhanced transcriber")
                job.stage = f'Transcribing {stem_name} (enhanced)'

                result = transcriber.transcribe(
                    audio_path=stem_path,
                    output_dir=str(midi_output_dir),
                    stem_type=stem_name.lower().split('_')[0],  # Handle guitar_left, etc.
                    detect_articulations=detect_articulations,
                    quantize=quantize,
                    quantize_grid=grid_size
                )

                if result.midi_path:
                    job.midi_files[stem_name] = result.midi_path
                    job.transcription_quality[stem_name] = result.quality_score
                    job.articulations[stem_name] = len(result.articulations)

                    if result.detected_key and not job.detected_key:
                        job.detected_key = result.detected_key

                    logger.info(f"✓ Enhanced MIDI for {stem_name}: "
                               f"{result.notes_count} notes, "
                               f"{len(result.articulations)} articulations, "
                               f"quality: {result.quality_score:.2f}")
                    successful += 1
                continue

            # Fall back to Basic Pitch for other stems
            logger.info(f"Transcribing {stem_name} with Basic Pitch...")
            job.stage = f'Transcribing {stem_name}'

            try:
                from basic_pitch.inference import predict_and_save
                from basic_pitch import ICASSP_2022_MODEL_PATH

                predict_and_save(
                    audio_path_list=[stem_path],
                    output_directory=str(midi_output_dir),
                    save_midi=True,
                    sonify_midi=False,
                    save_model_outputs=False,
                    save_notes=False,
                    model_or_model_path=ICASSP_2022_MODEL_PATH,
                    onset_threshold=0.6,
                    frame_threshold=0.4,
                    minimum_note_length=80,
                )

                # Find generated MIDI
                stem_basename = Path(stem_path).stem
                midi_file = midi_output_dir / f"{stem_basename}_basic_pitch.mid"
                if not midi_file.exists():
                    midi_file = midi_output_dir / f"{stem_basename}.mid"

                if midi_file.exists():
                    # Quantize
                    if quantize:
                        quantize_midi(str(midi_file), grid_size=grid_size)

                    job.midi_files[stem_name] = str(midi_file)
                    logger.info(f"✓ Created MIDI for {stem_name}")
                    successful += 1

            except ImportError:
                logger.warning(f"Basic Pitch not available for {stem_name}")

            job.progress = 60 + int((idx + 1) / total_stems * 35)

        except Exception as e:
            logger.error(f"Transcription failed for {stem_name}: {e}")
            logger.error(traceback.format_exc())

    logger.info(f"MIDI transcription complete: {successful}/{total_stems} stems")
    return successful > 0
'''


# ============================================================================
# STEP 4: Add chord detection to processing pipeline
# ============================================================================

CHORD_DETECTION_FUNCTION = '''
def detect_chords_for_job(job: ProcessingJob, audio_path: Path):
    """Detect chord progression from original audio or mix of stems."""
    if not CHORD_DETECTOR_AVAILABLE:
        return

    job.stage = 'Detecting chord progression'
    logger.info("🎸 Detecting chords...")

    try:
        detector = ChordDetector()

        # Prefer to analyze the original audio if we have it
        # Otherwise, use the guitar or piano stem
        analyze_path = str(audio_path)

        if not Path(analyze_path).exists():
            # Try guitar stem
            if 'guitar' in job.stems:
                analyze_path = job.stems['guitar']
            elif 'piano' in job.stems:
                analyze_path = job.stems['piano']
            else:
                logger.info("No suitable source for chord detection")
                return

        progression = detector.detect(analyze_path)

        # Store results
        job.chord_progression = [
            {
                'time': c.time,
                'duration': c.duration,
                'chord': c.chord,
                'confidence': c.confidence
            }
            for c in progression.chords
        ]
        job.detected_key = progression.key

        logger.info(f"✅ Detected {len(job.chord_progression)} chord changes, key: {progression.key}")

    except Exception as e:
        logger.warning(f"Chord detection failed: {e}")
'''


# ============================================================================
# STEP 5: Update process_audio to use new features
# ============================================================================

UPDATED_PROCESS_AUDIO = '''
def process_audio(job: ProcessingJob, audio_path: Path,
                  hq_vocals: bool = False, vocal_focus: bool = False,
                  enhance_stems: bool = False, stereo_split: bool = False,
                  detect_chords: bool = True):  # NEW parameter
    """Main processing pipeline with enhanced features."""
    try:
        job.status = 'processing'

        # Step 1: Separate stems (existing code)
        # ...

        # Step 2: Chord detection (NEW - before transcription)
        if detect_chords:
            detect_chords_for_job(job, audio_path)

        # Step 3: Enhanced MIDI transcription (UPDATED)
        job.progress = 50
        midi_success = transcribe_to_midi_enhanced(
            job,
            quantize=True,
            grid_size=0.125,  # 16th notes
            detect_articulations=True
        )

        # ... rest of existing code ...
'''


# ============================================================================
# STEP 6: Add new API endpoint for chords
# ============================================================================

NEW_API_ENDPOINT = '''
@app.route('/api/chords/<job_id>', methods=['GET'])
def get_chords(job_id):
    """Get detected chord progression for a job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    return jsonify({
        'chords': job.chord_progression,
        'key': job.detected_key,
        'job_id': job_id
    })
'''


# ============================================================================
# STEP 7: Update track_info.py to use extended band database
# ============================================================================

TRACK_INFO_UPDATE = '''
# Add to track_info.py after existing imports:

try:
    from band_database_extended import (
        get_player_positions as get_extended_positions,
        get_era_info as get_extended_era,
        get_album_info as get_extended_album,
        get_knowledge as get_extended_knowledge,
        should_stereo_split as extended_should_stereo_split,
    )
    EXTENDED_DB_AVAILABLE = True
except ImportError:
    EXTENDED_DB_AVAILABLE = False

# Then update fetch_track_info to check extended database:

def fetch_track_info(track_name: str, artist: str = None, source_url: str = None):
    """..."""
    # ... existing code ...

    # Check extended database if available
    if EXTENDED_DB_AVAILABLE and artist:
        ext_knowledge = get_extended_knowledge(artist)
        if ext_knowledge:
            info['bio'] = ext_knowledge.get('bio', info.get('bio'))
            info['members'] = ext_knowledge.get('members', info.get('members'))
            info['learning_tips'] = ext_knowledge.get('learning_tips')
            info['style'] = ext_knowledge.get('style')
            info['fetched_from'].append('extended_band_db')

            # Get extended player mapping
            ext_positions = get_extended_positions(artist)
            if ext_positions:
                info['player_mapping'] = ext_positions

        # Check extended album database
        if track_name:
            ext_album = get_extended_album(artist, track_name)
            if ext_album:
                info['album'] = ext_album.get('album_name')
                info['album_year'] = ext_album.get('year')
                info['album_description'] = ext_album.get('description')
                info['personnel'] = ext_album.get('personnel')
                info['era'] = ext_album.get('era')
                info['fetched_from'].append('extended_album_db')

    # ... rest of existing code ...
'''


# ============================================================================
# SUMMARY
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║              STEMSCRIBE ENHANCEMENT INTEGRATION GUIDE                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  New modules created:                                                 ║
║  ├── transcriber_enhanced.py  - Polyphonic accuracy, articulations   ║
║  ├── drum_transcriber_v2.py   - Ghost notes, cymbal differentiation  ║
║  ├── chord_detector.py        - Chord progression detection          ║
║  └── band_database_extended.py - Phish, Allman Brothers coverage     ║
║                                                                       ║
║  Integration steps:                                                   ║
║  1. Add imports to app.py                                             ║
║  2. Update ProcessingJob class                                        ║
║  3. Replace transcribe_to_midi with enhanced version                  ║
║  4. Add chord detection to pipeline                                   ║
║  5. Update process_audio function                                     ║
║  6. Add /api/chords endpoint                                          ║
║  7. Update track_info.py for extended band database                   ║
║                                                                       ║
║  See code blocks above for specific changes to make.                  ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")
