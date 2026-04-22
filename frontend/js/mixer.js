// StemScribe — Mixer / Web Audio / VU Meters / Playback
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.ensureMixerAudioCtx = function() {
        if (!SS.mixerAudioCtx || SS.mixerAudioCtx.state === 'closed') {
            SS.mixerAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (SS.mixerAudioCtx.state === 'suspended') {
            SS.mixerAudioCtx.resume().catch(function() {});
        }
        return SS.mixerAudioCtx;
    };

    SS.connectStemToWebAudio = function(name, audioEl) {
        var ctx = SS.ensureMixerAudioCtx();
        var source = ctx.createMediaElementSource(audioEl);
        var gainNode = ctx.createGain();
        var analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.8;
        source.connect(gainNode);
        gainNode.connect(analyser);
        analyser.connect(ctx.destination);
        return { source: source, gainNode: gainNode, analyser: analyser };
    };

    SS.updateStemVolumes = function() {
        var masterVol = document.getElementById('masterVolume').value / 100;
        var hasSolo = SS.soloedStems.size > 0;

        Object.entries(SS.stemAudios).forEach(function(entry) {
            var name = entry[0];
            var data = entry[1];
            var vol = data.volume * masterVol;

            if (hasSolo && !SS.soloedStems.has(name)) {
                vol = 0;
            }
            if (data.muted) {
                vol = 0;
            }
            if (data.gainNode) {
                data.gainNode.gain.value = vol;
            } else {
                data.audio.volume = vol;
            }
        });
    };

    SS.togglePlayback = function() {
        var btn = document.getElementById('masterPlayBtn');
        if (SS.isPlaying) {
            Object.values(SS.stemAudios).forEach(function(s) { s.audio.pause(); });
            btn.textContent = '\u25B6';
            SS.isPlaying = false;
        } else {
            SS.ensureMixerAudioCtx();
            var currentTime = Object.values(SS.stemAudios)[0]?.audio.currentTime || 0;
            var stems = Object.values(SS.stemAudios);
            stems.forEach(function(s) { s.audio.currentTime = currentTime; });
            stems.forEach(function(s) {
                s.audio.play().catch(function(e) { console.warn('Playback blocked:', e); });
            });
            btn.textContent = '\u23F8';
            SS.isPlaying = true;
        }
    };

    SS.updateTimeline = function() {
        if (Object.keys(SS.stemAudios).length === 0) return;
        var audio = Object.values(SS.stemAudios)[0].audio;
        var percent = (audio.currentTime / SS.duration) * 100 || 0;
        document.getElementById('timelineProgress').style.width = percent + '%';
        document.getElementById('timelineHandle').style.left = percent + '%';
        document.getElementById('currentTime').textContent = SS.formatTime(audio.currentTime);

        // Sync waveform cursor
        if (SS.updateWaveformProgress) {
            SS.updateWaveformProgress();
        }

        if (audio.currentTime >= SS.duration - 0.1 && SS.isPlaying) {
            Object.values(SS.stemAudios).forEach(function(s) { s.audio.currentTime = 0; });
        }
    };

    SS.updateFaderCap = function(stem, value) {
        var cap = document.getElementById('fader-cap-' + stem);
        if (cap) {
            var position = 8 + (100 - value) * 0.8;
            cap.style.top = position + 'px';
        }
    };

    // VU Meter Animation
    SS.startVUMeters = function() {
        Object.keys(SS.stemAudios).forEach(function(name) {
            SS.meterLevels[name] = 0;
            if (SS.stemAudios[name].analyser) {
                SS.analyserBuffers[name] = new Uint8Array(SS.stemAudios[name].analyser.frequencyBinCount);
            }
        });

        function getRmsLevel(name, data) {
            if (!data.analyser || !SS.analyserBuffers[name]) return 0;
            var buf = SS.analyserBuffers[name];
            data.analyser.getByteFrequencyData(buf);
            var sum = 0;
            for (var i = 0; i < buf.length; i++) {
                sum += buf[i] * buf[i];
            }
            return Math.sqrt(sum / buf.length) / 255;
        }

        function animateVU() {
            Object.entries(SS.stemAudios).forEach(function(entry) {
                var name = entry[0];
                var data = entry[1];
                var meterFill = document.getElementById('meter-fill-' + name);
                var vuNeedle = document.getElementById('vu-needle-' + name);
                var vuMeter = document.getElementById('vu-meter-' + name);

                var targetLevel = 0;

                if (SS.isPlaying && !data.muted) {
                    if (SS.soloedStems.size > 0 && !SS.soloedStems.has(name)) {
                        targetLevel = 0;
                    } else {
                        targetLevel = getRmsLevel(name, data);
                    }
                }

                if (targetLevel > SS.meterLevels[name]) {
                    SS.meterLevels[name] += (targetLevel - SS.meterLevels[name]) * 0.3;
                } else {
                    SS.meterLevels[name] += (targetLevel - SS.meterLevels[name]) * 0.08;
                }

                var percent = Math.max(0, Math.min(100, SS.meterLevels[name] * 100));

                if (meterFill) {
                    meterFill.style.height = percent + '%';
                }
                if (vuNeedle) {
                    var rotation = -45 + (percent * 0.9);
                    vuNeedle.style.transform = 'rotate(' + rotation + 'deg)';
                }
                if (vuMeter) {
                    vuMeter.classList.toggle('lit', percent > 5);
                }
                var peakLed = document.getElementById('peak-led-' + name);
                if (peakLed) {
                    peakLed.classList.toggle('active', percent > 85);
                }
            });

            SS.vuAnimationId = requestAnimationFrame(animateVU);
        }

        animateVU();
    };

    SS.setupMixerControls = function() {
        document.getElementById('masterPlayBtn').addEventListener('click', SS.togglePlayback);

        document.getElementById('masterVolume').addEventListener('input', function() {
            SS.updateStemVolumes();
        });

        // Timeline seeking (mouse + touch)
        var timeline = document.getElementById('timeline');
        var isDraggingTimeline = false;

        function seekToClientX(clientX) {
            var rect = timeline.getBoundingClientRect();
            var percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
            var seekTime = percent * SS.duration;
            Object.values(SS.stemAudios).forEach(function(s) { s.audio.currentTime = seekTime; });
            SS.updateTimeline();
        }

        timeline.addEventListener('mousedown', function(e) {
            isDraggingTimeline = true;
            seekToClientX(e.clientX);
        });

        document.addEventListener('mousemove', function(e) {
            if (isDraggingTimeline) seekToClientX(e.clientX);
        });

        document.addEventListener('mouseup', function() {
            isDraggingTimeline = false;
        });

        timeline.addEventListener('touchstart', function(e) {
            e.preventDefault();
            isDraggingTimeline = true;
            seekToClientX(e.touches[0].clientX);
        }, { passive: false });

        timeline.addEventListener('touchmove', function(e) {
            e.preventDefault();
            if (isDraggingTimeline) seekToClientX(e.touches[0].clientX);
        }, { passive: false });

        timeline.addEventListener('touchend', function() {
            isDraggingTimeline = false;
        });

        // Skip buttons
        document.getElementById('skipBackBtn').addEventListener('click', function() {
            var currentTime = Object.values(SS.stemAudios)[0]?.audio.currentTime || 0;
            var newTime = Math.max(0, currentTime - 10);
            Object.values(SS.stemAudios).forEach(function(s) { s.audio.currentTime = newTime; });
            SS.updateTimeline();
        });

        document.getElementById('skipForwardBtn').addEventListener('click', function() {
            var currentTime = Object.values(SS.stemAudios)[0]?.audio.currentTime || 0;
            var newTime = Math.min(SS.duration, currentTime + 10);
            Object.values(SS.stemAudios).forEach(function(s) { s.audio.currentTime = newTime; });
            SS.updateTimeline();
        });

        // Keyboard controls
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            // Global shortcuts (work on any screen)
            if (e.key === '?') {
                e.preventDefault();
                SS.toggleShortcutsModal();
                return;
            }
            if (e.key === 'Escape') {
                SS.hideShortcutsModal();
                return;
            }

            // Mixer shortcuts (only when results are active)
            if (!document.getElementById('resultsSection').classList.contains('active')) return;
            if (Object.keys(SS.stemAudios).length === 0) return;

            var currentTime = Object.values(SS.stemAudios)[0]?.audio.currentTime || 0;

            switch(e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    var backTime = Math.max(0, currentTime - 5);
                    Object.values(SS.stemAudios).forEach(function(s) { s.audio.currentTime = backTime; });
                    SS.updateTimeline();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    var fwdTime = Math.min(SS.duration, currentTime + 5);
                    Object.values(SS.stemAudios).forEach(function(s) { s.audio.currentTime = fwdTime; });
                    SS.updateTimeline();
                    break;
                case ' ':
                    e.preventDefault();
                    SS.togglePlayback();
                    break;
            }
        });

        // Mute buttons
        document.querySelectorAll('.mute-btn').forEach(function(btn) {
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                var stem = btn.dataset.stem;
                var stemData = SS.stemAudios[stem];
                stemData.muted = !stemData.muted;
                if (stemData.muted && window.SS_Analytics) SS_Analytics.stemMuted(stem);
                btn.classList.toggle('active', stemData.muted);
                document.getElementById('stem-' + stem).classList.toggle('muted', stemData.muted);
                SS.updateStemVolumes();
            });
        });

        // Solo buttons
        document.querySelectorAll('.solo-btn').forEach(function(btn) {
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                var stem = btn.dataset.stem;
                if (SS.soloedStems.has(stem)) {
                    SS.soloedStems.delete(stem);
                    btn.classList.remove('active');
                } else {
                    SS.soloedStems.add(stem);
                    if (window.SS_Analytics) SS_Analytics.stemSoloed(stem);
                    btn.classList.add('active');
                }
                SS.updateStemVolumes();
            });
        });

        // Volume sliders with fader cap sync
        document.querySelectorAll('.stem-volume-slider').forEach(function(slider) {
            slider.addEventListener('input', function(e) {
                var stem = slider.dataset.stem;
                SS.stemAudios[stem].volume = e.target.value / 100;
                SS.updateStemVolumes();
                SS.updateFaderCap(stem, e.target.value);
            });
        });

        // Fader cap dragging (mouse + touch)
        document.querySelectorAll('.fader-track').forEach(function(track) {
            var stemName = track.id.replace('fader-track-', '');
            var slider = track.querySelector('.stem-volume-slider');

            function updateFaderFromPosition(clientY) {
                var rect = track.getBoundingClientRect();
                var y = Math.max(0, Math.min(clientY - rect.top, rect.height));
                var percent = 100 - (y / rect.height * 100);
                slider.value = percent;
                slider.dispatchEvent(new Event('input'));
            }

            track.addEventListener('mousedown', function(e) {
                updateFaderFromPosition(e.clientY);

                var onMouseMove = function(ev) { updateFaderFromPosition(ev.clientY); };
                var onMouseUp = function() {
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                };

                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            });

            track.addEventListener('touchstart', function(e) {
                e.preventDefault();
                updateFaderFromPosition(e.touches[0].clientY);
            }, { passive: false });

            track.addEventListener('touchmove', function(e) {
                e.preventDefault();
                updateFaderFromPosition(e.touches[0].clientY);
            }, { passive: false });
        });

        // Timeline update interval
        setInterval(SS.updateTimeline, 100);

        // Start VU meters
        SS.startVUMeters();
    };

    SS.toggleMiniMute = function(stemName, btn) {
        var stemData = SS.stemAudios[stemName];
        if (stemData) {
            stemData.muted = !stemData.muted;
            if (stemData.gainNode) {
                stemData.gainNode.gain.value = stemData.muted ? 0 : stemData.volume;
            } else {
                stemData.audio.volume = stemData.muted ? 0 : stemData.volume;
            }
            btn.classList.toggle('active', stemData.muted);
            btn.parentElement.parentElement.classList.toggle('muted', stemData.muted);
        }
    };

    // Expose globally for onclick handlers in HTML
    window.toggleMiniMute = SS.toggleMiniMute;

})(window.StemScribe);
