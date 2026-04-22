// StemScriber — Wavesurfer.js Waveform Display & A-B Looping
window.StemScriber = window.StemScriber || {};

(function(SS) {
    'use strict';

    SS.masterWavesurfer = null;
    SS.regionsPlugin = null;
    SS.loopRegion = null;
    SS.loopState = 'idle'; // idle, setting-a, setting-b, active
    SS.loopPointA = null;

    SS.initWaveform = function(audioUrl) {
        var container = document.getElementById('masterWaveformContainer');
        var waveformEl = document.getElementById('masterWaveform');

        // Destroy previous instance
        if (SS.masterWavesurfer) {
            SS.masterWavesurfer.destroy();
            SS.masterWavesurfer = null;
        }

        if (!window.WaveSurfer) {
            console.log('WaveSurfer not loaded, skipping waveform');
            return;
        }

        container.style.display = 'block';

        // Get theme colors
        var isDark = !document.body.classList.contains('light-mode');
        var waveColor = isDark ? 'rgba(97, 175, 239, 0.4)' : 'rgba(60, 100, 160, 0.4)';
        var progressColor = isDark ? '#61afef' : '#3c64a0';
        var cursorColor = isDark ? '#ff6b9d' : '#d94070';

        // Create Regions plugin for A-B looping
        var regionsOpts = {};
        if (window.WaveSurfer.Regions) {
            SS.regionsPlugin = window.WaveSurfer.Regions.create(regionsOpts);
        } else if (window.RegionsPlugin) {
            SS.regionsPlugin = window.RegionsPlugin.create(regionsOpts);
        }

        var plugins = [];
        if (SS.regionsPlugin) plugins.push(SS.regionsPlugin);

        SS.masterWavesurfer = WaveSurfer.create({
            container: waveformEl,
            waveColor: waveColor,
            progressColor: progressColor,
            cursorColor: cursorColor,
            cursorWidth: 2,
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            height: 80,
            normalize: true,
            interact: true,
            hideScrollbar: true,
            plugins: plugins
        });

        // Load the audio URL (first stem, usually vocals)
        SS.masterWavesurfer.load(audioUrl);

        // Sync waveform seeks with mixer
        SS.masterWavesurfer.on('seeking', function(progress) {
            var seekTime = progress * SS.duration;
            SS.seekTo(seekTime);
            SS.updateTimeline();
        });

        // Clicking on waveform for A-B loop point setting
        SS.masterWavesurfer.on('click', function(relativeX) {
            if (SS.loopState === 'setting-a') {
                SS.loopPointA = relativeX * SS.duration;
                SS.loopState = 'setting-b';
                var loopBtn = document.getElementById('loopBtn');
                loopBtn.textContent = 'B?';
                loopBtn.classList.add('setting');
            } else if (SS.loopState === 'setting-b') {
                var pointB = relativeX * SS.duration;
                var start = Math.min(SS.loopPointA, pointB);
                var end = Math.max(SS.loopPointA, pointB);

                if (end - start < 0.5) return; // Minimum 0.5s loop

                SS.createLoopRegion(start, end);
                SS.loopState = 'active';
                var loopBtn2 = document.getElementById('loopBtn');
                if (loopBtn2) {
                    loopBtn2.textContent = 'A-B';
                    loopBtn2.classList.remove('setting');
                    loopBtn2.classList.add('active');
                }
                var lcb = document.getElementById('loopClearBtn');
                if (lcb) lcb.style.display = 'inline-block';
            }
        });

        // Don't play/pause through waveform -- we control playback via mixer
        SS.masterWavesurfer.on('play', function() {
            SS.masterWavesurfer.pause();
        });
    };

    SS.createLoopRegion = function(start, end) {
        if (!SS.regionsPlugin) return;

        // Clear existing region
        SS.clearLoopRegion();

        SS.loopRegion = SS.regionsPlugin.addRegion({
            start: start,
            end: end,
            color: 'rgba(255, 107, 157, 0.15)',
            drag: true,
            resize: true
        });

        // Start loop checking
        SS.startLoopCheck();
    };

    SS.clearLoopRegion = function() {
        if (SS.loopRegion) {
            SS.loopRegion.remove();
            SS.loopRegion = null;
        }
        SS.stopLoopCheck();
        SS.loopState = 'idle';
        SS.loopPointA = null;
        var loopBtn = document.getElementById('loopBtn');
        if (loopBtn) {
            loopBtn.textContent = 'A-B';
            loopBtn.classList.remove('active', 'setting');
        }
        var clearBtn = document.getElementById('loopClearBtn');
        if (clearBtn) clearBtn.style.display = 'none';
    };

    var loopCheckInterval = null;

    SS.startLoopCheck = function() {
        SS.stopLoopCheck();
        loopCheckInterval = setInterval(function() {
            if (!SS.loopRegion || !SS.isPlaying) return;
            var currentTime = SS.getPlaybackTime();
            if (currentTime >= SS.loopRegion.end) {
                SS.seekTo(SS.loopRegion.start);
            }
        }, 50);
    };

    SS.stopLoopCheck = function() {
        if (loopCheckInterval) {
            clearInterval(loopCheckInterval);
            loopCheckInterval = null;
        }
    };

    SS.updateWaveformProgress = function() {
        if (!SS.duration) return;
        var progress = SS.getPlaybackTime() / SS.duration;
        var clampedProgress = Math.min(1, Math.max(0, progress));

        // Update master waveform cursor
        if (SS.masterWavesurfer) {
            SS.masterWavesurfer.seekTo(clampedProgress);
        }

        // Update per-stem waveform cursors
        if (SS.stemWavesurfers) {
            Object.values(SS.stemWavesurfers).forEach(function(ws) {
                if (ws && !ws.isDestroyed) {
                    ws.seekTo(clampedProgress);
                }
            });
        }
    };

    SS.updateWaveformTheme = function(isDark) {
        if (SS.masterWavesurfer) {
            SS.masterWavesurfer.setOptions({
                waveColor: isDark ? 'rgba(97, 175, 239, 0.4)' : 'rgba(60, 100, 160, 0.4)',
                progressColor: isDark ? '#61afef' : '#3c64a0',
                cursorColor: isDark ? '#ff6b9d' : '#d94070'
            });
        }

        // Update per-stem waveform themes
        if (SS.stemWavesurfers) {
            Object.entries(SS.stemWavesurfers).forEach(function(entry) {
                var name = entry[0];
                var ws = entry[1];
                if (!ws || ws.isDestroyed) return;
                var cfg = SS.stemConfig[name] || {};
                var color = cfg.color || '#888';
                ws.setOptions({
                    waveColor: isDark ? SS._hexToRgba(color, 0.4) : SS._hexToRgba(color, 0.5),
                    progressColor: color,
                    cursorColor: isDark ? '#ff6b9d' : '#d94070'
                });
            });
        }
    };

    // Per-stem waveform instances
    SS.stemWavesurfers = {};

    SS._hexToRgba = function(hex, alpha) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
    };

    SS.initStemWaveforms = function(job, sortedStems) {
        if (!window.WaveSurfer) return;

        // Destroy previous stem wavesurfers
        SS.destroyStemWaveforms();

        var isDark = !document.body.classList.contains('light-mode');

        sortedStems.forEach(function(entry) {
            var name = entry[0];
            var container = document.getElementById('stem-waveform-' + name);
            if (!container) return;

            var cfg = SS.stemConfig[name] || { color: '#888' };
            var color = cfg.color || '#888';

            var audioUrl = job.enhanced_stems?.[name]
                ? SS.API_BASE + '/download/' + job.job_id + '/enhanced/' + name
                : SS.API_BASE + '/download/' + job.job_id + '/stem/' + name;

            var ws = WaveSurfer.create({
                container: container,
                waveColor: isDark ? SS._hexToRgba(color, 0.4) : SS._hexToRgba(color, 0.5),
                progressColor: color,
                cursorColor: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.4)',
                cursorWidth: 1,
                barWidth: 1,
                barGap: 1,
                barRadius: 1,
                height: 40,
                normalize: true,
                interact: false,
                hideScrollbar: true
            });

            // Load audio just for waveform visualization (no playback)
            ws.load(audioUrl);

            // Prevent playback from waveform -- playback is managed by the mixer
            ws.on('play', function() {
                ws.pause();
            });

            SS.stemWavesurfers[name] = ws;
        });
    };

    /**
     * Initialize stem waveforms using pre-computed peaks from the backend.
     * No audio loading = no AudioContext = no playback conflicts.
     */
    SS.initStemWaveformsFromPeaks = function(job, sortedStems) {
        if (!window.WaveSurfer) return;
        SS.destroyStemWaveforms();

        var isDark = !document.body.classList.contains('light-mode');

        sortedStems.forEach(function(entry) {
            var name = entry[0];
            var container = document.getElementById('stem-waveform-' + name);
            if (!container) return;

            var cfg = SS.stemConfig[name] || { color: '#888' };
            var color = cfg.color || '#888';

            var ws = WaveSurfer.create({
                container: container,
                waveColor: isDark ? SS._hexToRgba(color, 0.4) : SS._hexToRgba(color, 0.5),
                progressColor: color,
                cursorColor: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.4)',
                cursorWidth: 1,
                barWidth: 1,
                barGap: 1,
                barRadius: 1,
                height: 40,
                normalize: true,
                interact: false,
                hideScrollbar: true
            });

            SS.stemWavesurfers[name] = ws;

            // Fetch pre-computed peaks from backend (no audio decoding needed)
            fetch(SS.API_BASE + '/peaks/' + job.job_id + '/' + name)
                .then(function(resp) { return resp.ok ? resp.json() : null; })
                .then(function(data) {
                    if (data && data.peaks) {
                        ws.load('', [data.peaks], data.duration);
                    }
                })
                .catch(function(err) {
                    console.log('Peaks load failed for ' + name + ':', err.message);
                });
        });
    };

    SS.destroyStemWaveforms = function() {
        Object.values(SS.stemWavesurfers).forEach(function(ws) {
            if (ws && !ws.isDestroyed) {
                try { ws.destroy(); } catch(e) {}
            }
        });
        SS.stemWavesurfers = {};
    };

    SS.initLoopControls = function() {
        var loopBtn = document.getElementById('loopBtn');
        var loopClearBtn = document.getElementById('loopClearBtn');

        if (loopBtn) {
            loopBtn.addEventListener('click', function() {
                if (SS.loopState === 'idle') {
                    SS.loopState = 'setting-a';
                    loopBtn.textContent = 'A?';
                    loopBtn.classList.add('setting');
                } else if (SS.loopState === 'active') {
                    // Toggle loop off/on
                    SS.clearLoopRegion();
                } else {
                    // Cancel setting
                    SS.loopState = 'idle';
                    loopBtn.textContent = 'A-B';
                    loopBtn.classList.remove('setting');
                }
            });
        }

        if (loopClearBtn) {
            loopClearBtn.addEventListener('click', function() {
                SS.clearLoopRegion();
            });
        }
    };

})(window.StemScriber);
