// StemScribe — Mixer / Web Audio Engine / VU Meters / Playback
// Uses Web Audio API AudioBufferSourceNode for iOS Safari multi-track support.
// NO <audio> elements — all stems play through a single AudioContext.
window.StemScribe = window.StemScribe || {};

// ═══════════════════════════════════════════════════════════
// iOS Audio Session Unlock (shared across mixer + practice)
// iOS Safari plays Web Audio through the "ambient" category by default.
// When the physical silent/ringer switch is on, ambient audio is muted.
// Playing a short HTMLAudioElement on first user interaction forces iOS
// to switch to the "playback" audio category, which ignores the silent switch.
// Also warms up the AudioContext with a silent buffer.
// ═══════════════════════════════════════════════════════════
(function() {
    if (window.__iOSAudioUnlocked) return; // prevent double-binding if both files load
    var _unlocked = false;
    var _silentEl = null;

    function initIOSAudioSession() {
        if (_unlocked) return;
        _unlocked = true;
        window.__iOSAudioUnlocked = true;
        console.log('[iOS Audio] initIOSAudioSession — unlocking on first user interaction (mixer)');

        // Strategy 1: Modern Audio Session API (iOS 16.4+)
        // Sets audio category to "playback" which ignores the silent switch.
        if ('audioSession' in navigator) {
            try {
                navigator.audioSession.type = 'playback';
                console.log('[iOS Audio] navigator.audioSession.type set to "playback"');
            } catch (e) {
                console.log('[iOS Audio] audioSession.type set failed:', e.message);
            }
        }

        // Strategy 2: Silent looping MP3 via HTMLAudioElement (legacy fallback)
        // Forces iOS from "ambient" to "playback" audio channel.
        try {
            _silentEl = new Audio('data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA/+M4wAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAAbAAqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq1dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV////////////////////////////////////////////AAAAAExhdmM1OC4xMwAAAAAAAAAAAAAAACQDkAAAAAAAAAGw9wrNaQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/+MYxAAAAANIAAAAAExBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV/+MYxDsAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV/+MYxHYAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV');
            _silentEl.setAttribute('x-webkit-airplay', 'deny');
            _silentEl.setAttribute('playsinline', '');
            _silentEl.loop = true;
            _silentEl.play().then(function() {
                console.log('[iOS Audio] Silent looping MP3 playing — audio channel forced to playback');
            }).catch(function(e) {
                console.log('[iOS Audio] Silent MP3 play failed (expected on non-iOS):', e.message);
            });
        } catch (e) {}

        // Resume any existing AudioContext
        var SS = window.StemScribe;
        var ctx = (SS && SS.audioEngine && SS.audioEngine._ctx) ? SS.audioEngine._ctx : null;
        if (ctx && ctx.state === 'suspended') {
            ctx.resume().then(function() {
                console.log('[iOS Audio] AudioContext resumed from unlock handler');
            }).catch(function() {});
        }

        // Warm up AudioContext with a 1-sample silent buffer
        if (ctx) {
            try {
                var buf = ctx.createBuffer(1, 1, 22050);
                var src = ctx.createBufferSource();
                src.buffer = buf;
                src.connect(ctx.destination);
                src.start(0);
                console.log('[iOS Audio] Silent buffer played — AudioContext warmed up');
            } catch (e) {}
        }
    }

    // Clean up silent audio when page is backgrounded to save battery
    document.addEventListener('visibilitychange', function() {
        if (document.hidden && _silentEl) {
            _silentEl.pause();
            console.log('[iOS Audio] Paused silent MP3 (page hidden)');
        } else if (!document.hidden && _silentEl && _unlocked) {
            _silentEl.play().catch(function() {});
            console.log('[iOS Audio] Resumed silent MP3 (page visible)');
        }
    });

    document.addEventListener('touchend', initIOSAudioSession, { capture: true, passive: true });
    document.addEventListener('click', initIOSAudioSession, { capture: true });
})();

(function(SS) {
    'use strict';

    // ═══════════════════════════════════════════════════════════
    // StemAudioEngine — Web Audio API multi-track playback engine
    // ═══════════════════════════════════════════════════════════

    function StemAudioEngine() {
        this._ctx = null;
        this._stems = {};          // { name: { buffer, sourceNode, gainNode, analyser } }
        this._playing = false;
        this._startedAt = 0;       // ctx.currentTime when playback started
        this._offset = 0;          // offset into buffer when playback started
        this._duration = 0;
        this._playbackRate = 1.0;
        this._loaded = false;
        this._endTimer = null;
        this._unlocked = false; // iOS silent-buffer unlock flag
    }

    StemAudioEngine.prototype.init = function() {
        // Set audio session to "playback" before creating/resuming AudioContext
        if ('audioSession' in navigator) {
            try { navigator.audioSession.type = 'playback'; } catch(e) {}
        }
        var created = false;
        if (!this._ctx || this._ctx.state === 'closed') {
            this._ctx = new (window.AudioContext || window.webkitAudioContext)();
            created = true;
            console.log('[StemAudio:Mixer] Created AudioContext, state:', this._ctx.state, 'sampleRate:', this._ctx.sampleRate);

            // Context was recreated (e.g., iOS closed it while suspended).
            // All gain/analyser nodes from the old context are orphaned — rebuild them.
            if (Object.keys(this._stems).length > 0) {
                console.log('[StemAudio:Mixer] Rebuilding audio nodes for new context');
                this._rebuildNodes();
            }
        }
        // iOS Safari: must resume in user gesture call stack
        if (this._ctx.state === 'suspended') {
            console.log('[StemAudio:Mixer] Resuming suspended AudioContext');
            this._ctx.resume().catch(function(e) {
                console.warn('[StemAudio:Mixer] Resume failed:', e);
            });
        }

        // iOS Safari unlock: play a short silent buffer to truly unlock audio output.
        // Some iOS versions report "running" but won't output sound until a buffer
        // has been played inside a user gesture call stack.
        if (!this._unlocked) {
            try {
                var silentBuf = this._ctx.createBuffer(1, 1, this._ctx.sampleRate);
                var silentSrc = this._ctx.createBufferSource();
                silentSrc.buffer = silentBuf;
                silentSrc.connect(this._ctx.destination);
                silentSrc.start(0);
                this._unlocked = true;
                console.log('[StemAudio:Mixer] iOS unlock: played silent buffer');
            } catch (e) {
                console.warn('[StemAudio:Mixer] iOS unlock failed:', e);
            }
        }

        // Safari non-standard "interrupted" state (e.g., phone call, Siri)
        // Only add listener once per AudioContext instance
        if (created) {
            var self = this;
            this._ctx.addEventListener('statechange', function() {
                if (self._ctx.state === 'interrupted') {
                    // Context was interrupted — mark not playing so UI is correct
                    if (self._playing) {
                        self._offset = self.getCurrentTime();
                        self._playing = false;
                        var btn = document.getElementById('masterPlayBtn');
                        if (btn) btn.textContent = '\u25B6';
                        SS.isPlaying = false;
                    }
                } else if (self._ctx.state === 'running' && !self._playing) {
                    // Context resumed after interruption — don't auto-play
                }
            });
        }

        return this._ctx;
    };

    /**
     * Rebuild gain + analyser nodes when the AudioContext has been recreated.
     * Preserves existing audio buffers (they are context-independent ArrayBuffers
     * once decoded, but AudioBuffer objects ARE tied to a context — however,
     * they can still be used with createBufferSource on any context).
     */
    StemAudioEngine.prototype._rebuildNodes = function() {
        var self = this;
        Object.entries(this._stems).forEach(function(entry) {
            var name = entry[0];
            var stem = entry[1];

            // Disconnect old nodes (may throw if already dead — that's fine)
            if (stem.gainNode) try { stem.gainNode.disconnect(); } catch(e) {}
            if (stem.analyser) try { stem.analyser.disconnect(); } catch(e) {}

            // Create new nodes on the new context
            var gainNode = self._ctx.createGain();
            var analyser = self._ctx.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.8;
            gainNode.connect(analyser);
            analyser.connect(self._ctx.destination);

            stem.gainNode = gainNode;
            stem.analyser = analyser;
            stem.sourceNode = null; // old source is dead
        });
        console.log('[StemAudio:Mixer] Rebuilt', Object.keys(this._stems).length, 'stem node chains');
    };

    StemAudioEngine.prototype.getContext = function() {
        return this._ctx;
    };

    /**
     * Load stems from URLs. Fetches and decodes each stem sequentially
     * to avoid blocking the main thread on Safari.
     * @param {Object} stemUrls - { name: url, ... }
     * @param {Function} onProgress - callback(loaded, total)
     * @returns {Promise}
     */
    StemAudioEngine.prototype.loadStems = function(stemUrls, onProgress) {
        var self = this;
        var names = Object.keys(stemUrls);
        var total = names.length;
        var loaded = 0;
        var failed = 0;
        var failedStems = [];

        // Ensure context exists for decoding (may be suspended on iOS — that's OK)
        if (!this._ctx || this._ctx.state === 'closed') {
            this._ctx = new (window.AudioContext || window.webkitAudioContext)();
            console.log('[StemAudio:Mixer] Created AudioContext for decoding, state:', this._ctx.state, 'sampleRate:', this._ctx.sampleRate);
        }

        // Load a single stem with full error isolation — failure does NOT break the chain
        function loadOneStem(name) {
            console.log('[StemAudio:Mixer] Fetching stem:', name);
            return fetch(stemUrls[name])
                .then(function(resp) {
                    if (!resp.ok) throw new Error('HTTP ' + resp.status + ' for ' + name);
                    var contentLength = resp.headers.get('content-length');
                    console.log('[StemAudio:Mixer] Fetched:', name, 'status:', resp.status, 'content-length:', contentLength || 'unknown');
                    return resp.arrayBuffer();
                })
                .then(function(arrayBuf) {
                    if (!arrayBuf || arrayBuf.byteLength === 0) {
                        throw new Error('Empty response body for ' + name);
                    }
                    console.log('[StemAudio:Mixer] Decoding:', name, 'arrayBuffer:', arrayBuf.byteLength, 'bytes');
                    return self._decodeAudio(arrayBuf, name);
                })
                .then(function(audioBuffer) {
                    // Validate decoded buffer
                    if (!audioBuffer || audioBuffer.duration === 0 || audioBuffer.length === 0) {
                        throw new Error('Decoded buffer is empty for ' + name);
                    }

                    // Check for all-silent buffer (warn only, don't reject)
                    var channelData = audioBuffer.getChannelData(0);
                    var hasSignal = false;
                    var checkLen = Math.min(channelData.length, 44100); // first second
                    for (var i = 0; i < checkLen; i += 100) {
                        if (Math.abs(channelData[i]) > 0.0001) { hasSignal = true; break; }
                    }
                    if (!hasSignal) {
                        console.warn('[StemAudio:Mixer] WARNING: "' + name + '" appears silent (first second is all zeros)');
                    }

                    console.log('[StemAudio:Mixer] Decoded OK:', name,
                        'dur:', audioBuffer.duration.toFixed(2) + 's',
                        'ch:', audioBuffer.numberOfChannels,
                        'rate:', audioBuffer.sampleRate,
                        'samples:', audioBuffer.length);

                    // Create gain + analyser nodes (persistent — not recreated on play)
                    var gainNode = self._ctx.createGain();
                    var analyser = self._ctx.createAnalyser();
                    analyser.fftSize = 256;
                    analyser.smoothingTimeConstant = 0.8;
                    gainNode.connect(analyser);
                    analyser.connect(self._ctx.destination);

                    self._stems[name] = {
                        buffer: audioBuffer,
                        sourceNode: null,
                        gainNode: gainNode,
                        analyser: analyser
                    };

                    // Track max duration
                    if (audioBuffer.duration > self._duration) {
                        self._duration = audioBuffer.duration;
                    }

                    loaded++;
                    if (onProgress) onProgress(loaded, total);
                })
                .catch(function(err) {
                    // Error isolation: log failure, continue loading remaining stems
                    failed++;
                    failedStems.push(name);
                    console.error('[StemAudio:Mixer] FAILED stem "' + name + '":', err.message || err);
                    // Still count toward progress so the UI reflects this stem was attempted
                    loaded++;
                    if (onProgress) onProgress(loaded, total);
                });
        }

        // Sequential loading to avoid Safari decode bottleneck
        var chain = Promise.resolve();
        names.forEach(function(name) {
            chain = chain.then(function() {
                return loadOneStem(name);
            });
        });

        return chain.then(function() {
            // Mark loaded if at least 1 stem succeeded — partial playback beats no playback
            var succeeded = Object.keys(self._stems).length;
            if (succeeded > 0) {
                self._loaded = true;
                console.log('[StemAudio:Mixer] Loading complete:', succeeded + '/' + total, 'stems loaded');
                if (failed > 0) {
                    console.warn('[StemAudio:Mixer] Failed stems (' + failed + '):', failedStems.join(', '));
                }
            } else {
                console.error('[StemAudio:Mixer] ALL stems failed to load (' + total + ' attempted)');
                throw new Error('All ' + total + ' stems failed to load');
            }
        });
    };

    /**
     * Decode audio data — handles Safari's callback-only decodeAudioData.
     * iOS Safari can silently fail on first decode attempt, so we retry once
     * with a fresh ArrayBuffer copy if the first attempt fails.
     * @param {ArrayBuffer} arrayBuffer
     * @param {string} stemName - for logging
     */
    StemAudioEngine.prototype._decodeAudio = function(arrayBuffer, stemName) {
        var ctx = this._ctx;
        var label = stemName || 'unknown';

        function attemptDecode(buf) {
            return new Promise(function(resolve, reject) {
                var settled = false;
                function onDone(decoded) { if (!settled) { settled = true; resolve(decoded); } }
                function onFail(err) { if (!settled) { settled = true; reject(err || new Error('decodeAudioData failed')); } }
                try {
                    var promise = ctx.decodeAudioData(buf, onDone, onFail);
                    if (promise && typeof promise.then === 'function') {
                        promise.then(onDone).catch(onFail);
                    }
                } catch (e) {
                    onFail(e);
                }
            });
        }

        // First attempt
        return attemptDecode(arrayBuffer).catch(function(err) {
            // iOS Safari: decodeAudioData consumes the ArrayBuffer, making retry
            // impossible unless we copy it first. Since we're here on first failure,
            // we can't retry with the same buffer. Log detailed error info.
            console.error('[StemAudio:Mixer] decodeAudioData failed for "' + label + '":', err.message || err,
                '| buffer size:', arrayBuffer.byteLength,
                '| ctx.state:', ctx.state,
                '| ctx.sampleRate:', ctx.sampleRate);
            throw err;
        });
    };

    /**
     * Create and start AudioBufferSourceNode for each stem.
     * Source nodes are one-shot: created fresh each time play is called.
     */
    StemAudioEngine.prototype.play = function() {
        if (this._playing || !this._loaded) return;

        var self = this;
        console.log('[StemAudio:Mixer] play() called, ctx.state:', this._ctx ? this._ctx.state : 'null');

        // Safety: if context was closed (iOS can do this), rebuild via init()
        if (!this._ctx || this._ctx.state === 'closed') {
            console.warn('[StemAudio:Mixer] Context closed before play — reinitializing');
            this.init();
        }

        function startAllSources() {
            // Verify gain nodes are connected to the CURRENT context's destination.
            // If init() rebuilt the context, _rebuildNodes() already handled this,
            // but catch any edge case where nodes belong to a stale context.
            var firstStem = Object.values(self._stems)[0];
            if (firstStem && firstStem.gainNode && firstStem.gainNode.context !== self._ctx) {
                console.warn('[StemAudio:Mixer] Gain nodes bound to stale context — rebuilding');
                self._rebuildNodes();
            }

            var offset = self._offset;

            // Clamp offset to valid range
            if (offset >= self._duration) offset = 0;
            if (offset < 0) offset = 0;
            self._offset = offset;

            var started = 0;
            Object.entries(self._stems).forEach(function(entry) {
                var name = entry[0];
                var stem = entry[1];

                // Disconnect any existing source (shouldn't exist, but safety)
                if (stem.sourceNode) {
                    try { stem.sourceNode.disconnect(); } catch(e) {}
                    stem.sourceNode = null;
                }

                var source = self._ctx.createBufferSource();
                source.buffer = stem.buffer;
                source.playbackRate.value = self._playbackRate;
                source.connect(stem.gainNode);

                // Start from offset
                source.start(0, offset);
                stem.sourceNode = source;
                started++;

                // Handle natural end of buffer
                source.onended = function() {
                    if (self._playing) {
                        // Check if we've truly reached the end (not a seek/stop)
                        var currentPos = self.getCurrentTime();
                        if (currentPos >= self._duration - 0.15) {
                            // Loop back to start
                            self._offset = 0;
                            self._playing = false;
                            self.play();
                        }
                    }
                };
            });

            self._startedAt = self._ctx.currentTime;
            self._playing = true;
            console.log('[StemAudio:Mixer] Started', started, 'sources at offset', offset.toFixed(2), 'ctx.state:', self._ctx.state);
        }

        // iOS Safari: context may be suspended — must await resume before starting sources
        if (this._ctx.state === 'suspended' || this._ctx.state === 'interrupted') {
            console.log('[StemAudio:Mixer] Context suspended, resuming before play...');
            this._ctx.resume().then(function() {
                console.log('[StemAudio:Mixer] Context resumed, state:', self._ctx.state);
                startAllSources();
            }).catch(function(e) {
                console.error('[StemAudio:Mixer] Resume failed in play():', e);
                // Try starting anyway
                startAllSources();
            });
            return;
        }

        startAllSources();
    };

    StemAudioEngine.prototype.pause = function() {
        if (!this._playing) return;

        // Capture current position before stopping
        this._offset = this.getCurrentTime();
        this._playing = false;

        // Stop all source nodes
        Object.values(this._stems).forEach(function(stem) {
            if (stem.sourceNode) {
                try { stem.sourceNode.stop(); } catch(e) {}
                try { stem.sourceNode.disconnect(); } catch(e) {}
                stem.sourceNode = null;
            }
        });
    };

    StemAudioEngine.prototype.seek = function(time) {
        time = Math.max(0, Math.min(time, this._duration));

        if (this._playing) {
            // Stop current sources, update offset, restart
            Object.values(this._stems).forEach(function(stem) {
                if (stem.sourceNode) {
                    try { stem.sourceNode.stop(); } catch(e) {}
                    try { stem.sourceNode.disconnect(); } catch(e) {}
                    stem.sourceNode = null;
                }
            });
            this._playing = false;
            this._offset = time;
            this.play();
        } else {
            this._offset = time;
        }
    };

    StemAudioEngine.prototype.getCurrentTime = function() {
        if (!this._playing) return this._offset;
        var elapsed = (this._ctx.currentTime - this._startedAt) * this._playbackRate;
        var pos = this._offset + elapsed;
        return Math.min(pos, this._duration);
    };

    StemAudioEngine.prototype.getDuration = function() {
        return this._duration;
    };

    StemAudioEngine.prototype.isPlaying = function() {
        return this._playing;
    };

    StemAudioEngine.prototype.isLoaded = function() {
        return this._loaded;
    };

    StemAudioEngine.prototype.setVolume = function(stemName, vol) {
        var stem = this._stems[stemName];
        if (stem && stem.gainNode) {
            // Use setTargetAtTime to avoid clicks
            stem.gainNode.gain.setTargetAtTime(vol, this._ctx.currentTime, 0.015);
        }
    };

    StemAudioEngine.prototype.setPlaybackRate = function(rate) {
        if (this._playing) {
            // Capture current position at old rate before changing
            this._offset = this.getCurrentTime();
            this._startedAt = this._ctx.currentTime;
        }
        this._playbackRate = rate;
        // Update all active source nodes
        Object.values(this._stems).forEach(function(stem) {
            if (stem.sourceNode) {
                stem.sourceNode.playbackRate.value = rate;
            }
        });
    };

    StemAudioEngine.prototype.getAnalyser = function(stemName) {
        var stem = this._stems[stemName];
        return stem ? stem.analyser : null;
    };

    StemAudioEngine.prototype.getGainNode = function(stemName) {
        var stem = this._stems[stemName];
        return stem ? stem.gainNode : null;
    };

    StemAudioEngine.prototype.getStemNames = function() {
        return Object.keys(this._stems);
    };

    StemAudioEngine.prototype.hasStem = function(stemName) {
        return !!this._stems[stemName];
    };

    StemAudioEngine.prototype.dispose = function() {
        this.pause();
        var self = this;
        Object.values(this._stems).forEach(function(stem) {
            if (stem.gainNode) try { stem.gainNode.disconnect(); } catch(e) {}
            if (stem.analyser) try { stem.analyser.disconnect(); } catch(e) {}
        });
        this._stems = {};
        this._loaded = false;
        this._duration = 0;
        this._offset = 0;
        if (this._ctx && this._ctx.state !== 'closed') {
            this._ctx.close().catch(function() {});
        }
        this._ctx = null;
    };

    // ── Audio Debug Diagnostic (activated by ?audioDebug=1 or window.audioDebugMixer()) ──
    StemAudioEngine.prototype.debug = function() {
        var self = this;
        var lines = [];
        var isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
        lines.push('=== StemScribe Audio Debug (Mixer) ===');
        lines.push('Timestamp: ' + new Date().toISOString());
        lines.push('');
        lines.push('-- Platform --');
        lines.push('userAgent: ' + navigator.userAgent);
        lines.push('iOS detected: ' + isIOS);
        lines.push('platform: ' + navigator.platform);
        lines.push('maxTouchPoints: ' + navigator.maxTouchPoints);
        lines.push('window.AudioContext: ' + (typeof window.AudioContext));
        lines.push('window.webkitAudioContext: ' + (typeof window.webkitAudioContext));
        lines.push('');
        lines.push('-- AudioContext --');
        if (!this._ctx) {
            lines.push('ctx: NULL');
        } else {
            lines.push('ctx.state: ' + this._ctx.state);
            lines.push('ctx.sampleRate: ' + this._ctx.sampleRate);
            lines.push('ctx.currentTime: ' + this._ctx.currentTime.toFixed(4));
            lines.push('ctx.baseLatency: ' + (this._ctx.baseLatency !== undefined ? this._ctx.baseLatency : 'N/A'));
            lines.push('ctx.outputLatency: ' + (this._ctx.outputLatency !== undefined ? this._ctx.outputLatency : 'N/A'));
            lines.push('ctx.destination.channelCount: ' + this._ctx.destination.channelCount);
            lines.push('ctx.destination.maxChannelCount: ' + this._ctx.destination.maxChannelCount);
        }
        lines.push('');
        lines.push('-- Engine State --');
        lines.push('_playing: ' + this._playing);
        lines.push('_loaded: ' + this._loaded);
        lines.push('_offset: ' + this._offset.toFixed(4));
        lines.push('_startedAt: ' + this._startedAt.toFixed(4));
        lines.push('_playbackRate: ' + this._playbackRate);
        lines.push('_duration: ' + this._duration.toFixed(2));
        lines.push('getCurrentTime(): ' + this.getCurrentTime().toFixed(4));
        lines.push('');
        lines.push('-- Stems (' + Object.keys(this._stems).length + ') --');
        var stemNames = Object.keys(this._stems);
        for (var si = 0; si < stemNames.length; si++) {
            var name = stemNames[si];
            var stem = this._stems[name];
            lines.push('  [' + name + ']');
            if (stem.gainNode) {
                lines.push('    gainNode.gain.value: ' + stem.gainNode.gain.value.toFixed(4));
            } else {
                lines.push('    gainNode: NULL');
            }
            if (stem.sourceNode) {
                lines.push('    sourceNode: active, playbackRate: ' + stem.sourceNode.playbackRate.value);
            } else {
                lines.push('    sourceNode: null');
            }
            if (stem.buffer) {
                lines.push('    buffer.duration: ' + stem.buffer.duration.toFixed(2) + 's');
                lines.push('    buffer.numberOfChannels: ' + stem.buffer.numberOfChannels);
                lines.push('    buffer.sampleRate: ' + stem.buffer.sampleRate);
                lines.push('    buffer.length: ' + stem.buffer.length);
                var ch0 = stem.buffer.getChannelData(0);
                var hasNZ = false, mxA = 0;
                for (var ci = 0; ci < Math.min(100, ch0.length); ci++) {
                    var av = Math.abs(ch0[ci]);
                    if (av > 0) hasNZ = true;
                    if (av > mxA) mxA = av;
                }
                lines.push('    first100.hasNonZero: ' + hasNZ + ', maxAbs: ' + mxA.toFixed(6));
            } else {
                lines.push('    buffer: NULL');
            }
            if (stem.analyser) {
                lines.push('    analyser: present, fftSize: ' + stem.analyser.fftSize);
            }
            lines.push('');
        }
        lines.push('-- Audio Graph --');
        if (this._ctx) {
            for (var gi = 0; gi < stemNames.length; gi++) {
                var gn = stemNames[gi];
                var gs = this._stems[gn];
                var chain = gn + ': ';
                chain += gs.sourceNode ? 'BufferSource -> ' : '(no source) -> ';
                chain += gs.gainNode ? 'GainNode(' + gs.gainNode.gain.value.toFixed(3) + ') -> ' : '(no gain) -> ';
                chain += gs.analyser ? 'Analyser -> ' : '';
                chain += 'destination';
                lines.push('  ' + chain);
            }
        }
        var output = lines.join('\n');
        console.log(output);
        return output;
    };

    // Expose engine class
    SS.StemAudioEngine = StemAudioEngine;

    // ═══════════════════════════════════════════════════════════
    // Global engine instance
    // ═══════════════════════════════════════════════════════════
    SS.audioEngine = null;

    // ── MediaElement bridge ──────────────────────────────────────────────────
    // connectStemToWebAudio(name, audioEl) — wire an <audio> element into the
    // Web Audio graph. Called by results.js for every stem card.
    // Returns { source, gainNode, analyser } as expected by SS.stemAudios.
    SS.connectStemToWebAudio = function(name, audioEl) {
        if (!SS.mixerAudioCtx || SS.mixerAudioCtx.state === "closed") {
            SS.mixerAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        var ctx = SS.mixerAudioCtx;
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

    // Helper: get current playback time (works whether engine exists or not)
    SS.getPlaybackTime = function() {
        if (SS.audioEngine) return SS.audioEngine.getCurrentTime();
        var first = Object.values(SS.stemAudios)[0];
        return (first && first.audio) ? first.audio.currentTime : 0;
    };

    // Helper: seek all stems
    SS.seekTo = function(time) {
        if (SS.audioEngine && SS.audioEngine.isLoaded()) {
            SS.audioEngine.seek(time);
        } else {
            Object.values(SS.stemAudios).forEach(function(s) { if (s.audio) s.audio.currentTime = time; });
        }
    };

    // ═══════════════════════════════════════════════════════════
    // Volume / Mute / Solo
    // ═══════════════════════════════════════════════════════════

    SS.updateStemVolumes = function() {
        var masterVolEl = document.getElementById('masterVolume');
        var masterVol = masterVolEl ? masterVolEl.value / 100 : 0.8;
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
            // Always use Web Audio gain node
            if (SS.audioEngine) {
                SS.audioEngine.setVolume(name, vol);
            }
        });
    };

    // ═══════════════════════════════════════════════════════════
    // Playback control
    // ═══════════════════════════════════════════════════════════

    SS.togglePlayback = function() {
        var btn = document.getElementById('masterPlayBtn');

        if (SS.isPlaying) {
            // ── PAUSE ──
            if (SS.audioEngine && SS.audioEngine.isLoaded()) {
                SS.audioEngine.pause();
            } else {
                // Fallback: pause <audio> elements (MediaElement path)
                Object.values(SS.stemAudios).forEach(function(s) { if (s.audio) s.audio.pause(); });
                if (SS.mixerAudioCtx && SS.mixerAudioCtx.state === 'running') SS.mixerAudioCtx.suspend();
            }
            if (btn) btn.textContent = '\u25B6';
            SS.isPlaying = false;
        } else {
            // ── PLAY ──
            if (SS.audioEngine && SS.audioEngine.isLoaded()) {
                // StemAudioEngine path
                SS.audioEngine.init();
                console.log('[StemAudio:Mixer] togglePlayback: ctx.state after init():', SS.audioEngine.getContext() ? SS.audioEngine.getContext().state : 'null');
                SS.audioEngine.play();
            } else {
                // Fallback: play via <audio> elements (MediaElement path)
                var stems = Object.values(SS.stemAudios);
                if (stems.length === 0) { console.warn('No stems loaded'); return; }
                var currentTime = stems[0].audio ? stems[0].audio.currentTime : 0;
                if (SS.mixerAudioCtx && SS.mixerAudioCtx.state === 'suspended') SS.mixerAudioCtx.resume();
                stems.forEach(function(s) {
                    if (s.audio) {
                        s.audio.currentTime = currentTime;
                        s.audio.play().catch(function(e) { console.warn('Playback blocked:', e); });
                    }
                });
            }
            if (btn) btn.textContent = '\u23F8';
            SS.isPlaying = true;
        }
    };

    // ═══════════════════════════════════════════════════════════
    // Timeline / time display
    // ═══════════════════════════════════════════════════════════

    SS.updateTimeline = function() {
        if (Object.keys(SS.stemAudios).length === 0) return;

        var currentTime, dur;
        if (SS.audioEngine && SS.audioEngine.isLoaded()) {
            currentTime = SS.audioEngine.getCurrentTime();
            dur = SS.duration || SS.audioEngine.getDuration();
        } else {
            var firstStem = Object.values(SS.stemAudios)[0];
            if (!firstStem || !firstStem.audio) return;
            currentTime = firstStem.audio.currentTime;
            dur = SS.duration || firstStem.audio.duration || 0;
        }
        var percent = dur > 0 ? (currentTime / dur) * 100 : 0;

        var tlProg = document.getElementById('timelineProgress');
        if (tlProg) tlProg.style.width = percent + '%';
        var tlHandle = document.getElementById('timelineHandle');
        if (tlHandle) tlHandle.style.left = percent + '%';
        var ctEl = document.getElementById('currentTime');
        if (ctEl) ctEl.textContent = SS.formatTime(currentTime);

        // Sync waveform cursor
        if (SS.updateWaveformProgress) {
            SS.updateWaveformProgress();
        }
    };

    SS.updateFaderCap = function(stem, value) {
        var cap = document.getElementById('fader-cap-' + stem);
        if (cap) {
            var position = 8 + (100 - value) * 0.8;
            cap.style.top = position + 'px';
        }
    };

    // ═══════════════════════════════════════════════════════════
    // VU Meter Animation
    // ═══════════════════════════════════════════════════════════

    SS.startVUMeters = function() {
        Object.keys(SS.stemAudios).forEach(function(name) {
            SS.meterLevels[name] = 0;
            var analyser = (SS.audioEngine && SS.audioEngine.isLoaded()) ? SS.audioEngine.getAnalyser(name) : (SS.stemAudios[name] && SS.stemAudios[name].analyser) || null;
            if (analyser) {
                SS.analyserBuffers[name] = new Uint8Array(analyser.frequencyBinCount);
            }
        });

        function getRmsLevel(name) {
            var analyser = (SS.audioEngine && SS.audioEngine.isLoaded()) ? SS.audioEngine.getAnalyser(name) : (SS.stemAudios[name] && SS.stemAudios[name].analyser) || null;
            if (!analyser) return 0;
            if (!SS.analyserBuffers[name]) {
                SS.analyserBuffers[name] = new Uint8Array(analyser.frequencyBinCount);
            }
            var buf = SS.analyserBuffers[name];
            analyser.getByteFrequencyData(buf);
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
                        targetLevel = getRmsLevel(name);
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

    // ═══════════════════════════════════════════════════════════
    // Mixer controls setup
    // ═══════════════════════════════════════════════════════════

    SS.setupMixerControls = function() {
        // Play button
        try {
            var playBtn = document.getElementById('masterPlayBtn');
            if (playBtn) playBtn.addEventListener('click', SS.togglePlayback);
        } catch (e) { console.error('Play button setup failed:', e); }

        // Master volume
        try {
            var masterVol = document.getElementById('masterVolume');
            if (masterVol) masterVol.addEventListener('input', function() {
                SS.updateStemVolumes();
            });
        } catch (e) { console.error('Master volume setup failed:', e); }

        // Timeline seeking (mouse + touch)
        var timeline = document.getElementById('timeline');
        var isDraggingTimeline = false;

        // Timeline scrubbing
        try {
            if (timeline) {
                function seekToClientX(clientX) {
                    var rect = timeline.getBoundingClientRect();
                    var percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
                    var seekTime = percent * SS.duration;
                    SS.seekTo(seekTime);
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
            }
        } catch (e) { console.error('Timeline setup failed:', e); }

        // Skip buttons
        try {
            var skipBackBtn = document.getElementById('skipBackBtn');
            if (skipBackBtn) skipBackBtn.addEventListener('click', function() {
                var currentTime = SS.getPlaybackTime();
                var newTime = Math.max(0, currentTime - 10);
                SS.seekTo(newTime);
                SS.updateTimeline();
            });

            var skipFwdBtn = document.getElementById('skipForwardBtn');
            if (skipFwdBtn) skipFwdBtn.addEventListener('click', function() {
                var currentTime = SS.getPlaybackTime();
                var newTime = Math.min(SS.duration, currentTime + 10);
                SS.seekTo(newTime);
                SS.updateTimeline();
            });
        } catch (e) { console.error('Skip buttons setup failed:', e); }

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
            var resultsEl = document.getElementById('resultsSection');
            if (!resultsEl || !resultsEl.classList.contains('active')) return;
            if (Object.keys(SS.stemAudios).length === 0) return;

            var currentTime = SS.getPlaybackTime();

            switch(e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    var backTime = Math.max(0, currentTime - 5);
                    SS.seekTo(backTime);
                    SS.updateTimeline();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    var fwdTime = Math.min(SS.duration, currentTime + 5);
                    SS.seekTo(fwdTime);
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
                if (!stemData) return;
                stemData.muted = !stemData.muted;
                btn.classList.toggle('active', stemData.muted);
                var el = document.getElementById('stem-' + stem);
                if (el) el.classList.toggle('muted', stemData.muted);
                SS.updateStemVolumes();
            });
        });

        // Solo buttons
        document.querySelectorAll('.solo-btn').forEach(function(btn) {
            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                var stem = btn.dataset.stem;
                if (!SS.stemAudios[stem]) return;
                if (SS.soloedStems.has(stem)) {
                    SS.soloedStems.delete(stem);
                    btn.classList.remove('active');
                } else {
                    SS.soloedStems.add(stem);
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
            SS.updateStemVolumes();
            btn.classList.toggle('active', stemData.muted);
            btn.parentElement.parentElement.classList.toggle('muted', stemData.muted);
        }
    };

    // Expose globally for onclick handlers in HTML
    window.toggleMiniMute = SS.toggleMiniMute;

    // ── Mixer Audio Debug ──
    // Call window.audioDebugMixer() from console, or add ?audioDebug=1 to URL
    window.audioDebugMixer = function() {
        if (SS.audioEngine) return SS.audioEngine.debug();
        console.log('[AudioDebug:Mixer] No audioEngine instance yet');
        return 'No audioEngine';
    };

})(window.StemScribe);

