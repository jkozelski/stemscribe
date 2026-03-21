// StemScribe — Karaoke Mode
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    var karaokeActive = false;
    var karaokeLines = null;  // [{time, text}]
    var karaokeAnimId = null;
    var karaokeContainer = null;
    var previousVocalVolumes = {};  // Save vocal volumes before muting

    // Number of lines to show before/after active
    var LINES_BEFORE = 2;
    var LINES_AFTER = 3;

    SS.karaokeMode = false;

    SS.enterKaraoke = function(jobId) {
        karaokeContainer = document.getElementById('karaokeContainer');
        if (!karaokeContainer) return;

        // Fetch lyrics
        var lyricsUrl = SS.API_BASE + '/lyrics/' + jobId;
        karaokeContainer.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:rgba(255,255,255,0.4);font-size:1.2rem;">Loading lyrics...</div>';
        karaokeContainer.classList.add('visible');

        // Hide other views
        var alphaTab = document.getElementById('alphaTab');
        var chordChart = document.getElementById('chordChart');
        if (alphaTab) alphaTab.style.display = 'none';
        if (chordChart) chordChart.classList.remove('visible');

        fetch(lyricsUrl)
            .then(function(resp) { return resp.json(); })
            .then(function(data) {
                if (data.found && data.synced_lyrics && data.synced_lyrics.length > 0) {
                    karaokeLines = data.synced_lyrics;
                    _buildKaraokeUI(data);
                    _muteVocals();
                    karaokeActive = true;
                    SS.karaokeMode = true;
                    _startKaraokeSync();
                } else if (data.found && data.plain_lyrics) {
                    // Plain lyrics only — show static
                    _showPlainLyrics(data);
                    _muteVocals();
                    karaokeActive = true;
                    SS.karaokeMode = true;
                } else {
                    _showNoLyrics(jobId);
                }
            })
            .catch(function(err) {
                console.error('Karaoke lyrics fetch failed:', err);
                _showNoLyrics(jobId);
            });
    };

    SS.exitKaraoke = function() {
        karaokeActive = false;
        SS.karaokeMode = false;
        karaokeLines = null;

        if (karaokeAnimId) {
            cancelAnimationFrame(karaokeAnimId);
            karaokeAnimId = null;
        }

        // Hide karaoke container
        var container = document.getElementById('karaokeContainer');
        if (container) {
            container.classList.remove('visible');
            container.innerHTML = '';
        }

        // Restore vocal volumes
        _restoreVocals();

        // Restore previous view
        if (typeof switchView === 'function') {
            switchView(window.currentView || 'chords');
        }
    };

    function _muteVocals() {
        previousVocalVolumes = {};
        var vocalKeys = ['vocals', 'Vocals', 'vocal', 'Vocal'];
        Object.entries(SS.stemAudios || {}).forEach(function(entry) {
            var name = entry[0];
            var data = entry[1];
            var isVocal = vocalKeys.some(function(k) {
                return name.toLowerCase().indexOf(k.toLowerCase()) >= 0;
            });
            if (isVocal) {
                previousVocalVolumes[name] = { volume: data.volume, muted: data.muted };
                data.muted = true;
                if (SS.audioEngine) {
                    SS.audioEngine.setVolume(name, 0);
                }
                // Update mute button UI
                var muteBtn = document.querySelector('.mute-btn[data-stem="' + name + '"]');
                if (muteBtn) muteBtn.classList.add('active');
                var stemEl = document.getElementById('stem-' + name);
                if (stemEl) stemEl.classList.add('muted');
            }
        });
    }

    function _restoreVocals() {
        Object.entries(previousVocalVolumes).forEach(function(entry) {
            var name = entry[0];
            var saved = entry[1];
            var data = SS.stemAudios[name];
            if (data) {
                data.muted = saved.muted;
                data.volume = saved.volume;
                var vol = saved.muted ? 0 : saved.volume;
                if (SS.audioEngine) {
                    SS.audioEngine.setVolume(name, vol);
                }
                var muteBtn = document.querySelector('.mute-btn[data-stem="' + name + '"]');
                if (muteBtn) muteBtn.classList.toggle('active', saved.muted);
                var stemEl = document.getElementById('stem-' + name);
                if (stemEl) stemEl.classList.toggle('muted', saved.muted);
            }
        });
        previousVocalVolumes = {};
    }

    function _buildKaraokeUI(data) {
        var meta = data.metadata || {};
        var title = meta.title || '';
        var artist = meta.artist || '';

        karaokeContainer.innerHTML =
            '<div class="karaoke-header">' +
                '<div>' +
                    '<span class="karaoke-title">' + _escapeHtml(title) + '</span>' +
                    '<span class="karaoke-artist">' + _escapeHtml(artist) + '</span>' +
                '</div>' +
                '<button class="karaoke-exit-btn" onclick="StemScribe.exitKaraoke()">Exit Karaoke</button>' +
            '</div>' +
            '<div class="karaoke-lyrics" id="karaokeLyrics">' +
                '<div class="karaoke-lyrics-scroll" id="karaokeLyricsScroll"></div>' +
            '</div>' +
            '<div class="karaoke-transport">' +
                '<button onclick="StemScribe.togglePlayback();" class="play-btn" id="karaokePlayBtn">' +
                    (SS.isPlaying ? '\u23F8' : '\u25B6') +
                '</button>' +
                '<span class="karaoke-time" id="karaokeCurrentTime">0:00</span>' +
                '<div class="karaoke-timeline" id="karaokeTimeline">' +
                    '<div class="karaoke-timeline-progress" id="karaokeTimelineProgress"></div>' +
                '</div>' +
                '<span class="karaoke-time" id="karaokeTotalTime">' + SS.formatTime(SS.duration || 0) + '</span>' +
            '</div>';

        // Build lyric line elements
        var scroll = document.getElementById('karaokeLyricsScroll');
        karaokeLines.forEach(function(line, i) {
            var div = document.createElement('div');
            div.className = 'karaoke-line';
            div.setAttribute('data-index', i);
            div.setAttribute('data-time', line.time);
            div.textContent = line.text || '\u00A0'; // non-breaking space for empty lines
            scroll.appendChild(div);
        });

        // Timeline seeking
        var timeline = document.getElementById('karaokeTimeline');
        timeline.addEventListener('click', function(e) {
            var rect = timeline.getBoundingClientRect();
            var percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            var seekTime = percent * SS.duration;
            SS.seekTo(seekTime);
        });
    }

    function _showPlainLyrics(data) {
        var meta = data.metadata || {};
        var lines = (data.plain_lyrics || '').split('\n');

        karaokeContainer.innerHTML =
            '<div class="karaoke-header">' +
                '<div>' +
                    '<span class="karaoke-title">' + _escapeHtml(meta.title || '') + '</span>' +
                    '<span class="karaoke-artist">' + _escapeHtml(meta.artist || '') + '</span>' +
                '</div>' +
                '<button class="karaoke-exit-btn" onclick="StemScribe.exitKaraoke()">Exit Karaoke</button>' +
            '</div>' +
            '<div class="karaoke-lyrics" style="overflow-y:auto;justify-content:flex-start;">' +
                '<div class="karaoke-lyrics-scroll" style="gap:0.6rem;">' +
                    lines.map(function(l) {
                        return '<div class="karaoke-line" style="color:rgba(255,255,255,0.6);filter:none;transform:none;font-size:1.4rem;">' +
                            _escapeHtml(l || '\u00A0') + '</div>';
                    }).join('') +
                '</div>' +
            '</div>' +
            '<div class="karaoke-transport">' +
                '<button onclick="StemScribe.togglePlayback();" class="play-btn">' +
                    (SS.isPlaying ? '\u23F8' : '\u25B6') +
                '</button>' +
                '<span style="color:rgba(255,255,255,0.4);font-size:0.85rem;">Plain lyrics (no sync available)</span>' +
            '</div>';
    }

    function _showNoLyrics(jobId) {
        karaokeContainer.innerHTML =
            '<div class="karaoke-header">' +
                '<div><span class="karaoke-title">Karaoke Mode</span></div>' +
                '<button class="karaoke-exit-btn" onclick="StemScribe.exitKaraoke()">Exit</button>' +
            '</div>' +
            '<div class="karaoke-lyrics">' +
                '<div class="karaoke-no-lyrics">' +
                    '<p>No lyrics found for this track.</p>' +
                    '<p style="font-size:0.9rem;">Try searching with the correct artist and title.</p>' +
                    '<div style="margin-top:1.5rem;">' +
                        '<input type="text" id="karaokeArtistInput" placeholder="Artist" ' +
                            'style="padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.15);color:#fff;border-radius:6px;margin-right:0.5rem;font-family:inherit;">' +
                        '<input type="text" id="karaokeTitleInput" placeholder="Song title" ' +
                            'style="padding:0.5rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.15);color:#fff;border-radius:6px;margin-right:0.5rem;font-family:inherit;">' +
                        '<button class="karaoke-retry-btn" onclick="StemScribe._karaokeRetry(\'' + jobId + '\')">Search</button>' +
                    '</div>' +
                '</div>' +
            '</div>';
    }

    SS._karaokeRetry = function(jobId) {
        var artist = document.getElementById('karaokeArtistInput').value.trim();
        var title = document.getElementById('karaokeTitleInput').value.trim();
        if (!title) return;

        var url = SS.API_BASE + '/lyrics/' + jobId + '?artist=' + encodeURIComponent(artist) + '&title=' + encodeURIComponent(title);

        karaokeContainer.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:rgba(255,255,255,0.4);">Searching...</div>';
        karaokeContainer.classList.add('visible');

        fetch(url)
            .then(function(resp) { return resp.json(); })
            .then(function(data) {
                if (data.found && data.synced_lyrics && data.synced_lyrics.length > 0) {
                    karaokeLines = data.synced_lyrics;
                    _buildKaraokeUI(data);
                    _muteVocals();
                    karaokeActive = true;
                    SS.karaokeMode = true;
                    _startKaraokeSync();
                } else if (data.found && data.plain_lyrics) {
                    _showPlainLyrics(data);
                    _muteVocals();
                    karaokeActive = true;
                    SS.karaokeMode = true;
                } else {
                    _showNoLyrics(jobId);
                }
            })
            .catch(function() { _showNoLyrics(jobId); });
    };

    function _startKaraokeSync() {
        if (karaokeAnimId) cancelAnimationFrame(karaokeAnimId);

        var lineEls = document.querySelectorAll('.karaoke-line');
        var scroll = document.getElementById('karaokeLyricsScroll');
        var lyricsViewport = document.getElementById('karaokeLyrics');
        var lastActiveIndex = -1;

        function syncFrame() {
            if (!karaokeActive) return;

            if (!SS.audioEngine || !SS.audioEngine.isLoaded()) {
                karaokeAnimId = requestAnimationFrame(syncFrame);
                return;
            }

            var currentTime = SS.getPlaybackTime();

            // Update transport
            var progress = document.getElementById('karaokeTimelineProgress');
            var timeDisplay = document.getElementById('karaokeCurrentTime');
            var playBtn = document.getElementById('karaokePlayBtn');
            if (progress) {
                progress.style.width = ((currentTime / SS.duration) * 100) + '%';
            }
            if (timeDisplay) {
                timeDisplay.textContent = SS.formatTime(currentTime);
            }
            if (playBtn) {
                playBtn.textContent = SS.isPlaying ? '\u23F8' : '\u25B6';
            }

            // Find active line
            var activeIndex = -1;
            for (var i = karaokeLines.length - 1; i >= 0; i--) {
                if (currentTime >= karaokeLines[i].time) {
                    activeIndex = i;
                    break;
                }
            }

            // Update line classes
            if (activeIndex !== lastActiveIndex) {
                lastActiveIndex = activeIndex;

                lineEls.forEach(function(el, idx) {
                    el.classList.remove('active', 'past', 'approaching');

                    if (idx === activeIndex) {
                        el.classList.add('active');
                    } else if (idx < activeIndex) {
                        el.classList.add('past');
                    } else if (idx <= activeIndex + 2) {
                        el.classList.add('approaching');
                    }
                });

                // Scroll active line to center
                if (activeIndex >= 0 && lineEls[activeIndex]) {
                    var activeEl = lineEls[activeIndex];
                    var viewportHeight = lyricsViewport.clientHeight;
                    var lineTop = activeEl.offsetTop;
                    var lineHeight = activeEl.offsetHeight;
                    var scrollTarget = lineTop - (viewportHeight / 2) + (lineHeight / 2);

                    lyricsViewport.scrollTo({
                        top: scrollTarget,
                        behavior: 'smooth'
                    });
                }
            }

            karaokeAnimId = requestAnimationFrame(syncFrame);
        }

        karaokeAnimId = requestAnimationFrame(syncFrame);
    }

    function _escapeHtml(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

})(window.StemScribe);
