// StemScribe — Track Info Panel & Player Name Mapping
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.toggleTrackInfo = async function() {
        var panel = document.getElementById('trackInfoPanel');
        var btn = document.getElementById('infoToggleBtn');

        if (panel.style.display === 'none' || panel.style.display === '') {
            panel.style.display = 'block';
            btn.classList.add('active');

            if (SS.trackInfoLoaded && SS.currentTrackInfo) {
                SS.displayTrackInfo(SS.currentTrackInfo);
            } else {
                await SS.fetchTrackInfo();
            }
        } else {
            panel.style.display = 'none';
            btn.classList.remove('active');
        }
    };

    SS.fetchTrackInfo = async function() {
        var loadingEl = document.getElementById('infoLoading');
        var bodyEl = document.getElementById('infoBody');

        loadingEl.style.display = 'flex';
        loadingEl.innerHTML = '<div class="info-spinner"></div><span>Loading track info...</span>';
        bodyEl.style.display = 'none';

        var jobId = SS.currentJob?.job_id || SS.currentJobId;

        if (!jobId) {
            loadingEl.innerHTML = '<span style="color: var(--psych-pink);">\u274C No job available - process a song first</span>';
            return;
        }

        try {
            var controller = new AbortController();
            var timeoutId = setTimeout(function() { controller.abort(); }, 10000);

            var response = await fetch(SS.API_BASE + '/info/' + jobId, {
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!response.ok) {
                var errorText = await response.text();
                throw new Error('Server error: ' + response.status + ' - ' + errorText);
            }

            var info = await response.json();

            if (info.error) {
                throw new Error(info.error);
            }

            SS.currentTrackInfo = info;
            SS.trackInfoLoaded = true;
            SS.displayTrackInfo(info);

        } catch (error) {
            console.error('Failed to fetch track info:', error);
            var isTimeout = error.name === 'AbortError';
            loadingEl.innerHTML =
                '<span style="color: var(--psych-pink);">' +
                    (isTimeout ? '\u23F1\uFE0F Request timed out (10s)' : '\u274C ' + SS.escapeHtml(error.message)) +
                '</span>' +
                '<button onclick="StemScribe.fetchTrackInfo()" style="margin-top: 10px; padding: 5px 15px; background: var(--psych-orange); border: none; border-radius: 4px; cursor: pointer; color: white;">Retry</button>';
        }
    };

    SS.displayTrackInfo = function(info) {
        var loadingEl = document.getElementById('infoLoading');
        var bodyEl = document.getElementById('infoBody');

        loadingEl.style.display = 'none';
        bodyEl.style.display = 'block';

        if (info.player_mapping) {
            window.currentPlayerMapping = info.player_mapping;
            SS.updateStemLabels(info.player_mapping);
        }

        // Album info
        var albumSection = document.getElementById('albumSection');
        if (info.album) {
            document.getElementById('albumName').textContent = info.album;
            document.getElementById('albumYear').textContent = info.album_year || '';
            document.getElementById('albumDescription').textContent = info.album_description || '';
            albumSection.style.display = 'block';
        } else {
            albumSection.style.display = 'none';
        }

        // Era info
        var eraSection = document.getElementById('eraSection');
        if (info.era) {
            document.getElementById('eraKeyboardist').textContent = info.era.keyboardist;
            document.getElementById('eraYears').textContent = '(' + info.era.years + ')';
            document.getElementById('eraNotes').textContent = info.era.notes;
            eraSection.style.display = 'block';
        } else {
            eraSection.style.display = 'none';
        }

        // Artist bio
        var bioSection = document.getElementById('artistBioSection');
        var bioEl = document.getElementById('artistBio');
        var wikiLink = document.getElementById('wikiLink');

        var websiteLink = document.getElementById('artistWebsiteLink');
        var wikiSearchLink = document.getElementById('wikiSearchLink');

        if (info.bio) {
            bioEl.textContent = info.bio;
            bioSection.style.display = 'block';
            if (info.wikipedia_url) {
                wikiLink.href = info.wikipedia_url;
                wikiLink.style.display = 'inline-block';
            } else {
                wikiLink.style.display = 'none';
            }
            if (wikiSearchLink) {
                // Hide Wikipedia search if we already have a website or wiki URL
                if (info.artist && !info.wikipedia_url && !info.website_url) {
                    wikiSearchLink.href = 'https://en.wikipedia.org/w/index.php?search=' + encodeURIComponent(info.artist);
                    wikiSearchLink.style.display = 'inline-block';
                } else {
                    wikiSearchLink.style.display = 'none';
                }
            }
            if (websiteLink) {
                if (info.website_url) {
                    websiteLink.href = info.website_url;
                    websiteLink.style.display = 'inline-block';
                } else {
                    websiteLink.style.display = 'none';
                }
            }
            var spotifyLink = document.getElementById('spotifyLink');
            if (spotifyLink) {
                if (info.spotify_url) {
                    spotifyLink.href = info.spotify_url;
                    spotifyLink.style.display = 'inline-block';
                } else {
                    spotifyLink.style.display = 'none';
                }
            }
        } else {
            bioSection.style.display = 'none';
        }

        // Song info
        var songSection = document.getElementById('songInfoSection');
        var songEl = document.getElementById('songInfo');
        var songWikiLink = document.getElementById('songWikiLink');
        var geniusLink = document.getElementById('geniusLink');

        if (info.song_info || info.song_wikipedia_url || info.genius_url) {
            songEl.textContent = info.song_info || 'Click links below for lyrics and song details.';
            songSection.style.display = 'block';

            if (info.song_wikipedia_url) {
                songWikiLink.href = info.song_wikipedia_url;
                songWikiLink.style.display = 'inline-block';
            } else {
                songWikiLink.style.display = 'none';
            }

            if (info.genius_url) {
                geniusLink.href = info.genius_url;
                geniusLink.style.display = 'inline-block';
            } else {
                geniusLink.style.display = 'none';
            }
        } else {
            songSection.style.display = 'none';
        }

        // Album personnel
        var personnelSection = document.getElementById('personnelSection');
        var personnelList = document.getElementById('personnelList');
        if (info.personnel && Object.keys(info.personnel).length > 0) {
            personnelList.innerHTML = '';
            for (var pName in info.personnel) {
                var role = info.personnel[pName];
                var item = document.createElement('div');
                item.className = 'personnel-item';
                item.innerHTML =
                    '<span class="name">' + SS.escapeHtml(pName.split(' ').map(function(w) { return w.charAt(0).toUpperCase() + w.slice(1); }).join(' ')) + '</span>' +
                    '<span class="role">' + SS.escapeHtml(role) + '</span>';
                personnelList.appendChild(item);
            }
            personnelSection.style.display = 'block';
        } else {
            personnelSection.style.display = 'none';
        }

        // Band members
        var membersSection = document.getElementById('membersSection');
        var membersList = document.getElementById('membersList');
        if (info.members && Object.keys(info.members).length > 0 && !info.personnel) {
            membersList.innerHTML = '';
            for (var mName in info.members) {
                var mRole = info.members[mName];
                var mItem = document.createElement('div');
                mItem.className = 'member-item';
                mItem.innerHTML =
                    '<div class="name">' + SS.escapeHtml(mName.split(' ').map(function(w) { return w.charAt(0).toUpperCase() + w.slice(1); }).join(' ')) + '</div>' +
                    '<div class="role">' + SS.escapeHtml(mRole) + '</div>';
                membersList.appendChild(mItem);
            }
            membersSection.style.display = 'block';
        } else {
            membersSection.style.display = 'none';
        }

        // Learning tips
        var tipsSection = document.getElementById('learningTipsSection');
        var tipsEl = document.getElementById('learningTips');
        if (info.learning_tips) {
            tipsEl.textContent = info.learning_tips;
            tipsSection.style.display = 'block';
        } else {
            tipsSection.style.display = 'none';
        }

        // Stem-specific tips
        var stemTipsSection = document.getElementById('stemTipsSection');
        var stemTipsList = document.getElementById('stemTipsList');
        if (info.stem_tips && Object.keys(info.stem_tips).length > 0) {
            stemTipsList.innerHTML = '';
            for (var stem in info.stem_tips) {
                var tip = info.stem_tips[stem];
                var tipDisplayName = info.player_mapping?.[stem] || stem.replace('_', ' ');
                var tipItem = document.createElement('div');
                tipItem.className = 'stem-tip';
                tipItem.innerHTML =
                    '<div class="stem-name">' + SS.escapeHtml(tipDisplayName) + '</div>' +
                    '<div class="tip-text">' + SS.escapeHtml(tip) + '</div>';
                stemTipsList.appendChild(tipItem);
            }
            stemTipsSection.style.display = 'block';
        } else {
            stemTipsSection.style.display = 'none';
        }

        // Style
        var styleSection = document.getElementById('styleSection');
        var styleEl = document.getElementById('styleInfo');
        if (info.style) {
            styleEl.textContent = info.style;
            styleSection.style.display = 'block';
        } else {
            styleSection.style.display = 'none';
        }
    };

    SS.updateStemLabels = function(playerMapping) {
        if (!playerMapping) return;

        var stemCards = document.querySelectorAll('.stem-card');
        stemCards.forEach(function(card) {
            var stemName = card.dataset.stem;
            if (stemName && playerMapping[stemName]) {
                var labelEl = card.querySelector('.stem-name');
                if (labelEl) {
                    if (!labelEl.dataset.originalName) {
                        labelEl.dataset.originalName = labelEl.textContent;
                    }
                    var pName = playerMapping[stemName];
                    labelEl.innerHTML = SS.escapeHtml(pName) + ' <span class="stem-original">(' + SS.escapeHtml(stemName) + ')</span>';
                }
            }
        });
    };

    SS.autoFetchPlayerNames = async function() {
        if (!SS.currentJob) return;

        try {
            var response = await fetch(SS.API_BASE + '/info/' + SS.currentJob.job_id);
            var info = await response.json();

            SS.currentTrackInfo = info;
            SS.trackInfoLoaded = true;

            if (info.player_mapping) {
                window.currentPlayerMapping = info.player_mapping;
                SS.updateStemLabels(info.player_mapping);
            }
        } catch (error) {
            console.log('Could not auto-fetch player names:', error);
        }
    };

    // Expose globally for onclick handlers
    window.toggleTrackInfo = SS.toggleTrackInfo;

})(window.StemScribe);
