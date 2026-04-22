// StemScribe — Results Display / Stem Card Rendering
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.showResults = function(job) {
        document.getElementById('processingSection').classList.remove('active');
        document.getElementById('resultsSection').classList.add('active');
        SS.currentJob = job;
        SS.currentJobId = job.job_id; // FIXED: always set so openPracticeMode has the ID

        // Reset mixer state
        Object.values(SS.stemAudios).forEach(function(s) { s.audio.pause(); });
        SS.stemAudios = {};
        SS.analyserBuffers = {};
        SS.meterLevels = {};
        if (SS.destroyStemWaveforms) SS.destroyStemWaveforms();
        if (SS.mixerAudioCtx && SS.mixerAudioCtx.state !== 'closed') {
            SS.mixerAudioCtx.close().catch(function() {});
            SS.mixerAudioCtx = null;
        }
        SS.isPlaying = false;
        SS.soloedStems.clear();
        document.getElementById('masterPlayBtn').textContent = '\u25B6';

        if (job.metadata?.title) {
            document.getElementById('songCard').style.display = 'flex';
            document.getElementById('songTitle').textContent = job.metadata.title;
            document.getElementById('songArtist').textContent = job.metadata.artist || 'Unknown';
            var thumb = document.getElementById('songThumb');
            if (job.metadata.thumbnail) {
                thumb.src = job.metadata.thumbnail;
                thumb.style.display = 'block';
            } else {
                thumb.src = '';
                thumb.style.display = 'none';
            }
            document.getElementById('infoToggleBtn').style.display = 'block';
        } else {
            document.getElementById('songCard').style.display = 'flex';
            document.getElementById('songTitle').textContent = job.filename || 'Uploaded Track';
            document.getElementById('songArtist').textContent = 'Tap \u2139\uFE0F for track info';
            var thumb = document.getElementById('songThumb');
            thumb.src = '';
            thumb.style.display = 'none';
            document.getElementById('infoToggleBtn').style.display = 'block';
        }

        document.getElementById('trackInfoPanel').style.display = 'none';
        document.getElementById('infoToggleBtn').classList.remove('active');
        SS.trackInfoLoaded = false;

        var grid = document.getElementById('stemsGrid');
        grid.innerHTML = '';

        var stemOrder = [
            'vocals', 'vocals_lead', 'vocals_backing', 'backing_vocals',
            'guitar', 'guitar_2',
            'bass', 'bass_2',
            'drums', 'percussion_2',
            'piano', 'keys_2',
            'other',
            'other_vocals', 'other_guitar', 'other_bass', 'other_drums', 'other_piano', 'other_other',
            'other_deep',
        ];
        var sortedStems = Object.entries(job.stems).sort(function(a, b) {
            var aIdx = stemOrder.indexOf(a[0]);
            var bIdx = stemOrder.indexOf(b[0]);
            if (aIdx === -1 && bIdx === -1) return 0;
            if (aIdx === -1) return 1;
            if (bIdx === -1) return -1;
            return aIdx - bIdx;
        });

        var primaryStems = ['vocals', 'vocals_lead', 'vocals_backing', 'drums', 'bass', 'guitar', 'piano', 'other'];
        var hasCascadedStems = sortedStems.some(function(entry) { return !primaryStems.includes(entry[0]); });

        // Neve console wrapper
        var existingConsole = document.querySelector('.neve-console');
        if (!existingConsole) {
            var neveConsole = document.createElement('div');
            neveConsole.className = 'neve-console';

            var screwPositions = [
                { top: '8px', left: '8px' },
                { top: '8px', right: '8px' },
                { bottom: '45px', left: '8px' },
                { bottom: '45px', right: '8px' }
            ];
            screwPositions.forEach(function(pos) {
                var screw = document.createElement('div');
                screw.className = 'neve-screw';
                Object.assign(screw.style, pos);
                neveConsole.appendChild(screw);
            });

            var armrest = document.createElement('div');
            armrest.className = 'neve-armrest';
            var leftEnd = document.createElement('div');
            leftEnd.className = 'neve-armrest-end left';
            armrest.appendChild(leftEnd);
            var rightEnd = document.createElement('div');
            rightEnd.className = 'neve-armrest-end right';
            armrest.appendChild(rightEnd);
            neveConsole.appendChild(armrest);

            grid.parentNode.insertBefore(neveConsole, grid);
            neveConsole.appendChild(grid);
        }

        var cascadeHeaderAdded = false;
        sortedStems.forEach(function(entry) {
            var name = entry[0];
            var path = entry[1];
            var isPrimary = primaryStems.includes(name);

            if (hasCascadedStems && !isPrimary && !cascadeHeaderAdded) {
                cascadeHeaderAdded = true;
                var cascadeHeader = document.createElement('div');
                cascadeHeader.className = 'cascade-stems-header';
                cascadeHeader.innerHTML = '<span>\u{1F9E0} Deep Extraction (auto-detected instruments)</span>';
                cascadeHeader.style.cssText = 'grid-column: 1/-1; text-align:center; padding:1rem 0 0.5rem; font-family:Righteous; color:var(--psych-purple); border-top:2px solid rgba(198,120,221,0.4); margin-top:0.5rem; font-size: 0.9rem;';
                grid.appendChild(cascadeHeader);
            }

            var cfg = SS.stemConfig[name] || { icon: '\u{1F3B5}', color: '#888', label: name.replace(/_/g, ' ').toUpperCase() };
            var hasMidi = job.midi_files?.[name];
            var isCascaded = cfg.extracted || !isPrimary;

            var card = document.createElement('div');
            card.className = isCascaded ? 'stem-card cascaded-stem' : 'stem-card';
            card.id = 'stem-' + name;
            card.dataset.stem = name;
            card.style.setProperty('--stem-color', cfg.color);

            var playerName = window.currentPlayerMapping?.[name];
            var displayLabel = playerName
                ? SS.escapeHtml(playerName) + ' <span class="stem-original">(' + SS.escapeHtml(name) + ')</span>'
                : SS.escapeHtml(cfg.label || name);

            card.innerHTML =
                (isCascaded ? '<div class="cascade-badge">\u{1F9E0}</div>' : '') +
                '<div class="stem-icon">' + cfg.icon + '</div>' +
                '<div class="stem-name">' + displayLabel + '</div>' +
                '<div class="stem-waveform" id="stem-waveform-' + name + '"></div>' +
                '<div class="vu-meter" id="vu-meter-' + name + '">' +
                    '<div class="vu-face">' +
                        '<div class="vu-scale"></div>' +
                        '<div class="vu-red-zone"></div>' +
                        '<div class="vu-needle" id="vu-needle-' + name + '"></div>' +
                        '<div class="vu-label">VU</div>' +
                    '</div>' +
                '</div>' +
                '<div class="stem-mixer-controls">' +
                    '<button class="mute-btn" data-stem="' + name + '">M</button>' +
                    '<button class="solo-btn" data-stem="' + name + '">S</button>' +
                '</div>' +
                '<div class="fader-section">' +
                    '<div class="led-meter">' +
                        '<div class="peak-led" id="peak-led-' + name + '"></div>' +
                        '<div class="led-meter-track">' +
                            '<div class="led-meter-fill" id="meter-fill-' + name + '"></div>' +
                            '<div class="led-meter-segments"></div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="stem-volume">' +
                        '<div class="fader-track" id="fader-track-' + name + '">' +
                            '<div class="fader-groove"></div>' +
                            '<div class="fader-cap" id="fader-cap-' + name + '" style="top: 24px;"></div>' +
                            '<input type="range" class="stem-volume-slider" data-stem="' + name + '" min="0" max="100" value="80">' +
                        '</div>' +
                    '</div>' +
                '</div>' +
                '<div class="stem-actions">' +
                    (job.enhanced_stems?.[name] ?
                        '<a href="' + SS.API_BASE + '/download/' + job.job_id + '/enhanced/' + name + '" class="stem-btn enhanced" download title="Enhanced (cleaner)">\u2B07 \u2728</a>' +
                        '<a href="' + SS.API_BASE + '/download/' + job.job_id + '/stem/' + name + '" class="stem-btn raw" download title="Raw stem">\u2B07 Raw</a>' :
                        '<a href="' + SS.API_BASE + '/download/' + job.job_id + '/stem/' + name + '" class="stem-btn" download>\u2B07 DL</a>'
                    ) +
                    '<a href="' + SS.API_BASE + '/download/' + job.job_id + '/stem/' + name + '/mp3" class="stem-btn mp3-dl" download title="Download as MP3">\u2B07 MP3</a>' +
                    ((job.musicxml_files?.[name] || job.gp_files?.[name]) ? '<button class="stem-btn notation" onclick="StemScribe.openNotationForStem(\'' + name + '\')" title="Open in Practice Mode">\u25B6 TAB</button>' : '') +
                    (job.musicxml_files?.[name] ? '<button class="stem-btn notation" onclick="StemScribe.exportPDFForStem(\'' + name + '\')" title="Export sheet music as PDF">\u{1F4C4} PDF</button>' : '') +
                    (hasMidi ? '<a href="' + SS.API_BASE + '/download/' + job.job_id + '/midi/' + name + '" class="stem-btn midi" download title="Download MIDI file">\u{1F3B9}</a>' : '') +
                '</div>' +
                '<audio id="audio-' + name + '" src="' + (job.enhanced_stems?.[name] ? SS.API_BASE + '/download/' + job.job_id + '/enhanced/' + name : SS.API_BASE + '/download/' + job.job_id + '/stem/' + name) + '" preload="auto"></audio>';

            grid.appendChild(card);

            var audio = card.querySelector('#audio-' + name);
            var webAudio = SS.connectStemToWebAudio(name, audio);
            SS.stemAudios[name] = { audio: audio, muted: false, volume: 0.8, source: webAudio.source, gainNode: webAudio.gainNode, analyser: webAudio.analyser };
            webAudio.gainNode.gain.value = 0.8;

            setTimeout(function() { SS.updateFaderCap(name, 80); }, 100);

            audio.addEventListener('loadedmetadata', function() {
                if (SS.duration === 0) {
                    SS.duration = audio.duration;
                    document.getElementById('totalTime').textContent = SS.formatTime(SS.duration);
                }
            });
        });

        // Sub-stems from skills
        if (job.sub_stems && Object.keys(job.sub_stems).length > 0) {
            if (SS.useHierarchicalLayout) {
                SS.addHierarchicalSubstems(job, grid);
            } else {
                SS.addFlatSubstems(job, grid);
            }
        }

        SS.setupMixerControls();
        SS.autoFetchPlayerNames();

        // Initialize waveform display with the first stem's audio URL
        var firstStemEntry = sortedStems[0];
        if (firstStemEntry && SS.initWaveform) {
            var firstStemName = firstStemEntry[0];
            var firstAudioUrl = job.enhanced_stems?.[firstStemName]
                ? SS.API_BASE + '/download/' + job.job_id + '/enhanced/' + firstStemName
                : SS.API_BASE + '/download/' + job.job_id + '/stem/' + firstStemName;
            SS.initWaveform(firstAudioUrl);
        }

        // Initialize per-stem waveform visualizations
        if (SS.initStemWaveforms) {
            SS.initStemWaveforms(job, sortedStems);
        }

        // Apply current playback rate to all new stems
        if (SS.playbackRate && SS.playbackRate !== 1.0) {
            SS.setPlaybackRate(SS.playbackRate);
        }

        // Clear any existing loop
        if (SS.clearLoopRegion) {
            SS.clearLoopRegion();
        }
    };

    // Option A: Flat sub-stem layout
    SS.addFlatSubstems = function(job, grid) {
        var skillHeader = document.createElement('div');
        skillHeader.className = 'skill-stems-header';
        skillHeader.innerHTML = '<span>\u{1F3AF} Skill-Enhanced Stems</span>';
        skillHeader.style.cssText = 'grid-column: 1/-1; text-align:center; padding:1rem 0 0.5rem; font-family:Righteous; color:var(--psych-orange); border-top:1px solid rgba(255,123,84,0.3); margin-top:0.5rem;';
        grid.appendChild(skillHeader);

        Object.entries(job.sub_stems).forEach(function(entry) {
            var skillId = entry[0];
            var subStems = entry[1];
            var skill = SS.availableSkills.find(function(s) { return s.id === skillId; });
            Object.entries(subStems).forEach(function(subEntry) {
                var subStemName = subEntry[0];
                var relPath = subEntry[1];
                var filename = relPath.split('/').pop();
                var displayName = subStemName.replace(/_/g, ' ').toUpperCase();
                var color = SS.subStemColors[subStemName] || '#c678dd';
                SS.addSubStemCard(grid, job, skillId, subStemName, filename, displayName, color, skill);
            });
        });
    };

    // Option B: Hierarchical sub-stem layout
    SS.addHierarchicalSubstems = function(job, grid) {
        var skillParentMap = { 'vocal_virtuoso': 'vocals', 'horn_hunter': 'other', 'guitar_god': 'guitar', 'string_section': 'other', 'keys_kingdom': 'piano' };

        Object.entries(job.sub_stems).forEach(function(entry) {
            var skillId = entry[0];
            var subStems = entry[1];
            var skill = SS.availableSkills.find(function(s) { return s.id === skillId; });
            var parentStem = skillParentMap[skillId] || 'other';
            var parentCard = document.getElementById('stem-' + parentStem);

            if (parentCard) {
                parentCard.classList.add('has-substems');
                if (!parentCard.querySelector('.substems-toggle')) {
                    var toggleBtn = document.createElement('button');
                    toggleBtn.className = 'substems-toggle';
                    toggleBtn.innerHTML = (skill?.emoji || '\u{1F3AF}') + ' +' + Object.keys(subStems).length;
                    toggleBtn.onclick = function(e) { e.stopPropagation(); SS.toggleSubstems(parentStem); };
                    parentCard.appendChild(toggleBtn);
                }

                var container = document.getElementById('substems-' + parentStem);
                if (!container) {
                    container = document.createElement('div');
                    container.className = 'substems-container';
                    container.id = 'substems-' + parentStem;
                    container.innerHTML = '<div class="substem-parent-label">' + SS.escapeHtml(skill?.name || skillId) + ' sub-stems</div><div class="substems-grid"></div>';
                    parentCard.after(container);
                }

                var subGrid = container.querySelector('.substems-grid');
                Object.entries(subStems).forEach(function(subEntry) {
                    var subStemName = subEntry[0];
                    var relPath = subEntry[1];
                    var filename = relPath.split('/').pop();
                    var displayName = subStemName.replace(/_/g, ' ');

                    var miniCard = document.createElement('div');
                    miniCard.className = 'substem-mini-card';
                    miniCard.innerHTML =
                        '<div class="emoji">' + SS.escapeHtml(skill?.emoji || '\u{1F3AF}') + '</div>' +
                        '<div class="name">' + SS.escapeHtml(displayName) + '</div>' +
                        '<div class="mini-controls">' +
                            '<button class="mini-btn" onclick="StemScribe.toggleMiniMute(\'' + SS.escapeJsString(subStemName) + '\', this)">M</button>' +
                            '<a href="' + SS.API_BASE + '/download/' + encodeURIComponent(job.job_id) + '/substem/' + encodeURIComponent(skillId) + '/' + encodeURIComponent(filename) + '" class="mini-btn" download>\u2B07</a>' +
                        '</div>' +
                        '<audio id="audio-' + SS.escapeHtml(subStemName) + '" src="' + SS.API_BASE + '/download/' + encodeURIComponent(job.job_id) + '/substem/' + encodeURIComponent(skillId) + '/' + encodeURIComponent(filename) + '" preload="auto"></audio>';
                    subGrid.appendChild(miniCard);

                    var audio = miniCard.querySelector('#audio-' + subStemName);
                    var webAudio = SS.connectStemToWebAudio(subStemName, audio);
                    SS.stemAudios[subStemName] = { audio: audio, muted: false, volume: 0.8, isSubStem: true, source: webAudio.source, gainNode: webAudio.gainNode, analyser: webAudio.analyser };
                    webAudio.gainNode.gain.value = 0.8;
                });
            }
        });
    };

    SS.toggleSubstems = function(parentStem) {
        var container = document.getElementById('substems-' + parentStem);
        var toggle = document.querySelector('#stem-' + parentStem + ' .substems-toggle');
        if (container) container.classList.toggle('expanded');
        if (toggle) toggle.classList.toggle('expanded');
    };

    SS.addSubStemCard = function(grid, job, skillId, subStemName, filename, displayName, color, skill) {
        var card = document.createElement('div');
        card.className = 'stem-card sub-stem';
        card.id = 'stem-' + subStemName;
        card.style.setProperty('--stem-color', color);
        card.innerHTML =
            '<div class="stem-icon">' + SS.escapeHtml(skill?.emoji || '\u{1F3AF}') + '</div>' +
            '<div class="stem-name">' + SS.escapeHtml(displayName) + '</div>' +
            '<div class="stem-waveform" id="stem-waveform-' + SS.escapeHtml(subStemName) + '"></div>' +
            '<div class="vu-meter" id="vu-meter-' + SS.escapeHtml(subStemName) + '">' +
                '<div class="vu-face">' +
                    '<div class="vu-scale"></div>' +
                    '<div class="vu-red-zone"></div>' +
                    '<div class="vu-needle" id="vu-needle-' + SS.escapeHtml(subStemName) + '"></div>' +
                    '<div class="vu-label">VU</div>' +
                '</div>' +
            '</div>' +
            '<div class="stem-mixer-controls">' +
                '<button class="mute-btn" data-stem="' + SS.escapeHtml(subStemName) + '">M</button>' +
                '<button class="solo-btn" data-stem="' + SS.escapeHtml(subStemName) + '">S</button>' +
            '</div>' +
            '<div class="fader-section">' +
                '<div class="led-meter">' +
                    '<div class="peak-led" id="peak-led-' + SS.escapeHtml(subStemName) + '"></div>' +
                    '<div class="led-meter-track">' +
                        '<div class="led-meter-fill" id="meter-fill-' + SS.escapeHtml(subStemName) + '"></div>' +
                        '<div class="led-meter-segments"></div>' +
                    '</div>' +
                '</div>' +
                '<div class="stem-volume">' +
                    '<div class="fader-track" id="fader-track-' + SS.escapeHtml(subStemName) + '">' +
                        '<div class="fader-groove"></div>' +
                        '<div class="fader-cap" id="fader-cap-' + SS.escapeHtml(subStemName) + '" style="top: 24px;"></div>' +
                        '<input type="range" class="stem-volume-slider" data-stem="' + SS.escapeHtml(subStemName) + '" min="0" max="100" value="80">' +
                    '</div>' +
                '</div>' +
            '</div>' +
            '<div class="stem-actions">' +
                '<a href="' + SS.API_BASE + '/download/' + encodeURIComponent(job.job_id) + '/substem/' + encodeURIComponent(skillId) + '/' + encodeURIComponent(filename) + '" class="stem-btn" download>\u2B07 DL</a>' +
            '</div>' +
            '<audio id="audio-' + SS.escapeHtml(subStemName) + '" src="' + SS.API_BASE + '/download/' + encodeURIComponent(job.job_id) + '/substem/' + encodeURIComponent(skillId) + '/' + encodeURIComponent(filename) + '" preload="auto"></audio>';

        grid.appendChild(card);

        var audio = card.querySelector('#audio-' + subStemName);
        var subWebAudio = SS.connectStemToWebAudio(subStemName, audio);
        SS.stemAudios[subStemName] = { audio: audio, muted: false, volume: 0.8, isSubStem: true, source: subWebAudio.source, gainNode: subWebAudio.gainNode, analyser: subWebAudio.analyser };
        subWebAudio.gainNode.gain.value = 0.8;
        setTimeout(function() { SS.updateFaderCap(subStemName, 80); }, 100);
    };

    // Navigation helpers
    SS.openPracticeMode = function() {
        if (window.SS_Analytics) SS_Analytics.practiceModeOpened();
        var jobId = SS.currentJobId || (SS.currentJob && SS.currentJob.job_id);
        if (jobId) {
            window.open('practice.html?job=' + jobId, '_blank');
        } else {
            window.open('practice.html', '_blank');
        }
    };

    SS.openNotationForStem = function(stemName) {
        var jobId = SS.currentJobId || (SS.currentJob && SS.currentJob.job_id);
        if (jobId) {
            window.open('practice.html?job=' + jobId + '&stem=' + stemName, '_blank');
        } else {
            window.open('practice.html', '_blank');
        }
    };

    SS.exportPDFForStem = function(stemName) {
        var jobId = SS.currentJobId || (SS.currentJob && SS.currentJob.job_id);
        if (jobId) {
            window.open('practice.html?job=' + jobId + '&stem=' + stemName + '&autoExportPDF=true', '_blank');
        }
    };

    // Download all stems as ZIP
    SS.downloadStemsZip = function() {
        var jobId = SS.currentJobId || (SS.currentJob && SS.currentJob.job_id);
        if (!jobId) return;

        var btn = document.getElementById('downloadZipBtn');
        var icon = document.getElementById('downloadZipIcon');
        var label = document.getElementById('downloadZipLabel');
        if (!btn) return;

        // Show loading state
        btn.disabled = true;
        btn.style.opacity = '0.7';
        btn.style.pointerEvents = 'none';
        icon.textContent = '\u23F3';
        label.textContent = 'Preparing ZIP...';

        // Use fetch to track download, then trigger via blob URL
        fetch(SS.API_BASE + '/download/' + jobId + '/zip', {
            headers: SS.authHeaders ? SS.authHeaders() : {}
        })
        .then(function(response) {
            if (!response.ok) throw new Error('Download failed');
            return response.blob();
        })
        .then(function(blob) {
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = (SS.currentJob && SS.currentJob.title ? SS.currentJob.title : 'stems') + '.zip';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            // Reset button
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.pointerEvents = '';
            icon.textContent = '\uD83D\uDCE6';
            label.textContent = 'Download Stems (ZIP)';
        })
        .catch(function(err) {
            console.error('ZIP download error:', err);
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.pointerEvents = '';
            icon.textContent = '\u26A0\uFE0F';
            label.textContent = 'Download Failed — Retry';
            setTimeout(function() {
                icon.textContent = '\uD83D\uDCE6';
                label.textContent = 'Download Stems (ZIP)';
            }, 3000);
        });
    };

    // Expose globally for onclick handlers
    window.openPracticeMode = SS.openPracticeMode;
    window.openNotationForStem = SS.openNotationForStem;
    window.exportPDFForStem = SS.exportPDFForStem;

})(window.StemScribe);
