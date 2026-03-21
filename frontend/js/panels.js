// StemScribe — Settings Panel, Library Panel, Google Drive, Layout Toggles
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.initPanels = function() {
        // Settings Panel
        try {
            var settingsBtn = document.getElementById('settingsBtn');
            var settingsPanel = document.getElementById('settingsPanel');
            var settingsOverlay = document.getElementById('settingsOverlay');
            var closeSettings = document.getElementById('closeSettings');

            if (settingsBtn) settingsBtn.addEventListener('click', function() {
                if (settingsPanel) settingsPanel.classList.add('open');
                if (settingsOverlay) settingsOverlay.classList.add('open');
                SS.checkDriveStatus();
                SS.loadLocalStats();
            });

            if (closeSettings) closeSettings.addEventListener('click', SS.closeSettingsPanel);
            if (settingsOverlay) settingsOverlay.addEventListener('click', SS.closeSettingsPanel);
        } catch (e) { console.error('Settings panel setup failed:', e); }

        // Library Panel
        try {
            var libraryBtn = document.getElementById('libraryBtn');
            var libraryPanel = document.getElementById('libraryPanel');
            var libraryOverlay = document.getElementById('libraryOverlay');
            var closeLibrary = document.getElementById('closeLibrary');

            if (libraryBtn) libraryBtn.addEventListener('click', function() {
                if (libraryPanel) libraryPanel.classList.add('open');
                if (libraryOverlay) libraryOverlay.classList.add('open');
                SS.loadLibrary();
            });

            if (closeLibrary) closeLibrary.addEventListener('click', SS.closeLibraryPanel);
            if (libraryOverlay) libraryOverlay.addEventListener('click', SS.closeLibraryPanel);
        } catch (e) { console.error('Library panel setup failed:', e); }

        // Mixer Layout Toggle
        try {
            var mixerLayoutToggle = document.getElementById('mixerLayoutToggle');

            if (mixerLayoutToggle) {
                mixerLayoutToggle.checked = SS.useHierarchicalLayout;
                SS.updateLayoutOptionDisplay();

                mixerLayoutToggle.addEventListener('change', function() {
                    SS.useHierarchicalLayout = mixerLayoutToggle.checked;
                    localStorage.setItem('mixerLayout', SS.useHierarchicalLayout ? 'hierarchical' : 'dynamic');
                    SS.updateLayoutOptionDisplay();
                    SS.showToast(SS.useHierarchicalLayout ? '\u2713 Hierarchical layout enabled' : '\u2713 Dynamic layout enabled');
                });
            } else {
                SS.updateLayoutOptionDisplay();
            }
        } catch (e) { console.error('Mixer layout toggle setup failed:', e); }

        // Guitar Pro Tabs Toggle
        var gpTabsToggle = document.getElementById('gpTabsToggle');
        if (gpTabsToggle) {
            var savedGP = localStorage.getItem('gpTabs');
            gpTabsToggle.checked = savedGP === null ? true : savedGP === 'true';

            gpTabsToggle.addEventListener('change', function() {
                localStorage.setItem('gpTabs', gpTabsToggle.checked.toString());
                SS.showToast(gpTabsToggle.checked
                    ? '\u{1F3B8} Guitar Pro tabs enabled - will generate .gp5 files'
                    : '\u26A1 Guitar Pro tabs disabled - faster processing'
                );
            });
        }

        // Chord Detection Toggle
        var chordDetectionToggle = document.getElementById('chordDetectionToggle');
        if (chordDetectionToggle) {
            var savedChords = localStorage.getItem('chordDetection');
            chordDetectionToggle.checked = savedChords === null ? true : savedChords === 'true';

            chordDetectionToggle.addEventListener('change', function() {
                localStorage.setItem('chordDetection', chordDetectionToggle.checked.toString());
                SS.showToast(chordDetectionToggle.checked
                    ? '\u{1F3B5} Chord detection enabled - will analyze progressions'
                    : '\u26A1 Chord detection disabled - faster processing'
                );
            });
        }

        // Mobile menu toggle
        var mobileMenuBtn = document.getElementById('mobileMenuBtn');
        var navLinks = document.getElementById('navLinks');
        if (mobileMenuBtn && navLinks) {
            mobileMenuBtn.addEventListener('click', function() {
                navLinks.classList.toggle('open');
                mobileMenuBtn.textContent = navLinks.classList.contains('open') ? '\u2715' : '\u2630';
            });
        }

        // Google Drive connect button
        var connectDriveBtn = document.getElementById('connectDriveBtn');
        if (connectDriveBtn) connectDriveBtn.addEventListener('click', async function() {
            var btn = document.getElementById('connectDriveBtn');
            btn.textContent = '\u23F3 Authenticating...';
            btn.disabled = true;

            try {
                var response = await fetch(SS.API_BASE + '/drive/auth');
                var data = await response.json();

                if (data.status === 'authenticated') {
                    SS.showToast('\u2713 Connected to Google Drive!');
                    SS.checkDriveStatus();
                } else {
                    SS.showToast('Google Drive connection coming soon', true);
                    btn.textContent = '\u{1F517} Connect Google Drive';
                    btn.disabled = false;
                }
            } catch (error) {
                SS.showToast('Connection failed: ' + error.message, true);
                btn.textContent = '\u{1F517} Connect Google Drive';
                btn.disabled = false;
            }
        });

        // Cleanup button
        var cleanupBtn = document.getElementById('cleanupBtn');
        if (cleanupBtn) cleanupBtn.addEventListener('click', async function() {
            var btn = document.getElementById('cleanupBtn');
            var originalText = btn.textContent;
            btn.textContent = '\u{1F504} Cleaning up...';
            btn.disabled = true;

            try {
                var response = await fetch(SS.API_BASE + '/cleanup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_age_days: 7 })
                });
                var data = await response.json();

                if (data.freed_mb > 0) {
                    SS.showToast('\u2713 Freed ' + data.freed_mb + ' MB (' + data.deleted_files + ' files)');
                } else {
                    SS.showToast('\u2713 No old files to clean up');
                }
                SS.loadLocalStats();
            } catch (error) {
                SS.showToast('Cleanup failed: ' + error.message, true);
            }

            btn.textContent = originalText;
            btn.disabled = false;
        });
    };

    SS.closeSettingsPanel = function() {
        var sp = document.getElementById('settingsPanel');
        if (sp) sp.classList.remove('open');
        var so = document.getElementById('settingsOverlay');
        if (so) so.classList.remove('open');
    };

    SS.closeLibraryPanel = function() {
        var lp = document.getElementById('libraryPanel');
        if (lp) lp.classList.remove('open');
        var lo = document.getElementById('libraryOverlay');
        if (lo) lo.classList.remove('open');
    };

    SS.updateLayoutOptionDisplay = function() {
        var optA = document.getElementById('optionA');
        if (optA) optA.classList.toggle('active', !SS.useHierarchicalLayout);
        var optB = document.getElementById('optionB');
        if (optB) optB.classList.toggle('active', SS.useHierarchicalLayout);
    };

    SS.checkDriveStatus = async function() {
        var statusText = document.getElementById('driveStatusText');
        var statusDot = document.getElementById('driveDot');
        var connectBtn = document.getElementById('connectDriveBtn');
        var statsRow = document.getElementById('driveStatsRow');

        try {
            var response = await fetch(SS.API_BASE + '/drive/auth');
            var data = await response.json();

            if (data.status === 'authenticated') {
                if (statusText) statusText.textContent = 'Connected';
                if (statusDot) statusDot.className = 'status-dot green';
                if (connectBtn) { connectBtn.textContent = '\u2713 Connected to Google Drive'; connectBtn.disabled = true; }

                var statsResponse = await fetch(SS.API_BASE + '/drive/stats');
                var stats = await statsResponse.json();
                if (stats.exists) {
                    if (statsRow) statsRow.style.display = 'flex';
                    var dfc = document.getElementById('driveFileCount');
                    if (dfc) dfc.textContent = stats.folder_count + ' songs';
                }
            } else {
                if (statusText) statusText.textContent = 'Not connected';
                if (statusDot) statusDot.className = 'status-dot red';
                if (connectBtn) { connectBtn.textContent = '\u{1F517} Connect Google Drive'; connectBtn.disabled = false; }
            }
        } catch (error) {
            if (statusText) statusText.textContent = 'Unavailable';
            if (statusDot) statusDot.className = 'status-dot red';
            if (connectBtn) { connectBtn.textContent = '\u26A0\uFE0F Install Drive libraries first'; connectBtn.disabled = true; }
        }
    };

    SS.loadLocalStats = async function() {
        try {
            var response = await fetch(SS.API_BASE + '/jobs');
            var data = await response.json();
            var jobCount = data.jobs?.length || 0;
            var ljc = document.getElementById('localJobCount');
            if (ljc) ljc.textContent = jobCount + ' jobs';

            var estimatedMB = jobCount * 25;
            var maxMB = 1000;
            var percent = Math.min((estimatedMB / maxMB) * 100, 100);
            var df = document.getElementById('diskFill');
            if (df) df.style.width = percent + '%';
            var du = document.getElementById('diskUsed');
            if (du) du.textContent = '~' + estimatedMB + ' MB used';
        } catch (error) {
            var ljc2 = document.getElementById('localJobCount');
            if (ljc2) ljc2.textContent = 'Unable to fetch';
        }
    };

    SS.loadLibrary = async function() {
        var libraryList = document.getElementById('libraryList');
        libraryList.innerHTML = '<div class="library-loading">Loading library...</div>';

        try {
            var response = await fetch(SS.API_BASE + '/library');
            var data = await response.json();

            if (data.library && data.library.length > 0) {
                libraryList.innerHTML = data.library.map(function(item) {
                    // Parse artist from title ("Artist - Song") since YT metadata has uploader name
                    var libTitle = item.title || 'Unknown Track';
                    var libArtist = '';
                    var libDash = libTitle.indexOf(' - ');
                    if (libDash > 0) {
                        libArtist = libTitle.substring(0, libDash).trim();
                        libTitle = libTitle.substring(libDash + 3).trim();
                    }
                    return '<div class="library-item" onclick="StemScribe.loadFromLibrary(\'' + SS.escapeJsString(item.job_id) + '\')">' +
                        '<div class="library-item-header">' +
                            (item.thumbnail
                                ? '<img src="' + SS.escapeHtml(item.thumbnail) + '" class="library-item-thumb" alt="">'
                                : '<div class="library-item-thumb" style="display:flex;align-items:center;justify-content:center;font-size:1.5rem;">\u{1F3B5}</div>'
                            ) +
                            '<div class="library-item-info">' +
                                '<div class="library-item-title">' + SS.escapeHtml(libTitle) + '</div>' +
                                (libArtist ? '<div class="library-item-artist">' + SS.escapeHtml(libArtist) + '</div>' : '') +
                            '</div>' +
                            '<button class="library-item-delete" onclick="event.stopPropagation(); StemScribe.deleteFromLibrary(\'' + SS.escapeJsString(item.job_id) + '\')" title="Remove from library">\u{1F5D1}\uFE0F</button>' +
                        '</div>' +
                        '<div class="library-item-meta">' +
                            '<span>\u{1F39A}\uFE0F ' + item.stem_count + ' stems</span>' +
                            (item.has_midi ? '<span>\u{1F3B9} MIDI</span>' : '') +
                            (item.has_gp ? '<span>\u{1F3B8} Tabs</span>' : '') +
                            '<span>\u{1F4C5} ' + new Date(item.created_at * 1000).toLocaleDateString() + '</span>' +
                        '</div>' +
                    '</div>';
                }).join('');
            } else {
                libraryList.innerHTML =
                    '<div class="library-empty">' +
                        '<p>\u{1F4ED} No songs in your library yet</p>' +
                        '<p style="font-size:0.85rem; margin-top:0.5rem;">Process a song to add it to your library</p>' +
                    '</div>';
            }
        } catch (error) {
            console.error('Failed to load library:', error);
            libraryList.innerHTML = '<div class="library-empty">Failed to load library</div>';
        }
    };

    SS.loadFromLibrary = async function(jobId) {
        SS.closeLibraryPanel();
        SS.showToast('Loading from library...');

        try {
            var response = await fetch(SS.API_BASE + '/status/' + jobId);
            var job = await response.json();

            if (job.status === 'completed') {
                SS.currentJobId = jobId;
                SS.showResults(job);
                SS.showToast('\u2713 Loaded: ' + (job.metadata?.title || 'Track'));
            } else {
                SS.showToast('Job not ready', 'error');
            }
        } catch (error) {
            console.error('Failed to load from library:', error);
            SS.showToast('Failed to load track', 'error');
        }
    };

    SS.deleteFromLibrary = async function(jobId) {
        if (!confirm('Remove this song from your library? The files will be deleted.')) {
            return;
        }

        try {
            var response = await fetch(SS.API_BASE + '/library/' + jobId, {
                method: 'DELETE'
            });

            if (response.ok) {
                SS.showToast('Removed from library');
                SS.loadLibrary();
            } else {
                SS.showToast('Failed to delete', 'error');
            }
        } catch (error) {
            console.error('Failed to delete from library:', error);
            SS.showToast('Failed to delete', 'error');
        }
    };

    // Keyboard Shortcuts Modal
    SS.toggleShortcutsModal = function() {
        var modal = document.getElementById('shortcutsModal');
        if (modal) {
            var isOpen = modal.classList.contains('open');
            if (isOpen) {
                SS.hideShortcutsModal();
            } else {
                modal.classList.add('open');
            }
        }
    };

    SS.hideShortcutsModal = function() {
        var modal = document.getElementById('shortcutsModal');
        if (modal) {
            modal.classList.remove('open');
        }
    };

})(window.StemScribe);
