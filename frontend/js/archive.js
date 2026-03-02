// StemScribe — Archive.org Live Music Browser
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    var archiveSelectedCollection = null;

    SS.initArchive = function() {
        var archiveSearchInput = document.getElementById('archiveSearchInput');
        var archiveSearchBtn = document.getElementById('archiveSearchBtn');
        var archiveBackBtn = document.getElementById('archiveBackBtn');

        // Collection filter buttons
        document.querySelectorAll('.archive-filter').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var wasActive = btn.classList.contains('active');
                document.querySelectorAll('.archive-filter').forEach(function(b) { b.classList.remove('active'); });
                if (!wasActive) {
                    btn.classList.add('active');
                    archiveSelectedCollection = btn.dataset.collection;
                    var query = archiveSearchInput.value.trim() || btn.textContent.trim().split(' ').slice(1).join(' ');
                    archiveSearchInput.value = query;
                    SS.performArchiveSearch();
                } else {
                    archiveSelectedCollection = null;
                }
            });
        });

        archiveSearchBtn.addEventListener('click', SS.performArchiveSearch);
        archiveSearchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') SS.performArchiveSearch();
        });

        archiveBackBtn.addEventListener('click', function() {
            document.getElementById('archiveShowView').style.display = 'none';
            document.getElementById('archiveSearchView').style.display = 'block';
        });
    };

    SS.performArchiveSearch = async function() {
        var archiveSearchInput = document.getElementById('archiveSearchInput');
        var archiveSearchBtn = document.getElementById('archiveSearchBtn');
        var archiveResults = document.getElementById('archiveResults');
        var archiveYearInput = document.getElementById('archiveYearInput');

        var query = archiveSearchInput.value.trim();
        if (!query) return;

        var year = archiveYearInput.value.trim();
        archiveSearchBtn.disabled = true;
        archiveSearchBtn.textContent = '...';
        archiveResults.innerHTML = '<div class="archive-loading">\u{1F50D} Searching archive.org...</div>';

        try {
            var url = SS.API_BASE + '/archive/search?q=' + encodeURIComponent(query) + '&rows=30';
            if (archiveSelectedCollection) url += '&collection=' + archiveSelectedCollection;
            if (year) url += '&year=' + year;

            var resp = await fetch(url);
            var data = await resp.json();

            if (!resp.ok) {
                archiveResults.innerHTML = '<div class="archive-empty">Error: ' + SS.escapeHtml(data.error || 'Search failed') + '</div>';
                return;
            }

            if (!data.results || data.results.length === 0) {
                archiveResults.innerHTML = '<div class="archive-empty">No shows found. Try a different search.</div>';
                return;
            }

            archiveResults.innerHTML = data.results.map(function(show) {
                return '<div class="archive-show-card" onclick="StemScribe.loadArchiveShow(\'' + SS.escapeJsString(show.identifier) + '\')">' +
                    '<div class="archive-show-title">' + SS.escapeHtml(show.title) + '</div>' +
                    '<div class="archive-show-meta">' +
                        (show.date ? '<span>\u{1F4C5} ' + SS.escapeHtml(show.date) + '</span>' : '') +
                        (show.venue ? '<span>\u{1F4CD} ' + SS.escapeHtml(show.venue) + '</span>' : '') +
                        (show.avg_rating ? '<span class="archive-show-rating">\u2B50 ' + show.avg_rating.toFixed(1) + '</span>' : '') +
                        (show.source ? '<span>\u{1F399}\uFE0F ' + SS.escapeHtml(show.source.substring(0, 30)) + '</span>' : '') +
                    '</div>' +
                '</div>';
            }).join('');

        } catch (err) {
            archiveResults.innerHTML = '<div class="archive-empty">Search failed: ' + SS.escapeHtml(err.message) + '</div>';
        } finally {
            archiveSearchBtn.disabled = false;
            archiveSearchBtn.textContent = 'Search';
        }
    };

    SS.loadArchiveShow = async function(identifier) {
        var archiveSearchView = document.getElementById('archiveSearchView');
        var archiveShowView = document.getElementById('archiveShowView');
        var archiveShowDetails = document.getElementById('archiveShowDetails');

        archiveSearchView.style.display = 'none';
        archiveShowView.style.display = 'block';
        archiveShowDetails.innerHTML = '<div class="archive-loading">Loading show details...</div>';

        try {
            var resp = await fetch(SS.API_BASE + '/archive/show/' + encodeURIComponent(identifier));
            var data = await resp.json();

            if (!resp.ok) {
                archiveShowDetails.innerHTML = '<div class="archive-empty">Error: ' + SS.escapeHtml(data.error || 'Unknown error') + '</div>';
                return;
            }

            var show = data.show || {};
            var html =
                '<div style="margin-bottom: 1rem;">' +
                    '<div style="font-size: 1.1rem; font-weight: 600; color: var(--text);">' + SS.escapeHtml(show.title || identifier) + '</div>' +
                    '<div style="font-size: 0.8rem; color: var(--text-dim); margin-top: 0.25rem;">' +
                        (show.date ? '\u{1F4C5} ' + SS.escapeHtml(show.date) : '') + ' ' +
                        (show.venue ? '\u2022 \u{1F4CD} ' + SS.escapeHtml(show.venue) : '') + ' ' +
                        (show.source ? '\u2022 \u{1F399}\uFE0F ' + SS.escapeHtml(show.source) : '') +
                    '</div>' +
                '</div>';

            if (data.setlist && data.setlist.length > 0) {
                html += '<div style="font-size: 0.75rem; color: var(--psych-teal); margin-bottom: 0.5rem;">' +
                    'Setlist: ' + data.setlist.slice(0, 8).map(function(s) { return SS.escapeHtml(s); }).join(' \u2192 ') +
                    (data.setlist.length > 8 ? ' ...' : '') +
                '</div>';
            }

            if (data.tracks && data.tracks.length > 0) {
                html += '<div style="font-size: 0.75rem; color: var(--text-dim); margin-bottom: 0.5rem;">' + data.tracks.length + ' tracks available</div>';
                html += data.tracks.map(function(track, i) {
                    return '<div class="archive-track-row">' +
                        '<span class="archive-track-num">' + (track.track_number || (i + 1)) + '</span>' +
                        '<span class="archive-track-title">' + SS.escapeHtml(track.title || track.filename) + '</span>' +
                        '<button class="archive-track-btn" onclick="StemScribe.processArchiveTrack(\'' + SS.escapeJsString(identifier) + '\', \'' + SS.escapeJsString(track.filename) + '\', \'' + SS.escapeJsString(track.title || track.filename) + '\')">' +
                            '\u2726 Process' +
                        '</button>' +
                    '</div>';
                }).join('');

                html += '<button class="archive-process-all-btn" onclick="StemScribe.processArchiveBatch(\'' + SS.escapeJsString(identifier) + '\')">' +
                    '\u2726 Process All ' + data.tracks.length + ' Tracks' +
                '</button>';
            } else {
                html += '<div class="archive-empty">No audio tracks found for this show.</div>';
            }

            archiveShowDetails.innerHTML = html;

        } catch (err) {
            archiveShowDetails.innerHTML = '<div class="archive-empty">Failed to load show: ' + SS.escapeHtml(err.message) + '</div>';
        }
    };

    SS.processArchiveTrack = async function(identifier, filename, title) {
        document.getElementById('uploadSection').style.display = 'none';
        document.getElementById('processingSection').classList.add('active');
        document.getElementById('resultsSection').classList.remove('active');

        SS.displayedProgress = 0;
        SS.targetProgress = 0;
        SS.lastProgressUpdate = Date.now();
        SS.updateProgressDisplay(0);
        document.getElementById('stageText').textContent = 'Processing: ' + title;
        SS.startProgressAnimation();
        document.querySelectorAll('.city').forEach(function(c) { c.classList.remove('visited'); });
        document.getElementById('city-sf').classList.add('visited');

        try {
            var resp = await fetch(SS.API_BASE + '/archive/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    identifier: identifier,
                    filename: filename,
                    chord_detection: true,
                    gp_tabs: true,
                })
            });
            var data = await resp.json();
            if (data.job_id) {
                SS.currentJobId = data.job_id;
                SS.pollStatus();
            }
        } catch (err) {
            document.getElementById('stageText').textContent = 'Error: ' + err.message;
        }
    };

    SS.processArchiveBatch = async function(identifier) {
        try {
            var resp = await fetch(SS.API_BASE + '/archive/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    identifier: identifier,
                    chord_detection: true,
                    gp_tabs: true,
                })
            });
            var data = await resp.json();
            if (data.jobs && data.jobs.length > 0) {
                SS.currentJobId = data.jobs[0].job_id;
                document.getElementById('uploadSection').style.display = 'none';
                document.getElementById('processingSection').classList.add('active');
                document.getElementById('resultsSection').classList.remove('active');
                SS.displayedProgress = 0;
                SS.targetProgress = 0;
                SS.lastProgressUpdate = Date.now();
                SS.updateProgressDisplay(0);
                document.getElementById('stageText').textContent = 'Processing: ' + data.jobs[0].title + ' (1 of ' + data.jobs.length + ')';
                SS.startProgressAnimation();
                document.querySelectorAll('.city').forEach(function(c) { c.classList.remove('visited'); });
                document.getElementById('city-sf').classList.add('visited');
                SS.pollStatus();
                if (data.jobs.length > 1) {
                    SS.showToast('\u{1F3A4} ' + data.jobs.length + ' tracks queued from archive.org');
                }
            }
        } catch (err) {
            SS.showToast('Error: ' + err.message, true);
        }
    };

    // Expose globally for onclick handlers
    window.loadArchiveShow = SS.loadArchiveShow;

})(window.StemScribe);
