// StemScribe — Archive.org Live Music Browser
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    var archiveSelectedCollection = null;

    // Clean archive track titles — strip date prefixes like "gd69-05-23 t03 "
    function cleanTrackTitle(title) {
        if (!title) return title;
        // Strip patterns like "gd69-05-23 t03 " or "gd1972-08-21s1t08 "
        var cleaned = title
            .replace(/^[a-z]{2,4}\d{2,4}[-_]\d{2}[-_]\d{2}\s*[st]\d+[st]?\d*\s*/i, '')  // gd69-05-23 t03
            .replace(/^[a-z]{2,4}\d{2,4}[-_]\d{2}[-_]\d{2}\s+/i, '')  // gd69-05-23
            .replace(/^t\d+\s+/i, '')  // leftover "t03 "
            .replace(/^d\d+t\d+\s*/i, '')  // d1t08
            .trim();
        return cleaned || title;
    }

    SS.initArchive = function() {
        var archiveSearchInput = document.getElementById('archiveSearchInput');
        var archiveSearchBtn = document.getElementById('archiveSearchBtn');
        var archiveBackBtn = document.getElementById('archiveBackBtn');

        // Collection filter buttons
        try {
            document.querySelectorAll('.archive-filter').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    var wasActive = btn.classList.contains('active');
                    document.querySelectorAll('.archive-filter').forEach(function(b) { b.classList.remove('active'); });
                    if (!wasActive) {
                        btn.classList.add('active');
                        archiveSelectedCollection = btn.dataset.collection;
                        if (archiveSearchInput) {
                            var query = archiveSearchInput.value.trim() || btn.textContent.trim();
                            archiveSearchInput.value = query;
                        }
                        SS.performArchiveSearch();
                    } else {
                        archiveSelectedCollection = null;
                    }
                });
            });
        } catch (e) { console.error('Archive filter setup failed:', e); }

        if (archiveSearchBtn) archiveSearchBtn.addEventListener('click', SS.performArchiveSearch);
        if (archiveSearchInput) archiveSearchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') SS.performArchiveSearch();
        });

        if (archiveBackBtn) archiveBackBtn.addEventListener('click', function() {
            var sv = document.getElementById('archiveShowView');
            if (sv) sv.style.display = 'none';
            var ssv = document.getElementById('archiveSearchView');
            if (ssv) ssv.style.display = 'block';
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
                archiveResults.innerHTML = '<div class="archive-empty">No shows found for "' + SS.escapeHtml(query) + '". Note: not all bands are on Archive.org. Grateful Dead, Widespread Panic, Umphrey\'s McGee, String Cheese Incident, moe., and thousands of other bands have free live recordings available. Try searching by band name or date.</div>';
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

        if (archiveSearchView) archiveSearchView.style.display = 'none';
        if (archiveShowView) archiveShowView.style.display = 'block';
        if (archiveShowDetails) archiveShowDetails.innerHTML = '<div class="archive-loading">Loading show details...</div>';

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
                    var displayTitle = cleanTrackTitle(track.title || track.filename);
                    return '<div class="archive-track-row">' +
                        '<span class="archive-track-num">' + (track.track_number || (i + 1)) + '</span>' +
                        '<span class="archive-track-title">' + SS.escapeHtml(displayTitle) + '</span>' +
                        '<button class="archive-track-btn" onclick="StemScribe.processArchiveTrack(\'' + SS.escapeJsString(identifier) + '\', \'' + SS.escapeJsString(track.filename) + '\', \'' + SS.escapeJsString(displayTitle) + '\')">' +
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
        var _us = document.getElementById('uploadSection');
        if (_us) _us.style.display = 'none';
        var _ps = document.getElementById('processingSection');
        if (_ps) _ps.classList.add('active');
        var _rs = document.getElementById('resultsSection');
        if (_rs) _rs.classList.remove('active');

        SS.displayedProgress = 0;
        SS.targetProgress = 0;
        SS.lastProgressUpdate = Date.now();
        SS.updateProgressDisplay(0);
        var _st = document.getElementById('stageText');
        if (_st) _st.textContent = 'Processing: ' + title;
        SS.startProgressAnimation();
        document.querySelectorAll('.city').forEach(function(c) { c.classList.remove('visited'); });
        var _sf = document.getElementById('city-sf');
        if (_sf) _sf.classList.add('visited');

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
            var _stErr = document.getElementById('stageText');
            if (_stErr) _stErr.textContent = 'Error: ' + err.message;
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
                var _bus = document.getElementById('uploadSection');
                if (_bus) _bus.style.display = 'none';
                var _bps = document.getElementById('processingSection');
                if (_bps) _bps.classList.add('active');
                var _brs = document.getElementById('resultsSection');
                if (_brs) _brs.classList.remove('active');
                SS.displayedProgress = 0;
                SS.targetProgress = 0;
                SS.lastProgressUpdate = Date.now();
                SS.updateProgressDisplay(0);
                var _bst = document.getElementById('stageText');
                if (_bst) _bst.textContent = 'Processing: ' + data.jobs[0].title + ' (1 of ' + data.jobs.length + ')';
                SS.startProgressAnimation();
                document.querySelectorAll('.city').forEach(function(c) { c.classList.remove('visited'); });
                var _bsf = document.getElementById('city-sf');
                if (_bsf) _bsf.classList.add('visited');
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
