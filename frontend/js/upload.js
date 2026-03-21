// StemScribe — Upload / File Handling / URL Input
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.initUpload = function() {
        var dropZone = document.getElementById('dropZone');
        var fileInput = document.getElementById('fileInput');
        var urlInput = document.getElementById('urlInput');
        var submitBtn = document.getElementById('submitBtn');

        // Mode toggle
        try {
            document.querySelectorAll('.mode-btn').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    document.querySelectorAll('.mode-btn').forEach(function(b) { b.classList.remove('active'); });
                    btn.classList.add('active');
                    SS.currentMode = btn.dataset.mode;
                    if (dropZone) dropZone.style.display = SS.currentMode === 'file' ? 'block' : 'none';
                    var urlSec = document.getElementById('urlSection');
                    if (urlSec) urlSec.classList.toggle('active', SS.currentMode === 'url');
                    var archSec = document.getElementById('archiveSection');
                    if (archSec) archSec.classList.toggle('active', SS.currentMode === 'archive');
                    SS.updateSubmitBtn();
                });
            });
        } catch (e) { console.error('Mode toggle setup failed:', e); }

        // File handling
        try {
            if (dropZone && fileInput) {
                dropZone.addEventListener('click', function() { fileInput.click(); });
                dropZone.addEventListener('dragover', function(e) { e.preventDefault(); dropZone.classList.add('dragover'); });
                dropZone.addEventListener('dragleave', function() { dropZone.classList.remove('dragover'); });
                dropZone.addEventListener('drop', function(e) {
                    e.preventDefault();
                    dropZone.classList.remove('dragover');
                    if (e.dataTransfer.files.length) SS.handleFile(e.dataTransfer.files[0]);
                });
                fileInput.addEventListener('change', function(e) {
                    if (e.target.files.length) SS.handleFile(e.target.files[0]);
                });
            }
        } catch (e) { console.error('File input setup failed:', e); }

        // URL input
        try {
            if (urlInput) {
                urlInput.addEventListener('input', SS.updateSubmitBtn);
                urlInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && urlInput.value.trim()) {
                        e.preventDefault();
                        SS.startProcessing();
                    }
                });
            }
        } catch (e) { console.error('URL input setup failed:', e); }

        // Submit button
        try {
            if (submitBtn) submitBtn.addEventListener('click', SS.startProcessing);
        } catch (e) { console.error('Submit button setup failed:', e); }
    };

    SS.handleFile = function(file) {
        SS.selectedFile = file;
        document.querySelector('.drop-title').textContent = file.name;
        document.querySelector('.drop-icon').textContent = '\u2713';
        SS.updateSubmitBtn();
    };

    SS.updateSubmitBtn = function() {
        var submitBtn = document.getElementById('submitBtn');
        if (SS.currentMode === 'archive') {
            submitBtn.style.display = 'none';
        } else {
            submitBtn.style.display = '';
            submitBtn.disabled = SS.currentMode === 'file' ? !SS.selectedFile : !document.getElementById('urlInput').value.trim();
        }
    };

    SS.startProcessing = async function() {
        var uploadSection = document.getElementById('uploadSection');
        var processingSection = document.getElementById('processingSection');
        var resultsSection = document.getElementById('resultsSection');

        if (uploadSection) uploadSection.style.display = 'none';
        if (processingSection) processingSection.classList.add('active');
        if (resultsSection) resultsSection.classList.remove('active');

        SS.displayedProgress = 0;
        SS.targetProgress = 0;
        SS.lastProgressUpdate = Date.now();
        SS.updateProgressDisplay(0);
        var stageTextEl = document.getElementById('stageText');
        if (stageTextEl) stageTextEl.textContent = 'Initializing...';

        SS.startProgressAnimation();

        document.querySelectorAll('.city').forEach(function(c) { c.classList.remove('visited'); });
        var sfEl = document.getElementById('city-sf');
        if (sfEl) sfEl.classList.add('visited');

        // Track processing start time for ETA
        SS.processingStartTime = Date.now();

        // Add ETA display if not exists
        var etaDisplay = document.querySelector('.eta-display');
        if (!etaDisplay) {
            etaDisplay = document.createElement('div');
            etaDisplay.className = 'eta-display';
            etaDisplay.innerHTML = 'Estimated time: <span class="eta-time">~10-15 min</span>';
            document.querySelector('.progress-container').after(etaDisplay);
        }

        try {
            var response;
            var isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
            // Mobile: skip heavy Guitar Pro tab generation (still do chord detection + Songsterr)
            var gpTabs = isMobile ? false : (document.getElementById('gpTabsToggle')?.checked ?? true);
            var chordDetection = document.getElementById('chordDetectionToggle')?.checked ?? true;

            // Determine user plan from localStorage
            var betaData = JSON.parse(localStorage.getItem('stemscribe-beta') || 'null');
            var userPlan = (betaData && betaData.plan) ? betaData.plan : 'free';

            if (SS.currentMode === 'file') {
                var formData = new FormData();
                formData.append('file', SS.selectedFile);
                formData.append('gp_tabs', gpTabs.toString());
                formData.append('chord_detection', chordDetection.toString());
                formData.append('plan', userPlan);
                response = await fetch(SS.API_BASE + '/upload', { method: 'POST', body: formData });
            } else {
                response = await fetch(SS.API_BASE + '/url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: document.getElementById('urlInput').value.trim(),
                        gp_tabs: gpTabs,
                        chord_detection: chordDetection,
                        plan: userPlan
                    })
                });
            }
            var data = await response.json();
            if (data.error) throw new Error(data.error);
            SS.currentJobId = data.job_id;
            SS.pollStatus();
        } catch (error) {
            alert('Error: ' + error.message);
            SS.resetUI();
        }
    };

})(window.StemScribe);
