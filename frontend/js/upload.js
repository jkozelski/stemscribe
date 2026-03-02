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
        document.querySelectorAll('.mode-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.mode-btn').forEach(function(b) { b.classList.remove('active'); });
                btn.classList.add('active');
                SS.currentMode = btn.dataset.mode;
                dropZone.style.display = SS.currentMode === 'file' ? 'block' : 'none';
                document.getElementById('urlSection').classList.toggle('active', SS.currentMode === 'url');
                document.getElementById('archiveSection').classList.toggle('active', SS.currentMode === 'archive');
                SS.updateSubmitBtn();
            });
        });

        // File handling
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

        urlInput.addEventListener('input', SS.updateSubmitBtn);

        submitBtn.addEventListener('click', SS.startProcessing);
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

        uploadSection.style.display = 'none';
        processingSection.classList.add('active');
        resultsSection.classList.remove('active');

        SS.displayedProgress = 0;
        SS.targetProgress = 0;
        SS.lastProgressUpdate = Date.now();
        SS.updateProgressDisplay(0);
        document.getElementById('stageText').textContent = 'Initializing...';

        SS.startProgressAnimation();

        document.querySelectorAll('.city').forEach(function(c) { c.classList.remove('visited'); });
        document.getElementById('city-sf').classList.add('visited');

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
            var gpTabs = document.getElementById('gpTabsToggle')?.checked ?? true;
            var chordDetection = document.getElementById('chordDetectionToggle')?.checked ?? true;

            if (SS.currentMode === 'file') {
                var formData = new FormData();
                formData.append('file', SS.selectedFile);
                formData.append('gp_tabs', gpTabs.toString());
                formData.append('chord_detection', chordDetection.toString());
                response = await fetch(SS.API_BASE + '/upload', { method: 'POST', body: formData });
            } else {
                response = await fetch(SS.API_BASE + '/url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: document.getElementById('urlInput').value.trim(),
                        gp_tabs: gpTabs,
                        chord_detection: chordDetection
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
