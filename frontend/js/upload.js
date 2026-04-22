// StemScriber — Upload / File Handling / URL Input
window.StemScriber = window.StemScriber || {};

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

        // Drag-and-drop on drop zone (no click handler — buttons are separate)
        try {
            if (dropZone) {
                dropZone.addEventListener('dragover', function(e) { e.preventDefault(); dropZone.classList.add('dragover'); });
                dropZone.addEventListener('dragleave', function() { dropZone.classList.remove('dragover'); });
                dropZone.addEventListener('drop', function(e) {
                    e.preventDefault();
                    dropZone.classList.remove('dragover');
                    if (e.dataTransfer.files.length) SS.handleFile(e.dataTransfer.files[0]);
                });
            }
        } catch (e) { console.error('Drop zone setup failed:', e); }

        // File input change handler
        try {
            if (fileInput) {
                fileInput.addEventListener('change', function(e) {
                    if (e.target.files.length) SS.handleFile(e.target.files[0]);
                });
            }
        } catch (e) { console.error('File input setup failed:', e); }

        // Action buttons — all trigger the file picker
        try {
            ['browseFilesBtn', 'musicLibraryBtn', 'googleDriveBtn', 'icloudBtn'].forEach(function(id) {
                var btn = document.getElementById(id);
                if (btn) btn.addEventListener('click', function() { fileInput.click(); });
            });
        } catch (e) { console.error('Action button setup failed:', e); }

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

    // ── Upload Consent Gate ──
    // Set to true to require consent before first upload.
    // When false, the consent modal is never shown (disabled).
    SS.REQUIRE_UPLOAD_CONSENT = true;

    SS.hasUploadConsent = function() {
        if (!SS.REQUIRE_UPLOAD_CONSENT) return true;
        try {
            var consent = JSON.parse(sessionStorage.getItem('stemscriber-upload-consent') || 'null');
            return consent && consent.agreed === true;
        } catch (e) { return false; }
    };

    SS.saveUploadConsent = function() {
        sessionStorage.setItem('stemscriber-upload-consent', JSON.stringify({
            agreed: true,
            timestamp: new Date().toISOString(),
            version: '1.2'
        }));
    };

    SS.showConsentModal = function() {
        return new Promise(function(resolve) {
            // Build modal
            var overlay = document.createElement('div');
            overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:10000;display:flex;align-items:center;justify-content:center;';

            var modal = document.createElement('div');
            modal.style.cssText = 'background:#1a1a24;border:1px solid #2a2a35;border-radius:16px;padding:2rem;max-width:480px;width:90%;color:#e8e4df;font-family:Space Grotesk,sans-serif;';

            modal.innerHTML = '<h2 style="font-family:Righteous,cursive;color:#ff7b54;margin-bottom:1rem;font-size:1.3rem;">Before You Upload</h2>' +
                '<p style="font-size:0.9rem;line-height:1.6;color:#aaa;margin-bottom:1.5rem;">' +
                'By submitting audio or a YouTube URL, you confirm that you <strong style="color:#e8e4df;">own this content or have the legal right to submit it for processing</strong>. ' +
                'StemScriber is an audio processing service &mdash; we do not host your source audio, we do not claim ownership of your content, and <strong style="color:#e8e4df;">we do not use your content to train AI models</strong>. ' +
                'You remain responsible for how you use the output.</p>' +
                '<div style="display:flex;gap:0.75rem;justify-content:flex-end;">' +
                '<button id="consentCancel" style="background:transparent;border:1px solid #2a2a35;border-radius:8px;color:#7a7a85;padding:0.6rem 1.2rem;cursor:pointer;font-family:Space Grotesk,sans-serif;font-size:0.85rem;">Cancel</button>' +
                '<button id="consentAgree" style="background:linear-gradient(135deg,#1a6a3a,#2a8a4a);border:none;border-radius:8px;color:#fff;padding:0.6rem 1.5rem;cursor:pointer;font-family:Space Grotesk,sans-serif;font-size:0.85rem;font-weight:600;">I Agree — Let Me Upload</button>' +
                '</div>';

            overlay.appendChild(modal);
            document.body.appendChild(overlay);

            document.getElementById('consentAgree').addEventListener('click', function() {
                SS.saveUploadConsent();
                document.body.removeChild(overlay);
                resolve(true);
            });

            document.getElementById('consentCancel').addEventListener('click', function() {
                document.body.removeChild(overlay);
                resolve(false);
            });

            overlay.addEventListener('click', function(e) {
                if (e.target === overlay) {
                    document.body.removeChild(overlay);
                    resolve(false);
                }
            });
        });
    };

    SS.startProcessing = async function() {
        // Check upload consent (disabled by default — flip SS.REQUIRE_UPLOAD_CONSENT to enable)
        if (!SS.hasUploadConsent()) {
            var consented = await SS.showConsentModal();
            if (!consented) return;
        }

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
        // Start rock-trivia rotator immediately — the first poll will keep it going.
        if (SS.startRockFactRotator) SS.startRockFactRotator();

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
            if (window.SS_Analytics) SS_Analytics.songUploaded();
            SS.pollStatus();
        } catch (error) {
            alert('Error: ' + error.message);
            SS.resetUI();
        }
    };

})(window.StemScriber);
