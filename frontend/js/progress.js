// StemScribe — Progress Animation & Polling
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.updateProgressDisplay = function(progress) {
        var pf = document.getElementById('progressFill');
        if (pf) pf.style.width = progress + '%';
        var bt = document.getElementById('busTrail');
        if (bt) bt.style.width = progress + '%';
        var tb = document.getElementById('tourBus');
        if (tb) tb.style.left = progress + '%';
        var pt = document.getElementById('progressText');
        if (pt) pt.textContent = Math.round(progress) + '%';

        var cd = document.getElementById('city-denver');
        if (cd) cd.classList.toggle('visited', progress >= 25);
        var cc = document.getElementById('city-chicago');
        if (cc) cc.classList.toggle('visited', progress >= 55);
        var cn = document.getElementById('city-nyc');
        if (cn) cn.classList.toggle('visited', progress >= 90);
    };

    function animateProgress() {
        var now = Date.now();
        var timeSinceUpdate = now - SS.lastProgressUpdate;
        var creepRate = 0.0005;
        var maxCreep = Math.min(SS.targetProgress - 1, SS.displayedProgress + (timeSinceUpdate * creepRate));

        if (SS.displayedProgress < SS.targetProgress) {
            var diff = SS.targetProgress - SS.displayedProgress;
            var step = Math.max(0.1, diff * 0.15);
            SS.displayedProgress = Math.min(SS.targetProgress, SS.displayedProgress + step);
            SS.updateProgressDisplay(SS.displayedProgress);
        } else if (SS.displayedProgress < maxCreep && SS.targetProgress < 100) {
            SS.displayedProgress = Math.min(maxCreep, SS.displayedProgress + 0.05);
            SS.updateProgressDisplay(SS.displayedProgress);
        }

        SS.progressAnimationId = requestAnimationFrame(animateProgress);
    }

    SS.startProgressAnimation = function() {
        if (!SS.progressAnimationId) {
            SS.displayedProgress = 0;
            SS.targetProgress = 0;
            SS.lastProgressUpdate = Date.now();
            SS.progressAnimationId = requestAnimationFrame(animateProgress);
        }
    };

    SS.stopProgressAnimation = function() {
        if (SS.progressAnimationId) {
            cancelAnimationFrame(SS.progressAnimationId);
            SS.progressAnimationId = null;
        }
    };

    SS.pollStatus = async function() {
        if (SS.currentJobId !== SS.lastPollJobId) {
            SS.lastPollEtag = null;
            SS.lastPollJobId = SS.currentJobId;
        }
        try {
            var headers = {};
            if (SS.lastPollEtag) headers['If-None-Match'] = SS.lastPollEtag;

            var response = await fetch(
                SS.API_BASE + '/status/' + SS.currentJobId + '?slim=1',
                { headers: headers }
            );

            if (response.status === 304) {
                setTimeout(SS.pollStatus, 1000);
                return;
            }

            var slim = await response.json();
            SS.lastPollEtag = response.headers.get('ETag');

            var stEl = document.getElementById('stageText');
            if (stEl) stEl.textContent = slim.stage || 'Processing...';

            var progress = slim.progress || 0;
            if (progress > SS.targetProgress) {
                SS.targetProgress = progress;
                SS.lastProgressUpdate = Date.now();
            }

            // Update ETA
            SS.updateETA(progress);

            if (slim.status === 'completed') {
                SS.targetProgress = 100;
                SS.displayedProgress = 100;
                SS.updateProgressDisplay(100);
                SS.stopProgressAnimation();
                var fullResp = await fetch(SS.API_BASE + '/status/' + SS.currentJobId);
                var job = await fullResp.json();
                SS.showResults(job);
            } else if (slim.status === 'failed') {
                SS.stopProgressAnimation();
                var errMsg = slim.error || 'Unknown error';
                if (errMsg.indexOf('plan') !== -1 && errMsg.indexOf('minutes') !== -1) {
                    SS.showDurationLimitModal(errMsg);
                } else {
                    alert('Failed: ' + errMsg);
                }
                SS.resetUI();
            } else {
                var interval = progress > 0 && progress < 100 ? 500 : 1000;
                setTimeout(SS.pollStatus, interval);
            }
        } catch (error) {
            setTimeout(SS.pollStatus, 2000);
        }
    };

    SS.updateETA = function(progress) {
        var etaDisplay = document.querySelector('.eta-display');
        if (etaDisplay && SS.processingStartTime) {
            var elapsed = (Date.now() - SS.processingStartTime) / 1000 / 60;

            if (progress > 5) {
                var totalEstimate = elapsed / (progress / 100);
                var remaining = Math.max(0, totalEstimate - elapsed);

                if (remaining < 1) {
                    etaDisplay.innerHTML = 'Almost done... <span class="eta-time">< 1 min</span>';
                } else {
                    etaDisplay.innerHTML = 'Time remaining: <span class="eta-time">~' + Math.ceil(remaining) + ' min</span>';
                }
            }
        }
    };

    SS.showDurationLimitModal = function(message) {
        var overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.8);z-index:9999;display:flex;align-items:center;justify-content:center;';
        var modal = document.createElement('div');
        modal.style.cssText = 'background:#1a1a24;border:1px solid #2a2a35;border-radius:16px;padding:2.5rem;max-width:440px;width:90%;text-align:center;color:#e8e4df;font-family:Space Grotesk,sans-serif;';
        modal.innerHTML = '<div style="font-size:2.5rem;margin-bottom:1rem;">⏱️</div>' +
            '<h3 style="font-size:1.3rem;margin-bottom:1rem;background:linear-gradient(135deg,#ff7b54,#ff6b9d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Song Too Long</h3>' +
            '<p style="color:#7a7a85;font-size:0.95rem;line-height:1.6;margin-bottom:1.5rem;">' + message + '</p>' +
            '<a href="/#pricing" style="display:inline-block;padding:0.8rem 2rem;background:linear-gradient(135deg,#ff7b54,#ff6b9d);border:none;border-radius:10px;color:white;font-weight:600;text-decoration:none;font-size:1rem;margin-bottom:0.75rem;">View Plans</a>' +
            '<br><button style="background:none;border:none;color:#7a7a85;cursor:pointer;margin-top:0.5rem;font-size:0.9rem;" onclick="this.closest(\'div[style*=fixed]\').remove()">Maybe later</button>';
        overlay.appendChild(modal);
        overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.remove(); });
        document.body.appendChild(overlay);
    };

})(window.StemScribe);
