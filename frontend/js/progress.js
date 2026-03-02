// StemScribe — Progress Animation & Polling
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.updateProgressDisplay = function(progress) {
        document.getElementById('progressFill').style.width = progress + '%';
        document.getElementById('busTrail').style.width = progress + '%';
        document.getElementById('tourBus').style.left = progress + '%';
        document.getElementById('progressText').textContent = Math.round(progress) + '%';

        document.getElementById('city-denver').classList.toggle('visited', progress >= 25);
        document.getElementById('city-chicago').classList.toggle('visited', progress >= 55);
        document.getElementById('city-nyc').classList.toggle('visited', progress >= 90);
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

            document.getElementById('stageText').textContent = slim.stage || 'Processing...';

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
                alert('Failed: ' + (slim.error || 'Unknown error'));
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

})(window.StemScribe);
