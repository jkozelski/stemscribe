// StemScribe — Playback Speed Control
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.playbackRate = 1.0;

    var speedPresets = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
    var currentPresetIndex = 3; // 1.0x

    SS.initSpeedControl = function() {
        var speedBtn = document.getElementById('speedBtn');
        var speedSlider = document.getElementById('speedSlider');

        if (!speedBtn) return;

        // Click cycles through presets
        speedBtn.addEventListener('click', function() {
            currentPresetIndex = (currentPresetIndex + 1) % speedPresets.length;
            var rate = speedPresets[currentPresetIndex];
            SS.setPlaybackRate(rate);
        });

        // Long press shows slider
        var pressTimer = null;
        speedBtn.addEventListener('mousedown', function() {
            pressTimer = setTimeout(function() {
                speedSlider.style.display = 'block';
                speedSlider.value = SS.playbackRate * 100;
            }, 500);
        });

        speedBtn.addEventListener('mouseup', function() {
            clearTimeout(pressTimer);
        });

        speedBtn.addEventListener('mouseleave', function() {
            clearTimeout(pressTimer);
        });

        if (speedSlider) {
            speedSlider.addEventListener('input', function() {
                var rate = parseInt(speedSlider.value) / 100;
                SS.setPlaybackRate(rate);
            });

            speedSlider.addEventListener('change', function() {
                // Hide slider after releasing
                setTimeout(function() {
                    speedSlider.style.display = 'none';
                }, 1000);
            });
        }
    };

    SS.setPlaybackRate = function(rate) {
        SS.playbackRate = rate;

        // Update all stem audio elements
        Object.values(SS.stemAudios).forEach(function(stemData) {
            stemData.audio.playbackRate = rate;
        });

        // Update button text
        var speedBtn = document.getElementById('speedBtn');
        if (speedBtn) {
            var displayRate = rate % 1 === 0 ? rate.toFixed(0) : rate.toFixed(2).replace(/0+$/, '');
            speedBtn.textContent = displayRate + 'x';
            speedBtn.classList.toggle('modified', rate !== 1.0);
        }

        // Find nearest preset index
        var closest = 0;
        var minDiff = Infinity;
        speedPresets.forEach(function(p, i) {
            var diff = Math.abs(p - rate);
            if (diff < minDiff) {
                minDiff = diff;
                closest = i;
            }
        });
        currentPresetIndex = closest;
    };

})(window.StemScribe);
