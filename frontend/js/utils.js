// StemScribe — Utility Functions
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.escapeHtml = function(text) {
        if (!text) return '';
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };

    SS.escapeJsString = function(text) {
        if (!text) return '';
        return String(text)
            .replace(/\\/g, '\\\\')
            .replace(/'/g, "\\'")
            .replace(/"/g, '\\"')
            .replace(/\n/g, '\\n')
            .replace(/\r/g, '\\r')
            .replace(/</g, '\\x3c')
            .replace(/>/g, '\\x3e');
    };

    SS.formatTime = function(seconds) {
        var mins = Math.floor(seconds / 60);
        var secs = Math.floor(seconds % 60);
        return mins + ':' + secs.toString().padStart(2, '0');
    };

    SS.showToast = function(message, isError) {
        if (isError === undefined) isError = false;
        var toast = document.getElementById('toast');
        var toastIcon = document.getElementById('toastIcon');
        var toastMessage = document.getElementById('toastMessage');
        if (!toast) return;

        if (toastIcon) toastIcon.textContent = isError ? '\u26A0\uFE0F' : '\u2713';
        if (toastMessage) toastMessage.textContent = message;
        toast.classList.toggle('error', isError);
        toast.classList.add('show');

        setTimeout(function() { toast.classList.remove('show'); }, 3000);
    };

    SS.resetUI = function() {
        // Stop progress animation
        SS.stopProgressAnimation();

        // Stop VU meter animation
        if (SS.vuAnimationId) {
            cancelAnimationFrame(SS.vuAnimationId);
            SS.vuAnimationId = null;
        }

        // Destroy waveform
        if (SS.masterWavesurfer) {
            SS.masterWavesurfer.destroy();
            SS.masterWavesurfer = null;
        }
        var waveformContainer = document.getElementById('masterWaveformContainer');
        if (waveformContainer) waveformContainer.style.display = 'none';

        // Clear loop
        if (SS.clearLoopRegion) SS.clearLoopRegion();

        // Stop any playing audio — dispose Web Audio engine
        if (SS.audioEngine) {
            SS.audioEngine.dispose();
            SS.audioEngine = null;
        }
        SS.stemAudios = {};
        SS.analyserBuffers = {};
        SS.meterLevels = {};
        SS.isPlaying = false;

        var us = document.getElementById('uploadSection');
        if (us) us.style.display = 'block';
        var ps = document.getElementById('processingSection');
        if (ps) ps.classList.remove('active');
        var rs = document.getElementById('resultsSection');
        if (rs) rs.classList.remove('active');
        SS.selectedFile = null;
        var dt = document.querySelector('.drop-title');
        if (dt) dt.textContent = 'Drop your track here';
        var di = document.querySelector('.drop-icon');
        if (di) di.textContent = '\u{1F3A7}';
        var ui = document.getElementById('urlInput');
        if (ui) ui.value = '';
        var sb = document.getElementById('submitBtn');
        if (sb) sb.disabled = true;
    };

})(window.StemScribe);
