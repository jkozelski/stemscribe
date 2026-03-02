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

        toastIcon.textContent = isError ? '\u26A0\uFE0F' : '\u2713';
        toastMessage.textContent = message;
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

        // Stop any playing audio
        Object.values(SS.stemAudios).forEach(function(s) { s.audio.pause(); });
        SS.stemAudios = {};
        SS.analyserBuffers = {};
        SS.meterLevels = {};
        // Close shared audio context
        if (SS.mixerAudioCtx && SS.mixerAudioCtx.state !== 'closed') {
            SS.mixerAudioCtx.close().catch(function() {});
            SS.mixerAudioCtx = null;
        }
        SS.audioContexts = {};
        SS.isPlaying = false;

        document.getElementById('uploadSection').style.display = 'block';
        document.getElementById('processingSection').classList.remove('active');
        document.getElementById('resultsSection').classList.remove('active');
        SS.selectedFile = null;
        document.querySelector('.drop-title').textContent = 'Drop your track here';
        document.querySelector('.drop-icon').textContent = '\u{1F3A7}';
        document.getElementById('urlInput').value = '';
        document.getElementById('submitBtn').disabled = true;
    };

})(window.StemScribe);
