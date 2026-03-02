// StemScribe — Light/Dark Theme Toggle
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    SS.isDarkMode = true;

    SS.initTheme = function() {
        // Load saved preference
        var saved = localStorage.getItem('stemscribe-theme');
        if (saved === 'light') {
            SS.setTheme('light');
        }

        var toggleBtn = document.getElementById('themeToggleBtn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', function() {
                SS.toggleTheme();
            });
        }
    };

    SS.toggleTheme = function() {
        if (SS.isDarkMode) {
            SS.setTheme('light');
        } else {
            SS.setTheme('dark');
        }
    };

    SS.setTheme = function(theme) {
        var body = document.body;
        var toggleBtn = document.getElementById('themeToggleBtn');

        if (theme === 'light') {
            body.classList.add('light-mode');
            SS.isDarkMode = false;
            if (toggleBtn) toggleBtn.textContent = '\u2600\uFE0F';
            localStorage.setItem('stemscribe-theme', 'light');
        } else {
            body.classList.remove('light-mode');
            SS.isDarkMode = true;
            if (toggleBtn) toggleBtn.textContent = '\u{1F319}';
            localStorage.setItem('stemscribe-theme', 'dark');
        }

        // Update waveform colors
        if (SS.updateWaveformTheme) {
            SS.updateWaveformTheme(SS.isDarkMode);
        }
    };

})(window.StemScribe);
