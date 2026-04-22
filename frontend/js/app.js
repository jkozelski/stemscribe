// StemScriber — Application Entry Point / Initialization
window.StemScriber = window.StemScriber || {};

(function(SS) {
    'use strict';

    // Load skills on page load
    SS.loadSkills = async function() {
        try {
            var res = await fetch(SS.API_BASE + '/skills');
            var data = await res.json();
            if (data.available && data.skills.length > 0) {
                SS.availableSkills = data.skills;
                console.log('Skills loaded:', SS.availableSkills.map(function(s) { return s.name; }).join(', '));
            }
        } catch (e) {
            console.log('Skills info not available:', e);
        }
    };

    // Initialize all modules
    SS.loadSkills();
    if (SS.initAuth) SS.initAuth();
    SS.initUpload();
    SS.initArchive();
    SS.initPanels();
    SS.initLoopControls();
    SS.initSpeedControl();
    SS.initTheme();

})(window.StemScriber);
