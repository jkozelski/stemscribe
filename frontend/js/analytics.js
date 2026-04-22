// StemScriber — Plausible Analytics Custom Events
// No cookies, GDPR-compliant by default
// Docs: https://plausible.io/docs/custom-event-goals
(function() {
    'use strict';

    window.SS_Analytics = {
        // Fire a Plausible custom event (no-op if script blocked)
        track: function(eventName, props) {
            if (typeof window.plausible === 'function') {
                window.plausible(eventName, props ? { props: props } : undefined);
            }
        },

        // Pre-defined events
        songUploaded:       function()     { this.track('song_uploaded'); },
        songProcessed:      function()     { this.track('song_processed'); },
        practiceModeOpened:  function()     { this.track('practice_mode_opened'); },
        karaokeModeOpened:   function()     { this.track('karaoke_mode_opened'); },
        stemMuted:           function(stem) { this.track('stem_muted',  { stem: stem }); },
        stemSoloed:          function(stem) { this.track('stem_soloed', { stem: stem }); }
    };
})();
