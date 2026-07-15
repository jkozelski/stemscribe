// StemScriber — Upload Options Modal
// Shown after file selection, before upload begins.
// Asks user what they want from the song, then sets the
// hidden gpTabsToggle + chordDetectionToggle inputs accordingly
// so the existing upload pipeline carries the choice through.

window.StemScriber = window.StemScriber || {};

(function(SS) {
    'use strict';

    // Choice keys
    var CHOICE_STEMS_ONLY = 'stems_only';
    var CHOICE_FULL = 'full';

    // Apply the choice by mutating the existing hidden/toggle inputs.
    // upload.js reads `gpTabsToggle.checked` and `chordDetectionToggle.checked`
    // at submit time — we set both to match the user's choice.
    function applyChoice(choice) {
        var gpTabs = document.getElementById('gpTabsToggle');
        var chord  = document.getElementById('chordDetectionToggle');

        var enableFull = (choice === CHOICE_FULL);

        // gpTabsToggle is a hidden input — ensure .checked works either way
        if (gpTabs) {
            try { gpTabs.checked = enableFull; } catch (e) { /* hidden inputs don't error, but guard anyway */ }
            // Also set the `value` for any code that reads it
            try { gpTabs.value = enableFull ? 'true' : 'false'; } catch (e) {}
        }
        if (chord) {
            try { chord.checked = enableFull; } catch (e) {}
        }

        // Stash choice for any downstream consumer / analytics
        try { SS.lastUploadChoice = choice; } catch (e) {}
    }

    // Public: open the modal and return a Promise that resolves to
    // the chosen option string, or null if the user cancels.
    SS.showUploadOptionsModal = function(fileName) {
        return new Promise(function(resolve) {
            var modal   = document.getElementById('uploadOptionsModal');
            var cards   = modal ? modal.querySelectorAll('.upload-option-card') : [];
            var fileEl  = document.getElementById('uploadOptionsFileName');
            var procBtn = document.getElementById('uploadOptionsProcessBtn');
            var cnclBtn = document.getElementById('uploadOptionsCancelBtn');
            var backdrop = modal ? modal.querySelector('.upload-options-backdrop') : null;

            if (!modal || !cards.length || !procBtn || !cnclBtn) {
                // Fallback: if markup is missing for any reason, default to stems-only
                // so we don't accidentally bill heavy work on a config mishap.
                console.warn('[upload-options-modal] markup missing, defaulting to stems-only');
                applyChoice(CHOICE_STEMS_ONLY);
                resolve(CHOICE_STEMS_ONLY);
                return;
            }

            // Reset state
            cards.forEach(function(c) { c.classList.remove('selected'); });
            procBtn.disabled = true;

            // Pre-select first card (stems-only) so the CTA is reachable on first interaction
            var defaultCard = modal.querySelector('.upload-option-card[data-choice="' + CHOICE_STEMS_ONLY + '"]');
            if (defaultCard) {
                defaultCard.classList.add('selected');
                procBtn.disabled = false;
            }

            if (fileEl && fileName) {
                fileEl.textContent = fileName;
            }

            // Show the modal
            modal.classList.add('open');
            document.body.classList.add('upload-options-open');

            var selectedChoice = defaultCard ? defaultCard.dataset.choice : null;

            // Card selection
            function onCardClick(e) {
                var card = e.currentTarget;
                cards.forEach(function(c) { c.classList.remove('selected'); });
                card.classList.add('selected');
                selectedChoice = card.dataset.choice;
                procBtn.disabled = false;
            }
            cards.forEach(function(c) { c.addEventListener('click', onCardClick); });

            // Keyboard support: arrow keys between options, Enter to confirm
            function onKey(e) {
                if (!modal.classList.contains('open')) return;
                if (e.key === 'Escape') {
                    e.preventDefault();
                    cleanup();
                    resolve(null);
                    return;
                }
                if (e.key === 'Enter') {
                    if (selectedChoice) {
                        e.preventDefault();
                        confirm();
                    }
                    return;
                }
                if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
                    e.preventDefault();
                    var idx = -1;
                    cards.forEach(function(c, i) { if (c.classList.contains('selected')) idx = i; });
                    var next;
                    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                        next = (idx + 1) % cards.length;
                    } else {
                        next = (idx - 1 + cards.length) % cards.length;
                    }
                    cards.forEach(function(c) { c.classList.remove('selected'); });
                    cards[next].classList.add('selected');
                    selectedChoice = cards[next].dataset.choice;
                    procBtn.disabled = false;
                }
            }
            document.addEventListener('keydown', onKey);

            // Cleanup helper
            function cleanup() {
                modal.classList.remove('open');
                document.body.classList.remove('upload-options-open');
                document.removeEventListener('keydown', onKey);
                cards.forEach(function(c) { c.removeEventListener('click', onCardClick); });
                procBtn.removeEventListener('click', confirm);
                cnclBtn.removeEventListener('click', cancel);
                if (backdrop) backdrop.removeEventListener('click', cancel);
            }

            function confirm() {
                if (!selectedChoice) return;
                applyChoice(selectedChoice);
                cleanup();
                resolve(selectedChoice);
            }

            function cancel() {
                cleanup();
                resolve(null);
            }

            procBtn.addEventListener('click', confirm);
            cnclBtn.addEventListener('click', cancel);
            if (backdrop) backdrop.addEventListener('click', cancel);
        });
    };

})(window.StemScriber);
