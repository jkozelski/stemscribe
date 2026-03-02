// StemScribe — Configuration & Global State
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    // API endpoint
    SS.API_BASE = (window.location.port === '' || window.location.port === '80' || window.location.port === '443')
        ? window.location.origin + '/api'
        : window.location.protocol + '//' + window.location.hostname + ':5555/api';

    // Global state
    SS.currentMode = 'file';
    SS.audioContexts = {};
    SS.selectedFile = null;
    SS.currentJobId = null;
    SS.currentJob = null;
    SS.stemAudios = {};
    SS.isPlaying = false;
    SS.duration = 0;
    SS.soloedStems = new Set();
    SS.trackInfoLoaded = false;
    SS.currentTrackInfo = null;
    SS.mixerAudioCtx = null;
    SS.availableSkills = [];
    SS.analyserBuffers = {};
    SS.meterLevels = {};
    SS.vuAnimationId = null;

    // Progress interpolation state
    SS.displayedProgress = 0;
    SS.targetProgress = 0;
    SS.progressAnimationId = null;
    SS.lastProgressUpdate = Date.now();

    // Polling state
    SS.lastPollEtag = null;
    SS.lastPollJobId = null;

    // ETA state
    SS.processingStartTime = null;

    // Mixer layout
    SS.useHierarchicalLayout = localStorage.getItem('mixerLayout') === 'hierarchical';

    // Stem configuration
    SS.stemConfig = {
        // Primary 6-stem separation
        vocals: { icon: '\u{1F399}\uFE0F', color: '#ff6b9d', label: 'VOCALS' },
        drums: { icon: '\u{1F941}', color: '#e5c07b', label: 'DRUMS' },
        bass: { icon: '\u{1F50A}', color: '#ff7b54', label: 'BASS' },
        guitar: { icon: '\u{1F3B8}', color: '#61afef', label: 'GUITAR' },
        piano: { icon: '\u{1F3B9}', color: '#98c379', label: 'KEYS' },
        other: { icon: '\u{1F39B}\uFE0F', color: '#c678dd', label: 'OTHER' },

        // Smart deep extraction stems (auto-named by smart_separate)
        backing_vocals: { icon: '\u{1F3A4}', color: '#ff85b3', label: 'BG VOCALS', extracted: true },
        vocals_lead: { icon: '\u{1F399}\uFE0F', color: '#ff4070', label: 'LEAD VOX', extracted: true },
        vocals_backing: { icon: '\u{1F3A4}', color: '#ff85b3', label: 'BG VOCALS', extracted: true },
        guitar_lead: { icon: '\u{1F3B8}', color: '#61afef', label: 'LEAD GTR', extracted: true },
        guitar_rhythm: { icon: '\u{1F3B8}', color: '#5a9bd4', label: 'RHYTHM GTR', extracted: true },
        guitar_2: { icon: '\u{1F3B8}', color: '#71bfff', label: 'GUITAR 2', extracted: true },
        keys_2: { icon: '\u{1F3B9}', color: '#a8d389', label: 'KEYS 2', extracted: true },
        bass_2: { icon: '\u{1F50A}', color: '#ff9574', label: 'BASS 2', extracted: true },
        percussion_2: { icon: '\u{1FA98}', color: '#f5d08b', label: 'PERC 2', extracted: true },
        other_deep: { icon: '\u{1F3B7}', color: '#d688ed', label: 'EXTRAS', extracted: true },

        // Legacy cascaded stems (backwards compatible)
        other_vocals: { icon: '\u{1F3A4}', color: '#ff85b3', label: 'BG VOCALS', extracted: true },
        other_drums: { icon: '\u{1FA98}', color: '#f5d08b', label: 'PERCUSSION', extracted: true },
        other_bass: { icon: '\u{1F3B5}', color: '#ff9574', label: 'SUB BASS', extracted: true },
        other_guitar: { icon: '\u{1FA95}', color: '#71bfff', label: 'STRINGS', extracted: true },
        other_piano: { icon: '\u{1FA97}', color: '#a8d389', label: 'PADS', extracted: true },
        other_other: { icon: '\u{1F3B7}', color: '#d688ed', label: 'BRASS/FX', extracted: true },
    };

    // Sub-stem colors for skills output
    SS.subStemColors = {
        // Vocal Virtuoso
        vocals_lead: '#ff6b9d',
        vocals_harmony: '#ff85b3',
        vocals_high: '#ffa5c3',
        vocals_low: '#e55b8d',
        // Horn Hunter
        horns: '#ffd700',
        other_no_horns: '#c678dd',
        // Guitar God
        guitar_lead: '#61afef',
        guitar_rhythm: '#5a9bd4',
        guitar_clean: '#98c379',
        guitar_distorted: '#e06c75',
        // String Section
        strings_high: '#d19a66',
        strings_low: '#c18a56',
        // Keys Kingdom
        keys_acoustic: '#98c379',
        keys_electric: '#61dafb',
        keys_synth: '#c678dd'
    };

})(window.StemScribe);
