// StemScribe — Progress Animation & Polling
window.StemScribe = window.StemScribe || {};

(function(SS) {
    'use strict';

    // ── Rock-and-roll trivia rotator for the stage text ─────────────────
    // Replaces technical backend messages ("Separating stems on cloud GPU (T4)")
    // with musician-friendly trivia. User doesn't need to know the mechanics.
    var ROCK_FACTS = [
        "Jimi Hendrix played guitar right-handed strung upside-down — left-handed but learned on his dad's right-handed guitar.",
        "The Beatles' \"A Day in the Life\" ends on an E-major chord played on three pianos at once — held for 40 seconds.",
        "Keith Richards wrote the riff to \"(I Can't Get No) Satisfaction\" in his sleep and recorded it on a cassette before forgetting it.",
        "Prince played every single instrument on his debut album \"For You\" — all 27 of them.",
        "Dave Grohl recorded every instrument on Foo Fighters' first album himself in under a week.",
        "John Bonham's kick-drum on \"When the Levee Breaks\" was recorded in a three-story stairwell.",
        "The opening drum break of \"Funky Drummer\" by James Brown is the most sampled drum loop in history.",
        "Stevie Wonder recorded \"Superstition\" entirely by himself — every drum, clav, bass, and vocal.",
        "Eddie Van Halen invented two-handed tapping after watching Jimmy Page play \"Heartbreaker\".",
        "Fleetwood Mac's \"Dreams\" was written by Stevie Nicks in about 10 minutes on a piano in Sly Stone's old studio.",
        "The bassline in Queen's \"Another One Bites the Dust\" was inspired by Chic's \"Good Times\".",
        "Chuck Berry only wrote one song in his life that went to #1: \"My Ding-a-Ling\".",
        "The Eagles' \"Hotel California\" solo took Don Felder and Joe Walsh three days to get right.",
        "Kurt Cobain wrote \"Smells Like Teen Spirit\" as an attempt to rip off the Pixies.",
        "Phil Collins learned drums by watching Buddy Rich — and later inherited his seat in Genesis.",
        "Robert Plant hit the high F5 at the end of \"Immigrant Song\" on the first take.",
        "Tom Petty turned down millions to license \"American Girl\" for commercials for 30 years.",
        "The Rolling Stones' \"Gimme Shelter\" features Merry Clayton, who was pregnant and miscarried that night.",
        "Jerry Garcia played with only nine fingers — he lost most of his right middle finger in a wood-chopping accident at age 4.",
        "Bruce Springsteen didn't release \"Born to Run\" until he'd recorded it over 50 times.",
        "Hendrix set his guitar on fire at Monterey Pop using lighter fluid and a box of kitchen matches.",
        "The guitar solo in \"Free Bird\" is 4 minutes 23 seconds long — longer than most pop songs entirely.",
        "Joni Mitchell invented over 50 unique guitar tunings because she couldn't press the strings hard due to childhood polio.",
        "Miles Davis' \"Kind of Blue\" was recorded with no rehearsal — the musicians saw the sketches for the first time in the studio.",
        "Motown house band The Funk Brothers played on more #1 hits than The Beatles, The Stones, and Elvis combined.",
        "Sly Stone's \"Everyday People\" was the first pop song to use an integrated band of mixed race AND gender.",
        "Tom Waits' voice dropped an octave between albums in 1983 — nobody knows exactly why.",
        "The entire album \"Dark Side of the Moon\" runs at exactly 42 minutes 49 seconds. It stayed on the Billboard 200 for 937 weeks.",
        "Paul McCartney wrote \"Yesterday\" in a dream and spent weeks convinced he'd stolen it from someone else.",
        "Aretha Franklin's \"Respect\" was originally an Otis Redding song — she flipped it and made it a feminist anthem in one take.",
        "The producer of \"Good Vibrations\" literally rented four different studios for one song so Brian Wilson could hear the room sounds he wanted."
    ];

    SS._factRotatorId = null;
    SS._factIndex = -1;
    SS._backendStage = null;
    SS.startRockFactRotator = function() {
        if (SS._factRotatorId) return;
        // Start on a random fact so reloads don't always see the same first one
        SS._factIndex = Math.floor(Math.random() * ROCK_FACTS.length);
        var showNext = function() {
            var el = document.getElementById('stageText');
            if (!el) return;
            SS._factIndex = (SS._factIndex + 1) % ROCK_FACTS.length;
            el.textContent = ROCK_FACTS[SS._factIndex];
        };
        showNext();
        SS._factRotatorId = setInterval(showNext, 22000);
    };
    SS.stopRockFactRotator = function() {
        if (SS._factRotatorId) { clearInterval(SS._factRotatorId); SS._factRotatorId = null; }
    };

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
            // Cap catchup speed so the bus actually crawls through intermediate numbers
            // on big backend jumps. 0.08% per frame = ~5%/sec at 60fps, so a 20->59
            // backend jump takes ~8 seconds of visible bus travel, not a lurch.
            var step = Math.max(0.05, Math.min(diff * 0.03, 0.08));
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

            // Keep the backend stage for debugging, but show rock trivia to the user.
            // People don't need to know the mechanics.
            SS._backendStage = slim.stage;
            if (!SS._factRotatorId) SS.startRockFactRotator();

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
                SS.stopRockFactRotator();
                if (window.SS_Analytics) SS_Analytics.songProcessed();
                var fullResp = await fetch(SS.API_BASE + '/status/' + SS.currentJobId);
                var job = await fullResp.json();
                SS.showResults(job);
            } else if (slim.status === 'failed') {
                SS.stopProgressAnimation();
                SS.stopRockFactRotator();
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
