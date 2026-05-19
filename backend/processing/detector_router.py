"""Detector router — Claude as the traffic cop.

Given a song title + artist, asks Claude which chord-detection path to use:
  - "jazz" → legacy stem-aware family-aware-consistency detector (nails Aja, Steely Dan,
    Jamiroquai per the Apr 25 sprint).
  - "general" → ACE + Jiang per-bar router (V3.1 default, best on rock/pop/blues).

Claude is good at "what kind of song is this?" because it knows the canonical chord
vocabulary for nearly every famous recording. It's bad at "what chord is in this audio
frame" — so we don't ask it to do that. Routing is its sweet spot.

Decisions are cached per (title_lower, artist_lower) tuple in a local JSON file. First
call to a new song costs ~$0.003; every subsequent upload of the same song hits cache
for free. Failure mode: any exception → return 'general' (the safe default path),
unless the outage allowlist matches a known jazz artist/title.

Pre-prod review hardening (2026-05-13, four-agent review):
  * P0 (B): 8s HTTP timeout on Anthropic call (post-sep semaphore was at risk of deadlock).
  * P0 (B): atomic cache writes (tmp + os.replace) — kill-mid-write no longer nukes cache.
  * P0 (B): fcntl.flock around read-modify-write — 4 concurrent workers no longer lose entries.
  * P1 (B): cache stamped with prompt-hash + model; mismatch invalidates entry.
  * P1 (B): low-confidence (< CACHE_CONFIDENCE_FLOOR) decisions returned but not persisted.
  * P1 (B): NFKD Unicode normalization — Mötley Crüe and Sigur Rós no longer collapse weirdly.
  * P1 (B): retry once on RateLimitError / 5xx.
  * P1 (B): tolerant JSON parser (direct → fence-strip → greedy-regex).
  * P1 (B): on 401, re-resolve API key once (handles keychain rotation without restart).
  * P1 (C): outage cascade — when API fails AND cache misses, a thin allowlist of canonical
    jazz artists/standards rescues known-jazz uploads from being silently routed to general.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level config -------------------------------------------------------

_CACHE_PATH = Path(
    os.environ.get(
        "DETECTOR_ROUTER_CACHE",
        str(Path(__file__).parent.parent / "data" / "detector_router_cache.json"),
    )
)
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # Cheap + fast; routing doesn't need the big model
_VALID_PATHS = {"jazz", "general"}
_REQUEST_TIMEOUT_S = 8.0          # P0 fix — bound semaphore exposure
_RETRY_ATTEMPTS = 2               # 1 initial + 1 retry on transient errors
_RETRY_SLEEP_S = 1.5
_CACHE_CONFIDENCE_FLOOR = 0.7     # don't persist anything below this
_VOICE_MEMO_HINT = re.compile(
    r"\b(voice memo|new recording|untitled|recording \d|session \d)\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = """\
You are routing audio chord-detection for StemScriber. Given a song title and artist, \
decide which detector path produces the best chord chart.

<paths>
"jazz" — Use when the song is jazz, jazz-fusion, smooth jazz, jazz-rock, or any genre \
where the canonical chord chart contains significant extension vocabulary (m7, m9, \
maj7, 9, 11, 13, m11, sus chords, slash chords). Examples:
  - Steely Dan: Aja, Peg, Black Cow, Do It Again, Reelin' in the Years, Dirty Work, Rikki
  - Jamiroquai: Cosmic Girl, Virtual Insanity, Alright
  - Pat Metheny, Weather Report, Herbie Hancock, smooth jazz, bossa nova
  - Songs explicitly built on m7-m7-7 changes throughout

"general" — Everything else. Rock, pop, blues, country, R&B, metal, classical, \
electronic, folk, singer-songwriter, indie. When unsure, choose "general". The general \
path uses a modern detector ensemble that's better than the jazz detector on these \
genres.
</paths>

<rules>
1. If you don't recognize the song with high confidence, choose "general".
2. If the song is a cover or live version, route based on the ORIGINAL's genre.
3. Borderline cases (jazz-rock hybrids, prog with extensions): choose "general" unless \
   the original is widely considered a jazz standard.
4. Return ONLY a single JSON object, no prose, no code fences.
</rules>

<output_schema>
{
  "path": "jazz" | "general",
  "confidence": float between 0 and 1,
  "reasoning": "one-line explanation, max 100 chars"
}
</output_schema>

<examples>
Input: "Aja" by "Steely Dan"
Output: {"path": "jazz", "confidence": 0.99, "reasoning": "Steely Dan jazz-fusion; m9/m11 throughout"}

Input: "Hotel California" by "Eagles"
Output: {"path": "general", "confidence": 0.95, "reasoning": "Classic rock; Bm-F#7-A-E vocabulary, not jazz-fusion"}

Input: "Into The Great Wide Open" by "Tom Petty and The Heartbreakers"
Output: {"path": "general", "confidence": 0.98, "reasoning": "Heartland rock; descending bass cliche but rock vocab"}

Input: "Free Fallin'" by "Tom Petty"
Output: {"path": "general", "confidence": 0.99, "reasoning": "Three-chord rock, F-Bb-C"}

Input: "Cosmic Girl" by "Jamiroquai"
Output: {"path": "jazz", "confidence": 0.97, "reasoning": "Acid-jazz/funk fusion; m7/m9 vamp"}

Input: "Untitled Demo" by "Unknown Artist"
Output: {"path": "general", "confidence": 0.3, "reasoning": "Unknown song; default general path"}
</examples>
"""

_PROMPT_HASH = hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]


# === API key resolution =====================================================

def _api_key() -> Optional[str]:
    """Resolve Anthropic API key from env or macOS keychain."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import subprocess
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "anthropic-api-key", "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.debug(
            "[detector_router] keychain lookup rc=%s stderr=%r",
            result.returncode, (result.stderr or "")[:120],
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[detector_router] keychain lookup failed: %s: %s", type(e).__name__, e)
    return None


# === Cache key normalization ================================================

_PAREN_RE = re.compile(r"\([^)]*\)")
_BRACK_RE = re.compile(r"\[[^\]]*\]")
_NOISE_SUFFIX_RE = re.compile(
    r"\b(remaster(ed)?|live|acoustic|demo|edit|version|mix|radio edit)\b.*$",
    re.IGNORECASE,
)
# Latin a-z + digits + common CJK + hiragana/katakana, everything else → space.
_KEEP_RE = re.compile(r"[^a-z0-9一-鿿぀-ヿ]+")
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    """NFKD + strip combining marks + drop bracketed annotations.

    "Mötley Crüe" → "motley crue"
    "Wonderwall (Live at Wembley)" → "wonderwall"
    "Sigur Rós" / "Sigur Ros" → both → "sigur ros"
    """
    if not s:
        return ""
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PAREN_RE.sub(" ", s)
    s = _BRACK_RE.sub(" ", s)
    s = _NOISE_SUFFIX_RE.sub("", s)
    s = _KEEP_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _cache_key(title: str, artist: str) -> str:
    return f"{_norm(title)}|||{_norm(artist)}"


# === Outage allowlist (P1 — Agent C) ========================================
# Thin safety net that only fires when (a) cache misses AND (b) API call fails
# after retries. Conservative by design: false-negatives (jazz routed to general
# during outage) are recoverable; false-positives (rock routed to jazz) regress
# the cohort F1. Keep this small.

_OUTAGE_JAZZ_ARTISTS = frozenset(_norm(a) for a in [
    "Steely Dan", "Jamiroquai", "Pat Metheny", "Weather Report",
    "Herbie Hancock", "Miles Davis", "John Coltrane", "Bill Evans",
    "Wayne Shorter", "Chick Corea", "Thelonious Monk", "Charlie Parker",
    "Dave Brubeck", "Chet Baker", "Ella Fitzgerald", "Sarah Vaughan",
    "Sonny Rollins", "Stan Getz", "Cannonball Adderley", "Wes Montgomery",
])
_OUTAGE_JAZZ_TITLES = frozenset(_norm(t) for t in [
    "misty", "autumn leaves", "all the things you are", "giant steps",
    "take five", "so what", "round midnight", "body and soul",
    "take the a train", "blue bossa", "stella by starlight", "blue in green",
    "my funny valentine", "summertime", "fly me to the moon",
])


def _outage_allowlist_lookup(title: str, artist: str) -> Optional[dict]:
    """Return a fallback decision if title/artist matches the outage allowlist."""
    artist_norm = _norm(artist)
    title_norm = _norm(title)
    if artist_norm and artist_norm in _OUTAGE_JAZZ_ARTISTS:
        return {
            "path": "jazz",
            "confidence": 0.6,
            "reasoning": "outage allowlist (artist)",
            "source": "fallback",
            "routing_status": "defaulted",
        }
    if title_norm and title_norm in _OUTAGE_JAZZ_TITLES:
        return {
            "path": "jazz",
            "confidence": 0.6,
            "reasoning": "outage allowlist (title)",
            "source": "fallback",
            "routing_status": "defaulted",
        }
    return None


# === Cache I/O ==============================================================

def _load_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[detector_router] cache file corrupt at %s (%s: %s) — starting empty",
            _CACHE_PATH, type(e).__name__, e,
        )
        return {}


def _atomic_write(cache: dict) -> None:
    """P0: write to tmp then os.replace — kill-mid-write cannot corrupt."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(_CACHE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
    os.replace(tmp, _CACHE_PATH)


def _save_entry(key: str, value: dict) -> None:
    """P0: file-locked read-modify-write so concurrent workers don't lose entries."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _CACHE_PATH.with_suffix(_CACHE_PATH.suffix + ".lock")
    try:
        import fcntl  # POSIX only — fine on Hetzner Linux + Jeff's Mac
        with open(lock_path, "a+") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                cache = _load_cache()
                cache[key] = value
                _atomic_write(cache)
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except ImportError:
        # Windows fallback — accept the small lost-update window.
        cache = _load_cache()
        cache[key] = value
        _atomic_write(cache)
    except Exception as e:  # noqa: BLE001
        logger.warning("[detector_router] cache save failed: %s: %s", type(e).__name__, e)


# === Cache versioning =======================================================

def _entry_is_fresh(entry: dict, model: str) -> bool:
    """An entry is fresh iff it was produced by the current model AND prompt."""
    if not isinstance(entry, dict):
        return False
    return entry.get("_model") == model and entry.get("_prompt_hash") == _PROMPT_HASH


# === JSON parsing — tolerant ===============================================

def _parse_router_json(text: str) -> Optional[dict]:
    """Direct parse → strip code fences → greedy regex → None."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    if fenced != text:
        try:
            return json.loads(fenced)
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# === Anthropic call =========================================================

def _build_client(api_key: str):
    """Construct an Anthropic client with a bounded timeout."""
    import anthropic
    # P0 fix: bound HTTP timeout. SDK default is 10 minutes — too long.
    return anthropic.Anthropic(api_key=api_key, timeout=_REQUEST_TIMEOUT_S)


def _call_claude(api_key: str, model: str, user_msg: str):
    """Make the Anthropic call with a single retry on transient 5xx / 429."""
    import anthropic
    last_err: Optional[Exception] = None
    current_key = api_key
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            client = _build_client(current_key)
            return client.messages.create(
                model=model,
                max_tokens=150,
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
            )
        except anthropic.AuthenticationError as e:
            # P1 fix: keychain may have rotated since process start. Re-resolve once.
            if attempt == 0:
                refreshed = _api_key()
                if refreshed and refreshed != current_key:
                    logger.warning("[detector_router] 401 — re-resolving API key (rotation?)")
                    current_key = refreshed
                    continue
            last_err = e
            break
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APITimeoutError,
                anthropic.APIConnectionError) as e:
            last_err = e
            logger.warning(
                "[detector_router] transient Anthropic error (attempt %d/%d): %s: %s",
                attempt + 1, _RETRY_ATTEMPTS, type(e).__name__, e,
            )
            if attempt + 1 < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_SLEEP_S)
                continue
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.error("[detector_router] unexpected Anthropic error: %s: %s",
                         type(e).__name__, e)
            break
    assert last_err is not None  # for type-checkers
    raise last_err


# === Public API =============================================================

def _fallback(reason: str, status: str = "errored", confidence: float = 0.0,
              title: str = "", artist: str = "") -> dict:
    """Return the safe-default decision, optionally upgraded by the outage allowlist."""
    allow = _outage_allowlist_lookup(title, artist) if (title or artist) else None
    if allow:
        allow["reasoning"] = f"{allow['reasoning']} ({reason})"
        return allow
    return {
        "path": "general",
        "confidence": confidence,
        "reasoning": reason,
        "source": "fallback",
        "routing_status": status,
    }


def _sanitize_for_prompt(s: str) -> str:
    """Defensive sanitization — limits prompt-injection blast radius."""
    return (s or "").replace('"', "'").replace("\n", " ").replace("\r", " ")[:200]


def route_detector(title: str, artist: str, *, model: str = _DEFAULT_MODEL) -> dict:
    """Return routing decision for a song.

    Returns dict with keys:
      path: "jazz" | "general"
      confidence: float in [0.0, 1.0]
      reasoning: str
      source: "cache" | "claude" | "fallback"
      routing_status: "routed" | "defaulted" | "errored"

    Always returns a valid path — never raises.
    """
    if not title and not artist:
        return _fallback("no metadata", status="defaulted")

    # P2 privacy hint: skip LLM for obviously personal recordings.
    if _VOICE_MEMO_HINT.search(title or ""):
        logger.info("[detector_router] voice-memo-like title — skipping LLM call")
        return _fallback("voice memo heuristic", status="defaulted",
                         title=title, artist=artist)

    key = _cache_key(title, artist)
    cache = _load_cache()
    cached = cache.get(key)
    if cached and _entry_is_fresh(cached, model):
        result = {k: v for k, v in cached.items() if not k.startswith("_")}
        result["source"] = "cache"
        result.setdefault("routing_status", "routed")
        return result

    api_key = _api_key()
    if not api_key:
        logger.error("[detector_router] no Anthropic key — defaulting to general")
        return _fallback("no API key", status="errored", title=title, artist=artist)

    try:
        import anthropic  # noqa: F401
    except ImportError:
        logger.error("[detector_router] anthropic SDK not installed — defaulting to general")
        return _fallback("anthropic SDK missing", status="errored",
                         title=title, artist=artist)

    safe_title = _sanitize_for_prompt(title)
    safe_artist = _sanitize_for_prompt(artist) or "Unknown"
    user_msg = f'Input: "{safe_title}" by "{safe_artist}"\nOutput:'

    try:
        resp = _call_claude(api_key, model, user_msg)
    except Exception as e:  # noqa: BLE001
        logger.error("[detector_router] Anthropic call failed after retries: %s: %s",
                     type(e).__name__, e)
        return _fallback(f"api error: {type(e).__name__}", status="errored",
                         title=title, artist=artist)

    text = ""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "")

    parsed = _parse_router_json(text)
    if parsed is None:
        logger.warning("[detector_router] unparsable response: %r", text[:160])
        return _fallback("unparsable response", status="errored",
                         title=title, artist=artist)

    path = parsed.get("path", "general")
    if path not in _VALID_PATHS:
        path = "general"

    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    decision = {
        "path": path,
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning", ""))[:200],
        "source": "claude",
        "routing_status": "routed",
    }

    # P1 fix: only persist confident decisions, and stamp with model + prompt hash.
    if confidence >= _CACHE_CONFIDENCE_FLOOR:
        persisted = {
            "path": decision["path"],
            "confidence": decision["confidence"],
            "reasoning": decision["reasoning"],
            "_model": model,
            "_prompt_hash": _PROMPT_HASH,
            "_cached_at": int(time.time()),
        }
        _save_entry(key, persisted)
    else:
        logger.info(
            "[detector_router] confidence %.2f below floor %.2f — not caching '%s' / '%s'",
            confidence, _CACHE_CONFIDENCE_FLOOR, safe_title, safe_artist,
        )

    return decision


# === Admin helpers ==========================================================

def invalidate_cache_entry(title: str, artist: str) -> bool:
    """Remove a single cache entry. Returns True if something was removed."""
    key = _cache_key(title, artist)
    cache = _load_cache()
    if key in cache:
        del cache[key]
        try:
            import fcntl
            lock_path = _CACHE_PATH.with_suffix(_CACHE_PATH.suffix + ".lock")
            with open(lock_path, "a+") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    _atomic_write(cache)
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except ImportError:
            _atomic_write(cache)
        return True
    return False


def clear_cache() -> int:
    """Wipe the cache. Returns number of entries removed."""
    cache = _load_cache()
    n = len(cache)
    _atomic_write({})
    return n


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "--clear":
        n = clear_cache()
        print(f"cleared {n} entries from {_CACHE_PATH}")
    elif len(sys.argv) >= 4 and sys.argv[1] == "--invalidate":
        ok = invalidate_cache_entry(sys.argv[2], sys.argv[3])
        print("removed" if ok else "no such entry")
    elif len(sys.argv) >= 3:
        result = route_detector(sys.argv[1], sys.argv[2])
        print(json.dumps(result, indent=2))
    else:
        print("Usage:")
        print("  detector_router.py <title> <artist>")
        print("  detector_router.py --invalidate <title> <artist>")
        print("  detector_router.py --clear")
        print("\nQuick smoke test:")
        for title, artist in [
            ("Aja", "Steely Dan"),
            ("Hotel California", "Eagles"),
            ("Mary Jane's Last Dance", "Tom Petty"),
            ("Cosmic Girl", "Jamiroquai"),
        ]:
            r = route_detector(title, artist)
            print(f"  {title} / {artist}: {r['path']} ({r['source']}, "
                  f"conf={r['confidence']}, status={r.get('routing_status')})")
            if r.get("reasoning"):
                print(f"    → {r['reasoning']}")
