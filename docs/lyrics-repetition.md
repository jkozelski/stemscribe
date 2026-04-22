# Lyrics Repetition Detector

## Module
`backend/lyrics_repetition_detector.py`

## Purpose
Identifies choruses and song section boundaries by detecting repeating lyric patterns. Choruses repeat with the same or similar lyrics — this is a strong structural signal that complements chord-based and energy-based structure detection.

## API
```python
from lyrics_repetition_detector import detect_lyrics_repetition

result = detect_lyrics_repetition(lyrics)
# lyrics: list of {"text": str, "start_time": float, "end_time": float}
```

## Output
```python
{
    "chorus_lines": [
        {"text": "Turn it around now baby", "instances": [{"start": 46.0, "end": 50.0}, {"start": 86.0, "end": 90.0}]}
    ],
    "sections": [
        {"name": "Intro", "start_time": 0.0, "end_time": 21.5, "confidence": 0.70},
        {"name": "Verse", "start_time": 21.5, "end_time": 46.0, "confidence": 0.75},
        {"name": "Chorus", "start_time": 46.0, "end_time": 60.0, "confidence": 0.90},
        ...
    ]
}
```

## Algorithm
1. **Fuzzy matching** — Compare all lyric line pairs using `difflib.SequenceMatcher` (threshold: 0.7 similarity)
2. **Chorus block detection** — Find groups of 2+ consecutive lines that repeat together as a unit (e.g., lines 4-6 match lines 11-13)
3. **Section labeling** — Lines before first chorus = Verse, repeated blocks = Chorus, lines between choruses = Verse, post-chorus non-repeating = Bridge
4. **Gap detection** — Instrumental gaps >5s between lyrics insert Solo sections; gap before first lyric = Intro

## Tested Against
- "The Time Comes" (Kozelski) — correctly identifies 4 repeating chorus lines, 2 chorus sections, 1 solo (50s instrumental gap), and intro

## Tests
```bash
cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/test_lyrics_repetition_detector.py -v
```
29 tests covering normalize, similarity, repeating lines, chorus blocks, section detection, edge cases, and real lyrics integration.

## CLI
```bash
./venv311/bin/python backend/lyrics_repetition_detector.py outputs/<job-id>/lyrics.json
```
