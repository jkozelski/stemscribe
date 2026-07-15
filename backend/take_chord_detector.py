"""Take-sized chord detector — for short solo recordings (session takes).

The song detectors (stem-aware, BTC/V8) assume full-length, full-band audio
and go blind on short sparse solo takes (7/6: a clean 19s G-Am-C-D take got
two silence-edge artifacts). This one does the simple thing that works:
windowed chroma -> chord-template match -> merge runs, with a silence gate.
"""
import numpy as np

_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _templates():
    out = []
    shapes = [
        ('', [0, 4, 7], [1.0, 0.9, 0.9]),
        ('m', [0, 3, 7], [1.0, 0.9, 0.9]),
        ('7', [0, 4, 7, 10], [1.0, 0.85, 0.8, 0.75]),
        ('maj7', [0, 4, 7, 11], [1.0, 0.85, 0.8, 0.75]),
        ('m7', [0, 3, 7, 10], [1.0, 0.85, 0.8, 0.75]),
    ]
    for suffix, ivs, ws in shapes:
        base = np.zeros(12)
        for iv, w in zip(ivs, ws):
            base[iv] = w
        for root in range(12):
            out.append((_NAMES[root] + suffix, np.roll(base, root)))
    return out


_TPL = _templates()


def detect_take_chords(wav_path, hop_sec=1.0, win_sec=2.0):
    """Return a list of {time, chord, confidence} events for a short take."""
    import librosa
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    dur = len(y) / sr
    if dur < 2.0:
        return []
    # silence gate: adaptive to the take's own loud parts
    frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    gate = max(0.01, float(np.percentile(frame_rms, 90)) * 0.15)

    events = []
    t = 0.0
    while t + win_sec <= dur + 0.5:
        a, b = int(t * sr), int(min(dur, t + win_sec) * sr)
        seg = y[a:b]
        if len(seg) < sr // 2:
            break
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < gate:
            t += hop_sec
            continue
        ch = librosa.feature.chroma_cqt(y=seg, sr=sr).mean(axis=1)
        n = np.linalg.norm(ch)
        if n < 1e-6:
            t += hop_sec
            continue
        ch = ch / n
        best, best_s = None, -1.0
        for name, tpl in _TPL:
            s = float(np.dot(ch, tpl / np.linalg.norm(tpl)))
            if s > best_s:
                best, best_s = name, s
        # prefer the plain triad when the 7th variant barely edges it out —
        # decaying acoustic overtones fake sevenths (G<->Gmaj7 flapping)
        scores = {name: float(np.dot(ch, tpl / np.linalg.norm(tpl))) for name, tpl in _TPL}
        if best.endswith('maj7') or best.endswith('m7') or best.endswith('7'):
            triad = best.replace('maj7', '').replace('m7', 'm').rstrip('7') or best
            if triad in scores and best_s - scores[triad] < 0.06:
                best, best_s = triad, scores[triad]
        if best_s < 0.72:
            t += hop_sec
            continue
        events.append({'time': round(t, 2), 'chord': best, 'confidence': round(best_s, 3)})
        t += hop_sec

    # merge consecutive same-chord windows; drop one-window blips between
    # identical neighbors (transition smear)
    merged = []
    for e in events:
        if merged and merged[-1]['chord'] == e['chord']:
            continue
        if (len(merged) >= 1 and events and
                merged[-1] is not events[0]):
            pass
        merged.append(e)
    # blip removal: (a) A B A one-hop sandwich -> A; (b) transition valleys —
    # a one-hop event less confident than BOTH neighbors is a window straddling
    # a chord change, not a chord
    cleaned = []
    for i, e in enumerate(merged):
        if 0 < i < len(merged) - 1:
            prev, nxt = merged[i - 1], merged[i + 1]
            one_hop = (nxt['time'] - e['time']) <= hop_sec + 0.01
            if prev['chord'] == nxt['chord'] and one_hop:
                continue
            if one_hop and e['confidence'] < prev['confidence'] and e['confidence'] < nxt['confidence']:
                continue
        cleaned.append(e)
    # re-merge after blip removal
    out = []
    for e in cleaned:
        if out and out[-1]['chord'] == e['chord']:
            continue
        out.append(e)
    return out


if __name__ == '__main__':
    import sys, json
    print(json.dumps(detect_take_chords(sys.argv[1]), indent=1))
