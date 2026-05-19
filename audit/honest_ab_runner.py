"""Honest A/B: 3 detectors x landmine on/off, through the REAL formatter,
scored with the v2 honest scorer. Read-only. No prod, no deploy."""
import sys, os, json, copy, io, contextlib
sys.path.insert(0, '/Users/jeffkozelski/stemscribe/backend')
sys.path.insert(0, '/Users/jeffkozelski/stemscribe/audit')

from score_chord_chart_v2 import score_v2, gt_bar_sequence

GT = {
    'pos': '/Users/jeffkozelski/stemscribe/audit/fixtures/ground_truth/bob-dylan__positively-4th-street.json',
    'iml': '/Users/jeffkozelski/stemscribe/audit/fixtures/ground_truth/the-beatles__in-my-life.json',
}
BASE = {'pos': '/tmp/forensic/pos', 'iml': '/tmp/forensic/iml'}


def load_events(name, kind):
    if kind == 'librosa':
        m = json.load(open(f'{BASE[name]}/job_metadata.json'))
        return copy.deepcopy(m['chord_progression'])
    d = json.load(open(f'{BASE[name]}/{kind}_events.json'))
    return [dict(time=e['time'], duration=e['duration'], chord=e['chord'],
                 root=e.get('root'), quality=e.get('quality', ''),
                 confidence=0.9) for e in d['events']]


def load_meta(name):
    return json.load(open(f'{BASE[name]}/job_metadata.json'))['metadata']


def load_words(name):
    return json.load(open(f'{BASE[name]}/word_ts.json'))


def run_format(name, events, disarm):
    prev = os.environ.get('CHART_FORMATTER_DISARM_SMOOTHING')
    if disarm:
        os.environ['CHART_FORMATTER_DISARM_SMOOTHING'] = 'true'
    else:
        os.environ.pop('CHART_FORMATTER_DISARM_SMOOTHING', None)
    try:
        import importlib
        import chart_formatter
        importlib.reload(chart_formatter)
        meta = load_meta(name)
        m = json.load(open(f'{BASE[name]}/job_metadata.json'))
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            chart = chart_formatter.format_chart(
                chord_events=copy.deepcopy(events),
                word_timestamps=copy.deepcopy(load_words(name)),
                title=name, artist='x',
                key=m.get('detected_key', 'C'),
                grid=copy.deepcopy(meta.get('grid')),
                bass_roots=copy.deepcopy(meta.get('bass_roots')),
            )
        return chart
    finally:
        if prev is None:
            os.environ.pop('CHART_FORMATTER_DISARM_SMOOTHING', None)
        else:
            os.environ['CHART_FORMATTER_DISARM_SMOOTHING'] = prev


def scoreline(name, chart):
    gt = json.load(open(GT[name]))
    r = score_v2(gt, chart)
    fl = r['axes']['3_flavor']['weighted_flavor']
    return dict(
        root=r['axes']['1_root']['f1'],
        pl_strict=r['axes']['2_placement']['strict_bar']['f1'],
        pl_holdinv=r['axes']['2_placement']['hold_invariant']['f1'],
        pl_bestoff=r['axes']['2_placement']['best_offset']['f1'],
        flavor=(round(fl, 3) if fl is not None else None),
        composite=r['composite']['composite'],
    )


CONFIGS = [
    ('librosa', 'librosa', False, 'librosa + landmine OFF  (PROD BASELINE)'),
    ('librosa', 'librosa', True,  'librosa + landmine DISARMED'),
    ('ace',     'ace',     True,  'ACE     + landmine DISARMED'),
    ('jiang',   'jiang',   True,  'Jiang   + landmine DISARMED'),
    # extra context: clean detectors WITH the landmine still armed
    ('ace',     'ace',     False, 'ACE     + landmine ON  (self-sabotaged)'),
    ('jiang',   'jiang',   False, 'Jiang   + landmine ON  (self-sabotaged)'),
]

results = {}
for song in ('iml', 'pos'):
    results[song] = []
    for key, kind, disarm, label in CONFIGS:
        try:
            ev = load_events(song, kind)
            ch = run_format(song, ev, disarm)
            sc = scoreline(song, ch)
            sc['_label'] = label
            sc['_n_events'] = len(ev)
            results[song].append(sc)
        except Exception as e:
            import traceback
            results[song].append({'_label': label, '_error': repr(e),
                                   '_tb': traceback.format_exc()[-500:]})

print(json.dumps(results, indent=2))
