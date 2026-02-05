# StemScribe Enhancement Plan
## Comprehensive Feature Roadmap

---

## 1. 🎚️ BETTER STEM SEPARATION

### 1.1 Multi-Model Support
**Current**: Only htdemucs_6s model
**Enhancement**: Allow users to choose from multiple Demucs models

```python
DEMUCS_MODELS = {
    'htdemucs_6s': {'stems': 6, 'quality': 'balanced', 'speed': 'fast'},
    'htdemucs_ft': {'stems': 4, 'quality': 'high', 'speed': 'slow'},
    'mdx_extra': {'stems': 4, 'quality': 'very_high', 'speed': 'very_slow'},
    'demucs_v4': {'stems': 4, 'quality': 'high', 'speed': 'medium'},
}
```

**Implementation**: Add model dropdown to UI, pass to backend

### 1.2 Improved Cascade Separation
**Current**: Basic re-run of Demucs on "other" stem
**Enhancement**: Smart cascaded separation with instrument detection

- Run Demucs again on stems with high "other" content
- Use audio fingerprinting to detect specific instruments before processing
- Add "cascade depth" option (1-3 levels)

### 1.3 Stem Bleed Reduction
**Enhancement**: Post-processing to reduce cross-stem bleed

```python
def reduce_bleed(target_stem, other_stems, aggressiveness=0.5):
    """
    Use spectral subtraction to remove content from target
    that appears in other stems
    """
```

### 1.4 Stems Quality Score
**Enhancement**: Score each stem's isolation quality

- Calculate spectral similarity between stems (lower = better separation)
- Show quality badges in UI (Excellent/Good/Fair/Poor)
- Auto-suggest re-processing for poor stems

---

## 2. 🎸 EXPANDED BAND/ARTIST SUPPORT

### 2.1 More Local Knowledge Bands
**Add to `track_info.py`:**

```python
LOCAL_KNOWLEDGE = {
    # JAM BANDS
    'widespread panic': {...},
    'moe.': {...},
    'string cheese incident': {...},
    'umphrey's mcgee': {...},
    'goose': {...},
    'billy strings': {...},
    'pigeons playing ping pong': {...},

    # CLASSIC ROCK
    'cream': {...},
    'jimi hendrix experience': {...},
    'the doors': {...},
    'santana': {...},
    'deep purple': {...},
    'black sabbath': {...},

    # SOUTHERN ROCK
    'lynyrd skynyrd': {...},
    'the marshall tucker band': {...},
    'molly hatchet': {...},

    # PROG ROCK
    'yes': {...},
    'genesis': {...},
    'king crimson': {...},
    'rush': {...},
    'tool': {...},
}
```

### 2.2 Setlist.fm Integration
**Purpose**: Identify live recordings and get setlist context

```python
def fetch_setlist_info(artist: str, date: str = None, venue: str = None):
    """
    Query setlist.fm API to find show details
    Returns: setlist, venue, tour name, notable moments
    """
```

### 2.3 Discogs Integration
**Purpose**: Album art, release variations, production credits

```python
def fetch_discogs_info(artist: str, album: str):
    """
    Get detailed production info from Discogs
    Returns: producer, engineer, studio, personnel, formats
    """
```

### 2.4 Live Recording Detection
**Enhancement**: Auto-detect if track is live vs studio

```python
def is_live_recording(audio_path: str, metadata: dict) -> bool:
    """
    Detect live recordings by:
    - Crowd noise at start/end
    - Audience applause detection
    - Title patterns ("Live at", date patterns)
    - Audio characteristics (room reverb, bleed)
    """
```

### 2.5 Era-Specific Lineups (More Bands)
**Add era databases for:**
- Fleetwood Mac (Peter Green → Bob Welch → Buckingham/Nicks)
- Genesis (Gabriel → Collins → Ray Wilson)
- Yes (multiple keyboardists/guitarists)
- Allman Brothers (post-Duane, Warren Haynes era)
- Phish (different periods)

---

## 3. 🎨 UI/UX IMPROVEMENTS

### 3.1 Waveform Visualization
**Enhancement**: Show audio waveforms for each stem

```javascript
// Using wavesurfer.js
const wavesurfer = WaveSurfer.create({
    container: '#waveform-' + stemName,
    waveColor: stemColors[stemName],
    progressColor: '#ff7b54',
    responsive: true,
    height: 60,
});
```

### 3.2 Player Name Labels on Mixer
**Enhancement**: Show actual player names on channel strips

```
┌──────────┬──────────┬──────────┐
│ Jerry    │ Bob      │ Phil     │
│ Garcia   │ Weir     │ Lesh     │
│ 🎸 LEFT  │ 🎸 RIGHT │ 🎸 BASS  │
├──────────┼──────────┼──────────┤
│ [FADER]  │ [FADER]  │ [FADER]  │
│ [M] [S]  │ [M] [S]  │ [M] [S]  │
└──────────┴──────────┴──────────┘
```

### 3.3 A/B Looping for Practice
**Enhancement**: Select loop region on waveform

```javascript
class ABLooper {
    setLoopStart(time) { this.loopA = time; }
    setLoopEnd(time) { this.loopB = time; }
    play() {
        // Loop between A and B points
    }
}
```

### 3.4 Speed/Pitch Control for Practice
**Enhancement**: Adjust playback speed without pitch change (and vice versa)

```javascript
// Using Tone.js or Web Audio API
const pitchShift = new Tone.PitchShift(0); // semitones
const playbackRate = 1.0; // 0.5 = half speed
```

### 3.5 Track Info Panel
**Enhancement**: Show rich artist/album info while playing

```
┌─────────────────────────────────────┐
│ 📀 Estimated Prophet               │
│ Terrapin Station (1977)            │
│                                     │
│ 👤 Keyboardist: Keith Godchaux     │
│ 🎷 Era: Post-Mickey, Pre-Brent     │
│                                     │
│ 📝 Learning Tips:                  │
│ Focus on the polyrhythmic interplay │
│ between Billy and Mickey...         │
└─────────────────────────────────────┘
```

### 3.6 Mobile-Responsive Mixer
**Enhancement**: Touch-friendly controls for mobile/tablet

### 3.7 Dark/Light Theme Toggle
**Enhancement**: User preference for visual theme

### 3.8 Keyboard Shortcuts
```
SPACE - Play/Pause
[ / ] - Set loop A/B points
L - Toggle loop
M - Mute selected stem
S - Solo selected stem
+/- - Adjust playback speed
```

---

## 4. ✨ NEW FEATURES

### 4.1 🎵 Key Detection
**Purpose**: Detect musical key of the song

```python
def detect_key(audio_path: str) -> dict:
    """
    Using librosa's chroma features + Krumhansl-Kessler algorithm
    Returns: {'key': 'G', 'mode': 'major', 'confidence': 0.85}
    """
    import librosa
    y, sr = librosa.load(audio_path)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Apply Krumhansl-Kessler key profiles
```

### 4.2 🎼 Chord Detection
**Purpose**: Detect chord progression over time

```python
def detect_chords(audio_path: str) -> list:
    """
    Returns timestamped chord changes
    [{'time': 0.0, 'chord': 'G'}, {'time': 2.5, 'chord': 'C'}, ...]
    """
```

**UI**: Overlay chord symbols on waveform

### 4.3 ⏱️ Tempo/Beat Grid
**Purpose**: Detect tempo and align to beat grid

```python
def detect_tempo_and_beats(audio_path: str) -> dict:
    """
    Returns: {
        'tempo': 125.5,
        'time_signature': '4/4',
        'beat_times': [0.0, 0.48, 0.96, ...],
        'downbeat_times': [0.0, 1.92, 3.84, ...]
    }
    """
```

**UI**: Show beat markers on waveform, snap loop points to beats

### 4.4 🔄 Section Detection
**Purpose**: Identify song sections (verse, chorus, bridge, jam)

```python
def detect_sections(audio_path: str) -> list:
    """
    Using structural segmentation
    Returns: [
        {'start': 0, 'end': 30, 'label': 'Intro'},
        {'start': 30, 'end': 90, 'label': 'Verse 1'},
        {'start': 90, 'end': 150, 'label': 'Chorus'},
        {'start': 150, 'end': 300, 'label': 'Jam'},
    ]
    """
```

**Libraries**: msaf, librosa.segment

### 4.5 🎸 Guitar Pro Export
**Purpose**: Export to .gp5/.gpx format for Guitar Pro

```python
def export_to_guitar_pro(midi_path: str, output_path: str,
                          tuning: str = 'standard', capo: int = 0):
    """
    Convert MIDI to Guitar Pro format with:
    - Proper tablature
    - Fingering suggestions
    - Tempo/time signature
    """
```

**Library**: PyGuitarPro

### 4.6 🥁 Click Track Generation
**Purpose**: Generate metronome click track synced to song

```python
def generate_click_track(tempo: float, beat_times: list,
                         output_path: str, time_sig: str = '4/4'):
    """
    Generate click track audio file
    - Accent on downbeats
    - Optional count-in
    """
```

### 4.7 📱 PWA Support
**Purpose**: Install as mobile app

```javascript
// service-worker.js for offline support
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open('stemscribe-v1').then((cache) => {
            return cache.addAll(['/index.html', '/app.js', '/styles.css']);
        })
    );
});
```

### 4.8 🔗 Share/Collaborate
**Purpose**: Share processed stems with bandmates

- Generate shareable link for processed job
- Optional password protection
- Expiring links (24h/7d/30d)

### 4.9 📊 Practice Statistics
**Purpose**: Track practice sessions and progress

```python
practice_stats = {
    'total_time': '45:32',
    'songs_practiced': 12,
    'most_practiced_stems': ['guitar', 'bass'],
    'favorite_artists': ['Grateful Dead', 'Phish'],
    'streak_days': 7,
}
```

### 4.10 🎤 Backing Track Mode
**Purpose**: Mute one stem for play-along practice

- One-click "Practice Guitar" mode (mutes guitar, plays rest)
- Adjustable backing track volume
- Loop section for drilling

### 4.11 📝 Notation Display Improvements
**Purpose**: Better sheet music / tab display

- Render MusicXML in browser using OpenSheetMusicDisplay
- Tab vs Standard notation toggle
- Print-friendly view
- Scroll sync with playback

### 4.12 🤖 AI-Powered Features

#### 4.12.1 Similar Song Recommendations
```python
def find_similar_songs(audio_features: dict) -> list:
    """
    Recommend similar songs based on:
    - Tempo, key, genre
    - Harmonic complexity
    - Jam length / structure
    """
```

#### 4.12.2 Learning Path Generator
```python
def generate_learning_path(song: str, skill_level: str) -> list:
    """
    AI suggests practice approach:
    1. Learn the chord progression first
    2. Practice the verse riff at 75% speed
    3. Focus on the Jerry/Bob interplay in the jam
    """
```

---

## 5. 🛠️ TECHNICAL IMPROVEMENTS

### 5.1 Async Processing with Celery
**Purpose**: Better job queue management

```python
from celery import Celery
app = Celery('stemscribe', broker='redis://localhost:6379')

@app.task
def process_audio_async(job_id, audio_path, options):
    # Background processing
```

### 5.2 WebSocket Progress Updates
**Purpose**: Real-time progress without polling

```python
from flask_socketio import SocketIO, emit
socketio = SocketIO(app)

def update_progress(job_id, progress, stage):
    socketio.emit('progress', {
        'job_id': job_id,
        'progress': progress,
        'stage': stage
    })
```

### 5.3 Caching Layer (Redis)
**Purpose**: Cache MusicBrainz/Wikipedia results

```python
import redis
cache = redis.Redis()

def fetch_with_cache(key, fetch_fn, ttl=86400):
    cached = cache.get(key)
    if cached:
        return json.loads(cached)
    result = fetch_fn()
    cache.setex(key, ttl, json.dumps(result))
    return result
```

### 5.4 S3/R2 Storage for Stems
**Purpose**: Store processed stems in cloud storage

```python
import boto3
s3 = boto3.client('s3')

def upload_to_s3(local_path: str, job_id: str):
    s3.upload_file(local_path, 'stemscribe-stems', f'{job_id}/{filename}')
```

### 5.5 Docker Deployment
**Purpose**: Easy deployment with Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: ./backend
    ports: ["5000:5000"]
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
  redis:
    image: redis:alpine
  celery:
    build: ./backend
    command: celery -A app.celery worker
```

---

## 📋 PRIORITY MATRIX

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Key Detection | High | Low | 🔴 P1 |
| Chord Detection | High | Medium | 🔴 P1 |
| A/B Looping | High | Low | 🔴 P1 |
| Speed Control | High | Low | 🔴 P1 |
| Waveform Display | High | Medium | 🔴 P1 |
| More Bands | Medium | Low | 🟡 P2 |
| Beat Grid | Medium | Medium | 🟡 P2 |
| Section Detection | Medium | High | 🟡 P2 |
| Setlist.fm Integration | Medium | Medium | 🟡 P2 |
| Guitar Pro Export | Medium | High | 🟢 P3 |
| Practice Stats | Low | Medium | 🟢 P3 |
| PWA Support | Low | Medium | 🟢 P3 |
| Docker Deployment | High | Medium | 🟡 P2 |

---

## 🚀 SUGGESTED IMPLEMENTATION ORDER

### Phase 1: Quick Wins (1-2 days each)
1. ✅ Key Detection
2. ✅ Speed/Pitch Control
3. ✅ A/B Looping
4. ✅ More Bands in knowledge base

### Phase 2: Core Improvements (3-5 days each)
5. Waveform Display with wavesurfer.js
6. Chord Detection
7. Beat Grid / Tempo Detection
8. Player Names on Mixer

### Phase 3: Advanced Features (1-2 weeks each)
9. Section Detection
10. Guitar Pro Export
11. Setlist.fm + Discogs Integration
12. Practice Statistics

### Phase 4: Infrastructure (1-2 weeks)
13. Docker Deployment
14. WebSocket Progress
15. Cloud Storage

---

*Generated by Claude - Ready to discuss priorities and start implementing!*
