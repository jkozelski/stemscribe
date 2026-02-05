# StemScribe Session Notes - February 5, 2026

## ✅ Completed Today

### 1. **Songsterr/Ultimate Guitar Integration**
- Added `/api/find-tabs/<job_id>` endpoint
- Links appear in header of practice mode
- Opens in new tabs (iframe blocked by both sites)

### 2. **Stereo Splitting UI**
- Added ⇆ button on each stem channel
- Analyzes stereo field before splitting
- Creates left/right/center components

### 3. **Lead/Backing Vocal Split** 🎤
- Installed `audio-separator` package with BS-Roformer models
- Created `enhanced_separator.py` module
- Added 🎤 button on Vocals channel → AI splits lead vs backing
- Uses two-pass UVR karaoke method

### 4. **Guitar Lead/Rhythm Training Setup** 🎸
- Cloned ZFTurbo's Music-Source-Separation-Training framework
- Created custom MelBand-Roformer config
- Built dataset preparation script with 20 classic rock songs
- Ready to run on your Mac: `~/stemscribe/train_guitar_model/RUN_ME.sh`

---

## 🧪 To Test Tomorrow

1. **Start server**:
   ```bash
   cd ~/stemscribe && python3 backend/app.py
   ```

2. **Test vocal split**: Load any song in practice mode, click 🎤 on vocals

3. **Test tab links**: Should see Songsterr/UG buttons in header

4. **Run guitar training** (when you have time):
   ```bash
   cd ~/stemscribe/train_guitar_model
   ./RUN_ME.sh
   ```

---

## 📁 New Files Created

```
stemscribe/
├── backend/
│   ├── enhanced_separator.py    # New AI separator module
│   ├── songsterr.py             # Songsterr API client
│   └── app.py                   # Updated with new endpoints
├── frontend/
│   └── practice.html            # Updated with new buttons/links
├── train_guitar_model/
│   ├── RUN_ME.sh               # One-click training script
│   ├── prepare_dataset.py      # Downloads & preps 20 songs
│   └── config.yaml             # MelBand-Roformer config
└── AI_MUSIC_RESEARCH_2026.md   # Full research document
```

---

## 📊 Research Highlights

- **BS-Roformer** beats Demucs for vocals (SDR 12.9 vs ~10)
- **Lead/rhythm guitar** separation possible by training custom model
- **Moises.ai** is only commercial solution for lead/rhythm guitar
- **Training our own** is feasible with stereo-panned songs as ground truth

---

*Sleep well! 🌙*
