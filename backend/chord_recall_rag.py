"""
RAG-based chord chart recall for StemScriber.

Embeds all 15,400+ songs' lyrics using a sentence transformer.
At inference, Whisper lyrics are embedded and matched against the index.
Returns the correct chart for known songs with near-perfect accuracy.

Usage:
    # Build index (one-time, ~5 min on CPU):
    python chord_recall_rag.py --build

    # In the app:
    from chord_recall_rag import ChordRecallRAG
    rag = ChordRecallRAG()
    result = rag.search("another night stirring with the boys in a cloud")
    # → {"match": True, "confidence": 0.97, "title": "Thunderhead", "artist": "Kozelski", "chart": {...}}
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

# Paths
CHORD_LIBRARY = Path(__file__).parent / "chord_library"
INDEX_DIR = Path(__file__).parent / "training_data" / "chord_recall_index"
EMBEDDINGS_FILE = INDEX_DIR / "lyrics_embeddings.npy"
METADATA_FILE = INDEX_DIR / "lyrics_metadata.json"

# Minimum confidence to return a match (0-1, cosine similarity)
MIN_CONFIDENCE = 0.65


class ChordRecallRAG:
    """Search chord charts by lyrics using sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._embeddings = None
        self._metadata = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_index(self):
        """Load pre-built embeddings and metadata."""
        if self._embeddings is None:
            if not EMBEDDINGS_FILE.exists():
                raise FileNotFoundError(
                    f"RAG index not found at {EMBEDDINGS_FILE}. "
                    "Run: python chord_recall_rag.py --build"
                )
            self._embeddings = np.load(str(EMBEDDINGS_FILE))
            with open(METADATA_FILE) as f:
                self._metadata = json.load(f)

    def search(self, lyrics: str, top_k: int = 3) -> dict:
        """
        Search for a song by its lyrics.

        Args:
            lyrics: Transcribed lyrics (from Whisper or user input)
            top_k: Number of candidates to return

        Returns:
            dict with match info: {match, confidence, title, artist, chart, candidates}
        """
        self._load_index()

        if not lyrics or len(lyrics.strip()) < 10:
            return {"match": False, "reason": "lyrics too short", "candidates": []}

        # Embed the query lyrics
        query_embedding = self.model.encode([lyrics], normalize_embeddings=True)[0]

        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        similarities = self._embeddings @ query_embedding

        # Get top indices, deduplicate by chart_path (same song, different embeddings)
        top_indices = np.argsort(similarities)[::-1]

        candidates = []
        seen_paths = set()
        for idx in top_indices:
            meta = self._metadata[idx]
            if meta["chart_path"] in seen_paths:
                continue
            seen_paths.add(meta["chart_path"])
            candidates.append({
                "title": meta["title"],
                "artist": meta["artist"],
                "confidence": float(similarities[idx]),
                "chart_path": meta["chart_path"],
            })
            if len(candidates) >= top_k:
                break

        # Check if best match exceeds threshold
        best = candidates[0] if candidates else None
        if best and best["confidence"] >= MIN_CONFIDENCE:
            # Load the full chart
            chart_path = Path(__file__).parent / best["chart_path"]
            if chart_path.exists():
                with open(chart_path) as f:
                    chart = json.load(f)
            else:
                chart = None

            return {
                "match": True,
                "confidence": best["confidence"],
                "title": best["title"],
                "artist": best["artist"],
                "chart": chart,
                "candidates": candidates,
            }

        return {
            "match": False,
            "reason": f"best match {best['confidence']:.2f} below threshold {MIN_CONFIDENCE}",
            "candidates": candidates,
        }

    def search_by_title_artist(self, title: str, artist: str = "") -> dict:
        """
        Search by title + artist string (fallback when no lyrics available).
        Uses the title/artist text as the search query.
        """
        query = f"{title} {artist}".strip()
        return self.search(query)


def build_index(model_name: str = "all-MiniLM-L6-v2"):
    """
    Build the RAG index from the chord library.
    Extracts lyrics from each chart, embeds them, saves to disk.
    """
    from sentence_transformers import SentenceTransformer

    print("=" * 60)
    print("StemScriber Chord Recall RAG — Building Index")
    print("=" * 60)

    model = SentenceTransformer(model_name)
    print(f"Model: {model_name}")

    # Extract lyrics from all charts — embed each line AND full song
    print("\n[1/3] Extracting lyrics from chord library...")
    songs = []

    for artist_dir in sorted(CHORD_LIBRARY.iterdir()):
        if not artist_dir.is_dir():
            continue
        for chart_file in sorted(artist_dir.glob("*.json")):
            try:
                with open(chart_file) as f:
                    chart = json.load(f)

                title = chart.get("title", chart_file.stem)
                artist = chart.get("artist", artist_dir.name)
                chart_path = str(chart_file.relative_to(Path(__file__).parent))

                all_lyrics = []
                for section in chart.get("sections", []):
                    section_lyrics = []
                    for line in section.get("lines", []):
                        if line.get("lyrics"):
                            lyric = line["lyrics"].strip()
                            if lyric:
                                section_lyrics.append(lyric)
                                all_lyrics.append(lyric)

                    # Embed each section (2-4 lines grouped)
                    if section_lyrics:
                        songs.append({
                            "title": title,
                            "artist": artist,
                            "lyrics": " ".join(section_lyrics),
                            "chart_path": chart_path,
                        })

                # Also embed full song lyrics
                if all_lyrics:
                    songs.append({
                        "title": title,
                        "artist": artist,
                        "lyrics": " ".join(all_lyrics),
                        "chart_path": chart_path,
                    })

                    # And title + artist as a search key
                    songs.append({
                        "title": title,
                        "artist": artist,
                        "lyrics": f"{title} by {artist}",
                        "chart_path": chart_path,
                    })

            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Skipping {chart_file}: {e}")

    print(f"  Found {len(songs)} songs with lyrics")

    # Embed all lyrics
    print("\n[2/3] Embedding lyrics (this takes a few minutes)...")
    lyrics_texts = [s["lyrics"] for s in songs]

    # Batch encode for speed
    embeddings = model.encode(
        lyrics_texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,  # For cosine similarity via dot product
    )

    print(f"  Embeddings shape: {embeddings.shape}")

    # Save index
    print("\n[3/3] Saving index...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    np.save(str(EMBEDDINGS_FILE), embeddings)

    # Save metadata (without lyrics to save space)
    metadata = [
        {"title": s["title"], "artist": s["artist"], "chart_path": s["chart_path"]}
        for s in songs
    ]
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    index_size_mb = EMBEDDINGS_FILE.stat().st_size / (1024 * 1024)
    meta_size_mb = METADATA_FILE.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"DONE!")
    print(f"  Songs indexed: {len(songs)}")
    print(f"  Embeddings: {EMBEDDINGS_FILE} ({index_size_mb:.1f} MB)")
    print(f"  Metadata: {METADATA_FILE} ({meta_size_mb:.1f} MB)")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"{'=' * 60}")

    # Quick test
    print("\nQuick test — searching for Thunderhead...")
    rag = ChordRecallRAG(model_name)
    rag._embeddings = embeddings
    rag._metadata = metadata

    result = rag.search("another night stirring with the boys in a cloud")
    if result["match"]:
        print(f"  ✓ Found: {result['title']} by {result['artist']} (confidence: {result['confidence']:.3f})")
    else:
        print(f"  ✗ No match: {result.get('reason', 'unknown')}")
        if result.get("candidates"):
            for c in result["candidates"]:
                print(f"    {c['title']} by {c['artist']} ({c['confidence']:.3f})")

    return len(songs)


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        build_index()
    else:
        print("Usage: python chord_recall_rag.py --build")
        print("  Builds the lyrics embedding index from the chord library.")
