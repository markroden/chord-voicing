"""Chord detection using chord-extractor library or librosa fallback."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np


@dataclass
class ChordEvent:
    """Represents a detected chord at a specific time."""
    start_time: float  # seconds
    end_time: float    # seconds
    chord_name: str    # e.g., "Am7", "G", "F#m"

    @property
    def duration(self) -> float:
        """Duration of the chord in seconds."""
        return self.end_time - self.start_time


# Chord templates for major and minor triads
CHORD_TEMPLATES = {
    'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    'D#': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    'F#': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    'G#': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    'A': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'A#': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    'B': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'C#m': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    'D#m': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'Fm': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'F#m': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    'Gm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    'G#m': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    'Am': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'A#m': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    'Bm': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
}


def _match_chord(chroma_vector: np.ndarray, major_bias: float = 0.15) -> str:
    """Match a chroma vector to the best chord template.

    Args:
        chroma_vector: 12-element chroma feature vector
        major_bias: Bonus added to major chord scores (minor chords must
                   beat major by this margin to be selected)
    """
    best_chord = 'N'
    best_score = -1

    # Normalize chroma vector
    norm = np.linalg.norm(chroma_vector)
    if norm > 0:
        chroma_vector = chroma_vector / norm
    else:
        return 'N'

    scores = {}
    for chord_name, template in CHORD_TEMPLATES.items():
        template_arr = np.array(template, dtype=float)
        template_arr = template_arr / np.linalg.norm(template_arr)
        score = np.dot(chroma_vector, template_arr)

        # Add bias for major chords (they're more common in pop/rock)
        if 'm' not in chord_name:
            score += major_bias

        scores[chord_name] = score
        if score > best_score:
            best_score = score
            best_chord = chord_name

    # If the match is too weak, return no chord
    if best_score < 0.5:
        return 'N'

    return best_chord


class ChordDetectorLibrosa:
    """Chord detector using librosa's chroma features (no VAMP required)."""

    def __init__(self, hop_length: int = 2048, frame_length: float = 0.5):
        """
        Initialize the librosa-based chord detector.

        Args:
            hop_length: Hop length for chroma calculation
            frame_length: Length of each analysis frame in seconds
        """
        self.hop_length = hop_length
        self.frame_length = frame_length

    def detect(self, audio_path: str | Path) -> List[ChordEvent]:
        """Detect chords using librosa chroma features."""
        import librosa

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=22050)
        duration = len(y) / sr

        # Calculate chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)

        # Calculate frame times
        frame_times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sr,
            hop_length=self.hop_length
        )

        # Group frames into larger segments for more stable detection
        segment_samples = int(self.frame_length * sr / self.hop_length)

        events = []
        prev_chord = None
        segment_start = 0.0

        for i in range(0, chroma.shape[1], segment_samples):
            end_idx = min(i + segment_samples, chroma.shape[1])
            segment_chroma = np.mean(chroma[:, i:end_idx], axis=1)

            chord = _match_chord(segment_chroma)
            current_time = frame_times[i] if i < len(frame_times) else duration

            if chord != prev_chord and prev_chord is not None:
                # End the previous chord
                if events:
                    events[-1] = ChordEvent(
                        start_time=events[-1].start_time,
                        end_time=current_time,
                        chord_name=events[-1].chord_name
                    )
                # Start new chord
                if chord != 'N':
                    events.append(ChordEvent(
                        start_time=current_time,
                        end_time=duration,
                        chord_name=chord
                    ))
            elif prev_chord is None and chord != 'N':
                events.append(ChordEvent(
                    start_time=current_time,
                    end_time=duration,
                    chord_name=chord
                ))

            prev_chord = chord

        return events

    def detect_with_duration(self, audio_path: str | Path, audio_duration: float) -> List[ChordEvent]:
        """Detect chords with accurate end time for the last chord."""
        events = self.detect(audio_path)
        if events:
            events[-1] = ChordEvent(
                start_time=events[-1].start_time,
                end_time=audio_duration,
                chord_name=events[-1].chord_name
            )
        return events


class ChordDetectorChordino:
    """Wrapper around chord-extractor/Chordino (requires VAMP plugins)."""

    def __init__(self):
        """Initialize the chord detector with Chordino."""
        from chord_extractor.extractors import Chordino
        self.extractor = Chordino()

    def detect(self, audio_path: str | Path) -> List[ChordEvent]:
        """Detect chords using Chordino VAMP plugin."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        raw_chords = self.extractor.extract(str(audio_path))
        if not raw_chords:
            return []

        events = []
        for i, (chord_name, start_time) in enumerate(raw_chords):
            if i < len(raw_chords) - 1:
                end_time = raw_chords[i + 1][1]
            else:
                end_time = start_time + 4.0

            events.append(ChordEvent(
                start_time=start_time,
                end_time=end_time,
                chord_name=chord_name
            ))

        return events

    def detect_with_duration(self, audio_path: str | Path, audio_duration: float) -> List[ChordEvent]:
        """Detect chords with accurate end time for the last chord."""
        events = self.detect(audio_path)
        if events:
            events[-1] = ChordEvent(
                start_time=events[-1].start_time,
                end_time=audio_duration,
                chord_name=events[-1].chord_name
            )
        return events


class ChordDetector:
    """
    Chord detector that tries Chordino first, falls back to librosa.

    Chordino (VAMP) provides better accuracy but requires plugin installation.
    Librosa fallback works out of the box but is less accurate.
    """

    def __init__(self, prefer_librosa: bool = False):
        """
        Initialize the chord detector.

        Args:
            prefer_librosa: If True, always use librosa instead of trying Chordino
        """
        self._chordino_failed = False
        self._prefer_librosa = prefer_librosa

    def _try_chordino(self, audio_path: str | Path) -> Optional[List[ChordEvent]]:
        """Try to detect chords using Chordino."""
        if self._chordino_failed or self._prefer_librosa:
            return None

        try:
            detector = ChordDetectorChordino()
            return detector.detect(audio_path)
        except Exception as e:
            self._chordino_failed = True
            warnings.warn(
                f"Chordino unavailable ({e}), falling back to librosa. "
                "For better accuracy, install VAMP plugins from "
                "https://code.soundsoftware.ac.uk/projects/vamp-plugin-pack"
            )
            return None

    def detect(self, audio_path: str | Path) -> List[ChordEvent]:
        """Detect chords in an audio file."""
        # Try Chordino first
        result = self._try_chordino(audio_path)
        if result is not None:
            return result

        # Fall back to librosa
        detector = ChordDetectorLibrosa()
        return detector.detect(audio_path)

    def detect_with_duration(self, audio_path: str | Path, audio_duration: float) -> List[ChordEvent]:
        """Detect chords with accurate end time for the last chord."""
        events = self.detect(audio_path)
        if events:
            events[-1] = ChordEvent(
                start_time=events[-1].start_time,
                end_time=audio_duration,
                chord_name=events[-1].chord_name
            )
        return events


def detect_chords(audio_path: str | Path) -> List[ChordEvent]:
    """Convenience function to detect chords in an audio file."""
    detector = ChordDetector()
    return detector.detect(audio_path)
