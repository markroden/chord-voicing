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

    def __init__(self, hop_length: int = 2048, frame_length: float = 1.0,
                 min_chord_duration: float = 0.5):
        """
        Initialize the librosa-based chord detector.

        Args:
            hop_length: Hop length for chroma calculation
            frame_length: Length of each analysis frame in seconds (longer = more stable)
            min_chord_duration: Minimum chord duration in seconds (filter noise)
        """
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.min_chord_duration = min_chord_duration

    def detect(self, audio_path: str | Path) -> List[ChordEvent]:
        """Detect chords using librosa chroma features."""
        import librosa

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=22050)
        duration = len(y) / sr

        # Calculate chroma features with harmonic component only
        # This helps filter out percussion/noise
        y_harmonic = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=self.hop_length)

        # Apply median filtering to smooth chroma features
        from scipy.ndimage import median_filter
        chroma = median_filter(chroma, size=(1, 9))

        # Calculate frame times
        frame_times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sr,
            hop_length=self.hop_length
        )

        # Group frames into larger segments for more stable detection
        segment_samples = int(self.frame_length * sr / self.hop_length)

        raw_events = []
        prev_chord = None

        for i in range(0, chroma.shape[1], segment_samples):
            end_idx = min(i + segment_samples, chroma.shape[1])
            segment_chroma = np.mean(chroma[:, i:end_idx], axis=1)

            chord = _match_chord(segment_chroma)
            current_time = frame_times[i] if i < len(frame_times) else duration

            if chord != prev_chord and prev_chord is not None:
                # End the previous chord
                if raw_events:
                    raw_events[-1] = ChordEvent(
                        start_time=raw_events[-1].start_time,
                        end_time=current_time,
                        chord_name=raw_events[-1].chord_name
                    )
                # Start new chord
                if chord != 'N':
                    raw_events.append(ChordEvent(
                        start_time=current_time,
                        end_time=duration,
                        chord_name=chord
                    ))
            elif prev_chord is None and chord != 'N':
                raw_events.append(ChordEvent(
                    start_time=current_time,
                    end_time=duration,
                    chord_name=chord
                ))

            prev_chord = chord

        # Filter out very short chords (likely noise)
        events = []
        for event in raw_events:
            if event.duration >= self.min_chord_duration:
                events.append(event)
            elif events:
                # Extend the previous chord instead
                events[-1] = ChordEvent(
                    start_time=events[-1].start_time,
                    end_time=event.end_time,
                    chord_name=events[-1].chord_name
                )

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


class ChordDetectorEssentia:
    """Chord detector using Essentia (works on M1 Mac)."""

    def __init__(self, frame_size: int = 8192, hop_size: int = 4096,
                 min_chord_duration: float = 0.3):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.min_chord_duration = min_chord_duration

    def detect(self, audio_path: str | Path) -> List[ChordEvent]:
        """Detect chords using Essentia's HPCP + ChordsDetection."""
        import essentia.standard as es

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        sr = 44100
        audio = es.MonoLoader(filename=str(audio_path), sampleRate=sr)()
        duration = len(audio) / sr

        # Setup algorithms
        w = es.Windowing(type='blackmanharris62')
        spectrum = es.Spectrum()
        peaks = es.SpectralPeaks(
            orderBy='magnitude',
            magnitudeThreshold=0.0001,
            minFrequency=40,
            maxFrequency=5000,
            maxPeaks=100
        )
        hpcp = es.HPCP(size=36, harmonics=8, windowSize=1.0)
        chords_algo = es.ChordsDetection(
            hopSize=self.hop_size,
            sampleRate=sr,
            windowSize=2
        )

        # Compute HPCP frames
        hpcp_frames = []
        for frame in es.FrameGenerator(audio, frameSize=self.frame_size, hopSize=self.hop_size):
            spec = spectrum(w(frame))
            freqs, mags = peaks(spec)
            h = hpcp(freqs, mags)
            hpcp_frames.append(h)

        hpcp_array = np.array(hpcp_frames)
        chord_list, strength_list = chords_algo(hpcp_array)

        # Group consecutive same chords
        events = []
        prev_chord = None
        start_time = 0.0

        for i, chord in enumerate(chord_list):
            time = i * self.hop_size / sr
            if chord != prev_chord:
                if prev_chord is not None:
                    chord_duration = time - start_time
                    if chord_duration >= self.min_chord_duration:
                        events.append(ChordEvent(
                            start_time=start_time,
                            end_time=time,
                            chord_name=prev_chord
                        ))
                start_time = time
                prev_chord = chord

        # Add final chord
        if prev_chord is not None:
            events.append(ChordEvent(
                start_time=start_time,
                end_time=duration,
                chord_name=prev_chord
            ))

        return events

    def detect_with_duration(self, audio_path: str | Path, audio_duration: float) -> List[ChordEvent]:
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

    def __init__(self, prefer_librosa: bool = False, frame_length: float = 1.0,
                 min_chord_duration: float = 0.5, use_essentia: bool = True):
        """
        Initialize the chord detector.

        Args:
            prefer_librosa: If True, always use librosa instead of trying other detectors
            frame_length: Analysis frame length in seconds (for librosa fallback)
            min_chord_duration: Minimum chord duration in seconds
            use_essentia: If True, try Essentia before librosa (recommended for M1 Mac)
        """
        self._chordino_failed = False
        self._essentia_failed = False
        self._prefer_librosa = prefer_librosa
        self._use_essentia = use_essentia
        self._frame_length = frame_length
        self._min_chord_duration = min_chord_duration

    def _try_essentia(self, audio_path: str | Path) -> Optional[List[ChordEvent]]:
        """Try to detect chords using Essentia."""
        if self._essentia_failed or self._prefer_librosa:
            return None

        try:
            detector = ChordDetectorEssentia(min_chord_duration=self._min_chord_duration)
            return detector.detect(audio_path)
        except Exception as e:
            self._essentia_failed = True
            warnings.warn(f"Essentia unavailable ({e}), trying other methods.")
            return None

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
        # Try Essentia first (best for M1 Mac)
        if self._use_essentia:
            result = self._try_essentia(audio_path)
            if result is not None:
                return result

        # Try Chordino (requires VAMP plugins)
        result = self._try_chordino(audio_path)
        if result is not None:
            return result

        # Fall back to librosa
        detector = ChordDetectorLibrosa(
            frame_length=self._frame_length,
            min_chord_duration=self._min_chord_duration
        )
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
