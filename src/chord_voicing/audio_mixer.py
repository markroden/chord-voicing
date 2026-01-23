"""Audio mixing and overlay functionality."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydub import AudioSegment

from .chord_detector import ChordEvent
from .chord_formatter import format_chord


class AudioMixer:
    """Mixes TTS audio clips with original audio at chord change timestamps."""

    def __init__(
        self,
        min_gap_seconds: float = 0.5,
        voice_volume_db: float = 0.0
    ):
        """
        Initialize the audio mixer.

        Args:
            min_gap_seconds: Minimum gap between chord announcements.
                            If chords change faster than this, some will be skipped.
            voice_volume_db: Volume adjustment for voice clips in dB
                            (positive = louder, negative = quieter)
        """
        self.min_gap_seconds = min_gap_seconds
        self.voice_volume_db = voice_volume_db

    def load_audio(self, audio_path: str | Path) -> AudioSegment:
        """
        Load an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            AudioSegment of the loaded audio
        """
        audio_path = Path(audio_path)
        return AudioSegment.from_file(str(audio_path))

    def create_voiced_track(
        self,
        duration_ms: int,
        chord_events: List[ChordEvent],
        tts_clips: Dict[str, AudioSegment],
        sample_rate: int = 44100,
        channels: int = 2
    ) -> Tuple[AudioSegment, List[Tuple[float, str]]]:
        """
        Create a track with only the voiced chord names.

        Args:
            duration_ms: Total duration of the track in milliseconds
            chord_events: List of chord events with timing
            tts_clips: Dictionary mapping spoken chord text to audio clips
            sample_rate: Sample rate for the silent track
            channels: Number of audio channels

        Returns:
            Tuple of (AudioSegment with voiced chords, list of (time, chord) that were voiced)
        """
        # Create silent track
        voiced_track = AudioSegment.silent(
            duration=duration_ms,
            frame_rate=sample_rate
        )

        if channels == 1:
            voiced_track = voiced_track.set_channels(1)
        else:
            voiced_track = voiced_track.set_channels(2)

        voiced_chords = []
        last_voice_end = -self.min_gap_seconds * 1000  # Start with negative to allow first chord

        for event in chord_events:
            # Convert chord symbol to spoken text
            spoken_text = format_chord(event.chord_name)
            if not spoken_text:
                continue  # Skip "N" (no chord) or unparseable chords

            # Check if we have a clip for this chord
            if spoken_text not in tts_clips:
                continue

            clip = tts_clips[spoken_text]

            # Apply volume adjustment
            if self.voice_volume_db != 0:
                clip = clip + self.voice_volume_db

            # Calculate position in milliseconds
            position_ms = int(event.start_time * 1000)

            # Check if there's enough gap from the last voice
            if position_ms < last_voice_end + (self.min_gap_seconds * 1000):
                continue  # Skip this chord, too close to the previous one

            # Ensure we don't go past the end
            if position_ms + len(clip) > duration_ms:
                # Trim the clip to fit
                clip = clip[:duration_ms - position_ms]

            # Overlay the clip
            voiced_track = voiced_track.overlay(clip, position=position_ms)
            voiced_chords.append((event.start_time, event.chord_name))
            last_voice_end = position_ms + len(clip)

        return voiced_track, voiced_chords

    def mix_tracks(
        self,
        original: AudioSegment,
        voiced: AudioSegment,
        original_volume_db: float = 0.0
    ) -> AudioSegment:
        """
        Mix the original audio with the voiced track.

        Args:
            original: Original audio track
            voiced: Voiced chord track
            original_volume_db: Volume adjustment for original track in dB

        Returns:
            Mixed AudioSegment
        """
        # Adjust original volume if needed
        if original_volume_db != 0:
            original = original + original_volume_db

        # Ensure both tracks have the same properties
        voiced = voiced.set_frame_rate(original.frame_rate)
        voiced = voiced.set_channels(original.channels)
        voiced = voiced.set_sample_width(original.sample_width)

        # Overlay voiced on original
        return original.overlay(voiced)

    def export(
        self,
        audio: AudioSegment,
        output_path: str | Path,
        format: str = "mp3",
        bitrate: str = "192k"
    ):
        """
        Export audio to a file.

        Args:
            audio: AudioSegment to export
            output_path: Path for the output file
            format: Output format (mp3, wav, etc.)
            bitrate: Bitrate for lossy formats
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio.export(
            str(output_path),
            format=format,
            bitrate=bitrate
        )


def process_audio(
    input_path: str | Path,
    chord_events: List[ChordEvent],
    tts_clips: Dict[str, AudioSegment],
    output_dir: str | Path,
    min_gap_seconds: float = 0.5,
    voice_volume_db: float = 0.0,
    original_volume_db: float = -3.0,  # Slightly reduce original for clarity
    voiced_only: bool = False
) -> Dict[str, Path]:
    """
    Process an audio file with chord voicing.

    Args:
        input_path: Path to input audio file
        chord_events: List of detected chord events
        tts_clips: Dictionary of TTS clips for chord names
        output_dir: Directory for output files
        min_gap_seconds: Minimum gap between chord announcements
        voice_volume_db: Volume adjustment for voice
        original_volume_db: Volume adjustment for original audio
        voiced_only: If True, only output the voiced track

    Returns:
        Dictionary with paths to output files
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mixer = AudioMixer(
        min_gap_seconds=min_gap_seconds,
        voice_volume_db=voice_volume_db
    )

    # Load original audio
    original = mixer.load_audio(input_path)
    duration_ms = len(original)

    # Create voiced track
    voiced_track, voiced_chords = mixer.create_voiced_track(
        duration_ms=duration_ms,
        chord_events=chord_events,
        tts_clips=tts_clips,
        sample_rate=original.frame_rate,
        channels=original.channels
    )

    outputs = {}

    # Export voiced-only track
    stem = input_path.stem
    voiced_path = output_dir / f"{stem}_voiced.mp3"
    mixer.export(voiced_track, voiced_path)
    outputs['voiced'] = voiced_path

    # Export mixed track (unless voiced_only)
    if not voiced_only:
        mixed = mixer.mix_tracks(original, voiced_track, original_volume_db)
        mixed_path = output_dir / f"{stem}_mixed.mp3"
        mixer.export(mixed, mixed_path)
        outputs['mixed'] = mixed_path

    return outputs
