"""Load chord timing from MIDI files."""

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pretty_midi


def load_chords_from_midi(midi_path: str | Path) -> List[Tuple[float, str]]:
    """
    Load chord names and timing from a MIDI file.

    Looks for chords in:
    1. Text events (lyrics, text, markers) containing chord names
    2. If no text chords found, tries to infer from note events

    Args:
        midi_path: Path to the MIDI file

    Returns:
        List of (time_in_seconds, chord_name) tuples sorted by time
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # First try to find chords in text/lyric/marker events
    chords = _extract_text_chords(midi)

    if chords:
        return sorted(chords, key=lambda x: x[0])

    # Fall back to inferring chords from notes
    chords = _infer_chords_from_notes(midi)
    return sorted(chords, key=lambda x: x[0])


def load_midi_info(midi_path: str | Path) -> Dict[str, Any]:
    """
    Load comprehensive info from a MIDI file.

    Args:
        midi_path: Path to the MIDI file

    Returns:
        Dictionary with keys:
        - chords: List of (time, chord_name) tuples
        - bpm: Tempo in BPM (or None if not found)
        - duration: Duration in seconds
        - time_signature: Tuple of (numerator, denominator) or None
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # Get chords
    chords = _extract_text_chords(midi)
    if not chords:
        chords = _infer_chords_from_notes(midi)
    chords = sorted(chords, key=lambda x: x[0])

    # Get BPM
    tempo_times, tempos = midi.get_tempo_changes()
    bpm = tempos[0] if len(tempos) > 0 else None

    # Get duration
    duration = midi.get_end_time()

    # Get time signature
    time_sig = None
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[-1]
        time_sig = (ts.numerator, ts.denominator)

    return {
        'chords': chords,
        'bpm': bpm,
        'duration': duration,
        'time_signature': time_sig,
    }


def _extract_text_chords(midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, str]]:
    """Extract chords from MIDI text events (lyrics, text, markers)."""
    chords = []

    # Common chord pattern: root note + optional accidental + optional quality
    # Matches: C, Am, F#m7, Bb, Dmaj7, G7, etc.
    chord_pattern = re.compile(
        r'^([A-G][#b]?)(m|min|maj|dim|aug|sus|add|M|°|ø|Δ)?'
        r'(maj|min|dim|aug)?'
        r'([0-9])?'
        r'(sus[24]|add[29]|b5|#5|b9|#9|#11|b13)?'
        r'(/[A-G][#b]?)?$'
    )

    # Check lyrics
    for lyric in midi.lyrics:
        text = lyric.text.strip()
        if chord_pattern.match(text) or _is_chord_name(text):
            chords.append((lyric.time, text))

    # Check text events
    for text_event in midi.text_events:
        text = text_event.text.strip()
        if chord_pattern.match(text) or _is_chord_name(text):
            chords.append((text_event.time, text))

    return chords


def _is_chord_name(text: str) -> bool:
    """Check if text looks like a chord name."""
    # Simple heuristic: starts with A-G, followed by common chord suffixes
    if not text or len(text) > 15:
        return False

    text = text.strip()
    if not text:
        return False

    # Must start with a note name
    if text[0] not in 'ABCDEFG':
        return False

    # Check for common chord patterns
    common_patterns = [
        r'^[A-G][#b]?$',  # Just root: C, F#, Bb
        r'^[A-G][#b]?m$',  # Minor: Am, F#m
        r'^[A-G][#b]?7$',  # Dominant 7: G7, B7
        r'^[A-G][#b]?m7$',  # Minor 7: Am7, Em7
        r'^[A-G][#b]?maj7$',  # Major 7: Cmaj7
        r'^[A-G][#b]?dim$',  # Diminished
        r'^[A-G][#b]?aug$',  # Augmented
        r'^[A-G][#b]?sus[24]$',  # Suspended
        r'^[A-G][#b]?add[29]$',  # Add chords
        r'^[A-G][#b]?m7b5$',  # Half-diminished
        r'^[A-G][#b]?[0-9]+$',  # Extended: C9, G13
        r'^[A-G][#b]?m[0-9]+$',  # Minor extended: Am9
        r'^[A-G][#b]?/[A-G][#b]?$',  # Slash chords: C/E
    ]

    for pattern in common_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True

    return False


def _infer_chords_from_notes(midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, str]]:
    """
    Infer chords from simultaneous note events.

    Groups notes that start within a small time window and tries to
    identify the chord from the pitch classes. Detects slash chords
    when bass note differs from chord root.
    """
    # Collect all notes with their start times
    all_notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            all_notes.append((note.start, note.pitch))

    if not all_notes:
        return []

    all_notes.sort(key=lambda x: x[0])

    # Group notes by time (within 50ms window)
    TIME_WINDOW = 0.05
    chords = []
    current_group = [all_notes[0]]

    for i in range(1, len(all_notes)):
        time, pitch = all_notes[i]
        if time - current_group[0][0] <= TIME_WINDOW:
            current_group.append((time, pitch))
        else:
            # Process current group
            if len(current_group) >= 3:  # Need at least 3 notes for a chord
                chord_time = current_group[0][0]
                pitches = [p for _, p in current_group]
                chord_name = _identify_chord_with_bass(pitches)
                if chord_name:
                    # Avoid duplicate chords at same time
                    if not chords or abs(chords[-1][0] - chord_time) > 0.1:
                        chords.append((chord_time, chord_name))
            current_group = [(time, pitch)]

    # Process last group
    if len(current_group) >= 3:
        chord_time = current_group[0][0]
        pitches = [p for _, p in current_group]
        chord_name = _identify_chord_with_bass(pitches)
        if chord_name:
            if not chords or abs(chords[-1][0] - chord_time) > 0.1:
                chords.append((chord_time, chord_name))

    return chords


def _identify_chord_with_bass(pitches: List[int]) -> Optional[str]:
    """
    Identify a chord from pitches, detecting slash chords when
    bass note differs from chord root.
    """
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    sorted_pitches = sorted(pitches)
    lowest_pitch = sorted_pitches[0]

    # Check if lowest note is significantly lower (likely a bass note)
    # If gap >= 5 semitones, treat as separate bass
    if len(sorted_pitches) >= 4 and sorted_pitches[1] - lowest_pitch >= 5:
        bass_note = NOTE_NAMES[lowest_pitch % 12]
        chord_pitches = sorted_pitches[1:]  # Chord without bass
        chord_name = _identify_chord(chord_pitches)

        if chord_name:
            # Extract chord root
            chord_root = chord_name[0]
            if len(chord_name) > 1 and chord_name[1] in '#b':
                chord_root = chord_name[:2]

            # Only add slash if bass differs from chord root
            if bass_note != chord_root:
                return f"{chord_name}/{bass_note}"
            return chord_name

    # No separate bass, use normal detection
    return _identify_chord(pitches)


def _identify_chord(pitches: List[int]) -> Optional[str]:
    """
    Identify a chord from a list of MIDI pitch values.

    Returns chord name or None if unrecognized.
    """
    # Convert to pitch classes (0-11)
    pitch_classes = sorted(set(p % 12 for p in pitches))

    if len(pitch_classes) < 3:
        return None

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Common chord templates (intervals from root)
    CHORD_TYPES = {
        (0, 4, 7): '',          # Major
        (0, 3, 7): 'm',         # Minor
        (0, 4, 7, 10): '7',     # Dominant 7
        (0, 3, 7, 10): 'm7',    # Minor 7
        (0, 4, 7, 11): 'maj7',  # Major 7
        (0, 3, 6): 'dim',       # Diminished
        (0, 4, 8): 'aug',       # Augmented
        (0, 3, 6, 9): 'dim7',   # Diminished 7
        (0, 3, 6, 10): 'm7b5',  # Half-diminished
        (0, 5, 7): 'sus4',      # Suspended 4
        (0, 2, 7): 'sus2',      # Suspended 2
    }

    # Try each pitch class as potential root
    for root in pitch_classes:
        # Normalize intervals relative to root
        intervals = tuple(sorted((p - root) % 12 for p in pitch_classes))

        # Check if this matches a known chord type
        for template, suffix in CHORD_TYPES.items():
            if _intervals_match(intervals, template):
                return NOTE_NAMES[root] + suffix

    # If no match, return the lowest note as root with "?"
    root = min(pitches) % 12
    return NOTE_NAMES[root] + "?"


def _intervals_match(intervals: Tuple[int, ...], template: Tuple[int, ...]) -> bool:
    """Check if intervals match a chord template (allowing extra notes)."""
    template_set = set(template)
    interval_set = set(intervals)
    # Template notes must be present
    return template_set.issubset(interval_set)
