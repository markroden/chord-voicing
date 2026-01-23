"""Convert chord symbols to spoken text."""

import re
from typing import Optional

# Root note pronunciation (lowercase to avoid TTS saying "capital")
NOTE_NAMES = {
    'C': 'c',
    'D': 'd',
    'E': 'e',
    'F': 'f',
    'G': 'g',
    'A': 'eh',  # Phonetic for note A
    'B': 'b',
}

# Accidental pronunciation
ACCIDENTALS = {
    '#': 'sharp',
    'b': 'flat',
    '##': 'double sharp',
    'bb': 'double flat',
}

# Chord quality to spoken text mapping
QUALITY_MAP = {
    # Major variants - empty string means just say the note name
    '': '',
    'maj': '',
    'M': '',

    # Minor variants
    'm': 'minor',
    'min': 'minor',
    '-': 'minor',

    # Seventh chords
    '7': 'seven',
    'maj7': 'major seven',
    'M7': 'major seven',
    'Maj7': 'major seven',
    'm7': 'minor seven',
    'min7': 'minor seven',
    '-7': 'minor seven',

    # Diminished
    'dim': 'diminished',
    'dim7': 'diminished seven',
    'o': 'diminished',
    'o7': 'diminished seven',

    # Augmented
    'aug': 'augmented',
    '+': 'augmented',
    'aug7': 'augmented seven',
    '+7': 'augmented seven',

    # Suspended
    'sus': 'suspended',
    'sus4': 'suspended four',
    'sus2': 'suspended two',
    '7sus4': 'seven suspended four',
    '7sus2': 'seven suspended two',

    # Other extensions
    '9': 'nine',
    'maj9': 'major nine',
    'm9': 'minor nine',
    '11': 'eleven',
    '13': 'thirteen',

    # Add chords
    'add9': 'add nine',
    'add2': 'add two',

    # Power chord
    '5': 'five',
}

# Pattern to parse chord symbols
# Matches: Root (A-G) + optional accidental (#, b) + optional quality
CHORD_PATTERN = re.compile(
    r'^([A-G])(#{1,2}|b{1,2})?(.*)$'
)


def format_chord(chord_symbol: str) -> Optional[str]:
    """
    Convert a chord symbol to spoken text.

    Args:
        chord_symbol: Chord symbol like "Am7", "F#", "Bbmaj7"

    Returns:
        Spoken text like "A minor seven", "F sharp major", "B flat major seven"
        Returns None if the chord cannot be parsed or is "N" (no chord)
    """
    if not chord_symbol or chord_symbol.strip() in ('N', 'N.C.', 'NC', ''):
        return None

    chord_symbol = chord_symbol.strip()

    match = CHORD_PATTERN.match(chord_symbol)
    if not match:
        return None

    root, accidental, quality = match.groups()

    # Build the spoken text
    parts = [NOTE_NAMES.get(root, root)]

    # Add accidental if present
    if accidental:
        parts.append(ACCIDENTALS.get(accidental, ''))

    # Add quality
    quality = quality or ''  # Default to empty string (major)

    # Try to match the quality directly first
    if quality in QUALITY_MAP:
        spoken_quality = QUALITY_MAP[quality]
    else:
        # Try to find a partial match for complex chords
        spoken_quality = _parse_complex_quality(quality)

    if spoken_quality:
        parts.append(spoken_quality)
    # If no quality (major chord), just use the note name

    return ' '.join(parts).strip()


def _parse_complex_quality(quality: str) -> Optional[str]:
    """
    Parse complex chord qualities that might not be in the direct mapping.

    Args:
        quality: The quality portion of the chord (e.g., "m7b5", "7#9")

    Returns:
        Spoken text for the quality, or None if unparseable
    """
    if not quality:
        return None

    # Common complex chords
    complex_map = {
        'm7b5': 'minor seven flat five',
        'min7b5': 'minor seven flat five',
        '-7b5': 'minor seven flat five',
        'ø': 'half diminished',
        'ø7': 'half diminished seven',
        '7b5': 'seven flat five',
        '7#5': 'seven sharp five',
        '7b9': 'seven flat nine',
        '7#9': 'seven sharp nine',
        'm6': 'minor six',
        '6': 'six',
        'maj6': 'major six',
        '6/9': 'six nine',
        'mmaj7': 'minor major seven',
        'm/maj7': 'minor major seven',
    }

    if quality in complex_map:
        return complex_map[quality]

    return None


def get_unique_chords(chord_names: list[str]) -> set[str]:
    """
    Get the set of unique chord names that need TTS generation.

    Args:
        chord_names: List of chord symbols

    Returns:
        Set of unique spoken chord texts (excluding None values)
    """
    unique = set()
    for chord in chord_names:
        spoken = format_chord(chord)
        if spoken:
            unique.add(spoken)
    return unique
