"""Parse chord charts to extract chord sequences."""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChartChord:
    """A chord from a chord chart."""
    chord: str
    section: str  # e.g., "Intro", "Verse 1", "Chorus"


def parse_chord_chart(chart_text: str) -> List[str]:
    """
    Parse a chord chart and extract the sequence of chords.

    Args:
        chart_text: The chord chart text (like from Ultimate Guitar)

    Returns:
        List of chord names in order of appearance
    """
    chords = []
    current_section = ""

    # Pattern to match section headers like [Intro], [Verse 1], etc.
    section_pattern = re.compile(r'\[([^\]]+)\]')

    # Pattern to match bar lines like | C | F | G | G |
    bar_pattern = re.compile(r'\|\s*([A-G][#b]?m?\d*(?:sus\d?|add\d|maj\d|dim|aug)?)\s*')

    # Pattern to match chord annotations above lyrics
    # Chords are typically uppercase letters at specific positions
    chord_pattern = re.compile(r'\b([A-G][#b]?m?\d*(?:sus\d?|add\d|maj\d|dim|aug)?)\b')

    lines = chart_text.split('\n')

    for i, line in enumerate(lines):
        # Check for section header
        section_match = section_pattern.search(line)
        if section_match:
            current_section = section_match.group(1)
            continue

        # Skip metadata lines
        if any(x in line.lower() for x in ['tuning:', 'key:', 'capo:', 'bpm:']):
            continue

        # Check for bar notation | C | F | G |
        bar_matches = bar_pattern.findall(line)
        if bar_matches:
            chords.extend(bar_matches)
            continue

        # Check if this looks like a chord line (mostly chords and spaces)
        # A chord line typically has chords spaced out with lots of whitespace
        stripped = line.strip()
        if stripped and not any(c.islower() for c in stripped.replace(' ', '')):
            # This might be a chord line - extract chords
            chord_matches = chord_pattern.findall(line)
            if chord_matches:
                chords.extend(chord_matches)
                continue

        # For lines with lyrics, check if there are chord annotations
        # These are usually on the line above, but sometimes inline
        if '(' in line:
            # Handle chords in parentheses like (C)
            paren_chords = re.findall(r'\(([A-G][#b]?m?\d*(?:sus\d?|add\d|maj\d|dim|aug)?)\)', line)
            chords.extend(paren_chords)

    return chords


def get_unique_chart_chords(chart_text: str) -> set:
    """Get the set of unique chords in a chart."""
    return set(parse_chord_chart(chart_text))


def create_chord_mapping(chart_chords: List[str]) -> dict:
    """
    Create a mapping to correct AI-detected chords based on chart.

    This maps chord roots to the most likely chord quality from the chart.
    For example, if chart has 'C' but AI detects 'Cm', we correct to 'C'.

    Args:
        chart_chords: List of chords from the chart

    Returns:
        Dictionary mapping chord roots to full chord names
    """
    # Count chord occurrences by root
    root_counts = {}
    for chord in chart_chords:
        # Extract root (first letter + optional sharp/flat)
        match = re.match(r'([A-G][#b]?)', chord)
        if match:
            root = match.group(1)
            if root not in root_counts:
                root_counts[root] = {}
            if chord not in root_counts[root]:
                root_counts[root][chord] = 0
            root_counts[root][chord] += 1

    # For each root, pick the most common chord
    mapping = {}
    for root, chord_counts in root_counts.items():
        most_common = max(chord_counts.keys(), key=lambda c: chord_counts[c])
        mapping[root] = most_common

    return mapping


def correct_chord(detected_chord: str, chord_mapping: dict) -> str:
    """
    Correct an AI-detected chord using the chart mapping.

    Args:
        detected_chord: The chord detected by AI
        chord_mapping: Mapping from chord roots to correct chord names

    Returns:
        Corrected chord name
    """
    if detected_chord in ('N', 'N.C.', ''):
        return detected_chord

    # Extract root
    match = re.match(r'([A-G][#b]?)', detected_chord)
    if match:
        root = match.group(1)
        if root in chord_mapping:
            return chord_mapping[root]

    return detected_chord


def load_chord_chart(file_path: str) -> str:
    """Load a chord chart from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
