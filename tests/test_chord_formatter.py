"""Tests for chord_formatter module."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chord_voicing.chord_formatter import format_chord, get_unique_chords


class TestFormatChord:
    """Tests for the format_chord function."""

    def test_major_chords(self):
        """Test major chord formatting."""
        assert format_chord("C") == "C major"
        assert format_chord("D") == "D major"
        assert format_chord("E") == "E major"
        assert format_chord("Cmaj") == "C major"
        assert format_chord("CM") == "C major"

    def test_minor_chords(self):
        """Test minor chord formatting."""
        assert format_chord("Am") == "A minor"
        assert format_chord("Dm") == "D minor"
        assert format_chord("Amin") == "A minor"
        assert format_chord("A-") == "A minor"

    def test_seventh_chords(self):
        """Test seventh chord formatting."""
        assert format_chord("G7") == "G seven"
        assert format_chord("Cmaj7") == "C major seven"
        assert format_chord("CM7") == "C major seven"
        assert format_chord("Am7") == "A minor seven"
        assert format_chord("Amin7") == "A minor seven"
        assert format_chord("A-7") == "A minor seven"

    def test_sharp_chords(self):
        """Test chords with sharps."""
        assert format_chord("F#") == "F sharp major"
        assert format_chord("F#m") == "F sharp minor"
        assert format_chord("C#7") == "C sharp seven"
        assert format_chord("G#m7") == "G sharp minor seven"

    def test_flat_chords(self):
        """Test chords with flats."""
        assert format_chord("Bb") == "B flat major"
        assert format_chord("Ebm") == "E flat minor"
        assert format_chord("Ab7") == "A flat seven"
        assert format_chord("Dbmaj7") == "D flat major seven"

    def test_diminished_chords(self):
        """Test diminished chord formatting."""
        assert format_chord("Cdim") == "C diminished"
        assert format_chord("Bdim7") == "B diminished seven"
        assert format_chord("Co") == "C diminished"
        assert format_chord("Bo7") == "B diminished seven"

    def test_augmented_chords(self):
        """Test augmented chord formatting."""
        assert format_chord("Caug") == "C augmented"
        assert format_chord("C+") == "C augmented"

    def test_suspended_chords(self):
        """Test suspended chord formatting."""
        assert format_chord("Dsus") == "D suspended"
        assert format_chord("Dsus4") == "D suspended four"
        assert format_chord("Dsus2") == "D suspended two"
        assert format_chord("G7sus4") == "G seven suspended four"

    def test_no_chord(self):
        """Test that no-chord symbols return None."""
        assert format_chord("N") is None
        assert format_chord("N.C.") is None
        assert format_chord("NC") is None
        assert format_chord("") is None
        assert format_chord(None) is None

    def test_complex_chords(self):
        """Test complex chord types."""
        assert format_chord("Am7b5") == "A minor seven flat five"
        assert format_chord("C6") == "C six"
        assert format_chord("Cm6") == "C minor six"
        assert format_chord("Cadd9") == "C add nine"

    def test_whitespace_handling(self):
        """Test that whitespace is handled properly."""
        assert format_chord("  Am  ") == "A minor"
        assert format_chord("C\n") == "C major"


class TestGetUniqueChords:
    """Tests for the get_unique_chords function."""

    def test_basic_unique(self):
        """Test getting unique chords from a list."""
        chords = ["C", "Am", "F", "G", "C", "Am"]
        unique = get_unique_chords(chords)
        assert "C major" in unique
        assert "A minor" in unique
        assert "F major" in unique
        assert "G major" in unique
        assert len(unique) == 4

    def test_filters_no_chord(self):
        """Test that N (no chord) is filtered out."""
        chords = ["C", "N", "Am", "N", "G"]
        unique = get_unique_chords(chords)
        assert len(unique) == 3
        assert "C major" in unique
        assert "A minor" in unique
        assert "G major" in unique

    def test_empty_list(self):
        """Test with empty list."""
        assert get_unique_chords([]) == set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
