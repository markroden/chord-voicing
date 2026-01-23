"""Main CLI entry point for chord voicing system."""

import argparse
import sys
from pathlib import Path

from .chord_detector import ChordDetector
from .chord_formatter import format_chord, get_unique_chords
from .tts_generator import TTSGenerator
from .audio_mixer import process_audio, AudioMixer


def main():
    """Main entry point for the chord voicing CLI."""
    parser = argparse.ArgumentParser(
        description="Detect chords in audio and generate voiced chord names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m chord_voicing song.mp3
  python -m chord_voicing song.mp3 --output-dir ./my_output
  python -m chord_voicing song.mp3 --voiced-only --rate 200
  python -m chord_voicing song.mp3 --min-gap 0.75 --voice-volume 3
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input audio file (MP3, WAV, etc.)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Directory for output files (default: output)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for TTS cache (default: cache)"
    )

    parser.add_argument(
        "--voiced-only",
        action="store_true",
        help="Only output the voiced track, not the mixed version"
    )

    parser.add_argument(
        "--rate", "-r",
        type=int,
        default=175,
        help="TTS speech rate in words per minute (default: 175)"
    )

    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.5,
        help="Minimum gap between chord announcements in seconds (default: 0.5)"
    )

    parser.add_argument(
        "--voice-volume",
        type=float,
        default=0.0,
        help="Voice volume adjustment in dB (default: 0.0)"
    )

    parser.add_argument(
        "--original-volume",
        type=float,
        default=-3.0,
        help="Original audio volume adjustment in dB (default: -3.0)"
    )

    parser.add_argument(
        "--list-chords",
        action="store_true",
        help="Just list detected chords without generating audio"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Processing: {input_path}")

    # Step 1: Detect chords
    if args.verbose:
        print("Detecting chords...")

    try:
        detector = ChordDetector()

        # Get audio duration for accurate last chord end time
        mixer = AudioMixer()
        original_audio = mixer.load_audio(input_path)
        audio_duration = len(original_audio) / 1000.0  # Convert ms to seconds

        chord_events = detector.detect_with_duration(input_path, audio_duration)
    except Exception as e:
        print(f"Error detecting chords: {e}", file=sys.stderr)
        sys.exit(1)

    if not chord_events:
        print("No chords detected in the audio file.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Detected {len(chord_events)} chord changes")

    # If just listing chords, print and exit
    if args.list_chords:
        print("\nDetected chords:")
        print("-" * 50)
        for event in chord_events:
            spoken = format_chord(event.chord_name) or "(no chord)"
            print(f"{event.start_time:6.2f}s - {event.end_time:6.2f}s: {event.chord_name:8s} ({spoken})")
        sys.exit(0)

    # Step 2: Generate TTS clips for unique chords
    if args.verbose:
        print("Generating TTS clips...")

    chord_names = [e.chord_name for e in chord_events]
    unique_spoken = get_unique_chords(chord_names)

    if args.verbose:
        print(f"Unique chords to voice: {len(unique_spoken)}")

    try:
        tts = TTSGenerator(
            cache_dir=args.cache_dir,
            rate=args.rate
        )
        tts_clips = tts.generate_clips(unique_spoken)
    except Exception as e:
        print(f"Error generating TTS: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Mix audio
    if args.verbose:
        print("Mixing audio...")

    try:
        outputs = process_audio(
            input_path=input_path,
            chord_events=chord_events,
            tts_clips=tts_clips,
            output_dir=args.output_dir,
            min_gap_seconds=args.min_gap,
            voice_volume_db=args.voice_volume,
            original_volume_db=args.original_volume,
            voiced_only=args.voiced_only
        )
    except Exception as e:
        print(f"Error mixing audio: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print("\nOutput files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")

    if args.verbose:
        print("\nDone!")


if __name__ == "__main__":
    main()
