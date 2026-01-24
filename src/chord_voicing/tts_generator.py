"""Text-to-speech generation for chord names using pyttsx3."""

import hashlib
from pathlib import Path
from typing import Dict, Optional

import pyttsx3
from pydub import AudioSegment


class TTSGenerator:
    """Generates and caches TTS audio clips for chord names."""

    def __init__(
        self,
        cache_dir: str | Path = "cache",
        rate: int = 175,
        volume: float = 1.0,
        voice_id: Optional[str] = None
    ):
        """
        Initialize the TTS generator.

        Args:
            cache_dir: Directory to cache generated audio clips
            rate: Speech rate (words per minute, default 175)
            volume: Volume level 0.0 to 1.0
            voice_id: Voice ID to use (e.g., 'com.apple.voice.compact.en-US.Samantha')
                      If None, uses system default
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self._engine: Optional[pyttsx3.Engine] = None
        self._clip_cache: Dict[str, AudioSegment] = {}

    def _create_engine(self) -> pyttsx3.Engine:
        """Create a fresh TTS engine with configured settings."""
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        if self.voice_id:
            engine.setProperty('voice', self.voice_id)
        return engine

    def _get_cache_path(self, text: str) -> Path:
        """Get the cache file path for a given text."""
        # Create a hash of the text and settings for unique filename
        voice_str = self.voice_id or "default"
        settings_str = f"{text}_{self.rate}_{self.volume}_{voice_str}"
        hash_val = hashlib.md5(settings_str.encode()).hexdigest()[:12]
        safe_name = "".join(c if c.isalnum() else "_" for c in text[:20])
        # Use .aiff extension since pyttsx3 on macOS outputs AIFF-C format
        return self.cache_dir / f"{safe_name}_{hash_val}.aiff"

    def generate_clip(self, text: str) -> AudioSegment:
        """
        Generate an audio clip for the given text.

        Uses cache if available, otherwise generates and caches.

        Args:
            text: Text to convert to speech

        Returns:
            AudioSegment containing the spoken text
        """
        # Check memory cache first
        if text in self._clip_cache:
            return self._clip_cache[text]

        # Check disk cache
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            clip = AudioSegment.from_file(str(cache_path))
            # Only use cached clip if it has audio
            if len(clip) > 0:
                self._clip_cache[text] = clip
                return clip
            else:
                # Remove corrupted cache file
                cache_path.unlink()

        # Generate new clip
        # pyttsx3 on macOS outputs AIFF-C format regardless of extension
        temp_path = self.cache_dir / f"_temp_{hash(text)}.aiff"

        try:
            # Create fresh engine for each clip - pyttsx3 on macOS has
            # synchronization issues when reusing the engine
            engine = self._create_engine()
            engine.save_to_file(text, str(temp_path))
            engine.runAndWait()
            engine.stop()

            # Load the clip using auto-detect format
            clip = AudioSegment.from_file(str(temp_path))

            # Verify clip has audio
            if len(clip) == 0:
                raise RuntimeError("Generated clip is empty")

            self._clip_cache[text] = clip

            # Move temp file to cache
            temp_path.rename(cache_path)

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to generate TTS for '{text}': {e}") from e

        return clip

    def generate_clips(self, texts: set[str]) -> Dict[str, AudioSegment]:
        """
        Generate audio clips for multiple texts.

        Args:
            texts: Set of texts to convert to speech

        Returns:
            Dictionary mapping text to AudioSegment
        """
        clips = {}
        for text in texts:
            clips[text] = self.generate_clip(text)
        return clips

    def clear_cache(self):
        """Clear both memory and disk cache."""
        self._clip_cache.clear()
        for pattern in ["*.wav", "*.aiff"]:
            for cache_file in self.cache_dir.glob(pattern):
                if not cache_file.name.startswith("_temp"):
                    cache_file.unlink()

    def get_clip_duration(self, text: str) -> float:
        """
        Get the duration of a clip in seconds.

        Args:
            text: Text that was converted to speech

        Returns:
            Duration in seconds
        """
        clip = self.generate_clip(text)
        return len(clip) / 1000.0  # pydub uses milliseconds

    def close(self):
        """Clean up the TTS engine."""
        if self._engine is not None:
            self._engine.stop()
            self._engine = None


def get_available_voices() -> list:
    """Get list of available TTS voices.

    Returns:
        List of dicts with 'id' and 'name' for each voice
    """
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    result = [{'id': v.id, 'name': v.name} for v in voices]
    engine.stop()
    return result


def get_english_voices() -> list:
    """Get list of English TTS voices.

    Returns:
        List of dicts with 'id' and 'name' for each English voice
    """
    voices = get_available_voices()
    english_voices = []
    for v in voices:
        # Filter for English voices
        vid = v['id'].lower()
        if 'en-us' in vid or 'en-gb' in vid or 'en-au' in vid or 'en-' in vid:
            english_voices.append(v)
        elif any(name in v['name'].lower() for name in ['samantha', 'daniel', 'fred', 'karen', 'moira', 'tessa', 'rishi']):
            english_voices.append(v)
    return english_voices


# Some good default voice IDs for macOS
VOICE_SAMANTHA = 'com.apple.voice.compact.en-US.Samantha'  # Female, American
VOICE_DANIEL = 'com.apple.voice.compact.en-GB.Daniel'  # Male, British
VOICE_KAREN = 'com.apple.voice.compact.en-AU.Karen'  # Female, Australian
VOICE_FRED = 'com.apple.speech.synthesis.voice.Fred'  # Male, classic Mac
VOICE_MOIRA = 'com.apple.voice.compact.en-IE.Moira'  # Female, Irish
VOICE_SHELLEY = 'com.apple.eloquence.en-US.Shelley'  # Warm, gentle
