"""Chord Editor GUI - Edit chord timing and names with audio playback."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable
import threading
import time

# Audio playback
import pygame

from .chord_detector import ChordDetector, ChordEvent
from .chord_formatter import format_chord


@dataclass
class EditableChord:
    """A chord that can be edited."""
    start_time: float
    chord_name: str
    id: int  # Unique identifier for tracking

    def to_dict(self):
        return {'start_time': self.start_time, 'chord_name': self.chord_name}

    @classmethod
    def from_dict(cls, d, id):
        return cls(start_time=d['start_time'], chord_name=d['chord_name'], id=id)


@dataclass
class EditableNote:
    """A text note that can be edited and announced via TTS."""
    start_time: float
    text: str
    id: int  # Unique identifier for tracking

    def to_dict(self):
        return {'start_time': self.start_time, 'text': self.text}

    @classmethod
    def from_dict(cls, d, id):
        return cls(start_time=d['start_time'], text=d['text'], id=id)


@dataclass
class EditableVoiceNote:
    """A recorded voice note."""
    start_time: float
    file_path: str  # Path to the audio file
    id: int  # Unique identifier for tracking

    def to_dict(self):
        return {'start_time': self.start_time, 'file_path': self.file_path}

    @classmethod
    def from_dict(cls, d, id):
        return cls(start_time=d['start_time'], file_path=d['file_path'], id=id)


@dataclass
class EditableImage:
    """An image marker on the timeline."""
    start_time: float
    file_path: str  # Path to the image file
    id: int  # Unique identifier for tracking

    def to_dict(self):
        return {'start_time': self.start_time, 'file_path': self.file_path}

    @classmethod
    def from_dict(cls, d, id):
        return cls(start_time=d['start_time'], file_path=d['file_path'], id=id)


class AudioPlayer:
    """Audio player using pygame."""

    def __init__(self):
        pygame.mixer.init(frequency=44100)
        self.loaded_file: Optional[str] = None
        self.duration: float = 0
        self._position: float = 0
        self._playing: bool = False
        self._play_start_time: float = 0
        self._play_start_pos: float = 0

    def load(self, filepath: str):
        """Load an audio file."""
        pygame.mixer.music.load(filepath)
        self.loaded_file = filepath

        # Get duration using pydub
        from pydub import AudioSegment
        audio = AudioSegment.from_file(filepath)
        self.duration = len(audio) / 1000.0
        self._position = 0

    def play(self, start_pos: float = None):
        """Start playback from current or specified position."""
        if start_pos is not None:
            self._position = start_pos

        pygame.mixer.music.play(start=self._position)
        self._playing = True
        self._play_start_time = time.time()
        self._play_start_pos = self._position

    def pause(self):
        """Pause playback."""
        if self._playing:
            pygame.mixer.music.pause()
            self._position = self.get_position()
            self._playing = False

    def unpause(self):
        """Resume playback."""
        if not self._playing:
            pygame.mixer.music.unpause()
            self._playing = True
            self._play_start_time = time.time()
            self._play_start_pos = self._position

    def stop(self):
        """Stop playback."""
        pygame.mixer.music.stop()
        self._playing = False
        self._position = 0

    def seek(self, position: float):
        """Seek to a position in seconds."""
        was_playing = self._playing
        pygame.mixer.music.stop()
        self._position = max(0, min(position, self.duration))
        if was_playing:
            self.play(self._position)

    def get_position(self) -> float:
        """Get current playback position in seconds."""
        if self._playing:
            elapsed = time.time() - self._play_start_time
            pos = self._play_start_pos + elapsed
            return min(pos, self.duration)
        return self._position

    def is_playing(self) -> bool:
        return self._playing and pygame.mixer.music.get_busy()

    def set_volume(self, volume: float):
        """Set music volume (0.0 to 1.0)."""
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))

    def get_volume(self) -> float:
        """Get current music volume."""
        return pygame.mixer.music.get_volume()

    def cleanup(self):
        pygame.mixer.quit()


class ChordTimeline(tk.Canvas):
    """Timeline canvas showing chords."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.duration: float = 60  # Default duration
        self.chords: List[EditableChord] = []
        self.notes: List[EditableNote] = []
        self.voice_notes: List[EditableVoiceNote] = []
        self.images: List[EditableImage] = []
        self.selected_chord: Optional[EditableChord] = None
        self.selected_note: Optional[EditableNote] = None
        self.selected_voice_note: Optional[EditableVoiceNote] = None
        self.selected_image: Optional[EditableImage] = None
        self.playhead_pos: float = 0
        self.zoom: float = 1.0  # pixels per second
        self.scroll_offset: float = 0  # time offset for scrolling

        # Zoom constraints
        self.min_zoom: float = 5.0  # minimum pixels per second
        self.max_zoom: float = 200.0  # maximum pixels per second
        self.fit_zoom: float = 1.0  # zoom level to fit entire duration

        # Callbacks
        self.on_chord_selected: Optional[Callable] = None
        self.on_chord_moved: Optional[Callable] = None
        self.on_chord_edited: Optional[Callable] = None
        self.on_note_selected: Optional[Callable] = None
        self.on_note_moved: Optional[Callable] = None
        self.on_note_edited: Optional[Callable] = None
        self.on_voice_note_selected: Optional[Callable] = None
        self.on_voice_note_moved: Optional[Callable] = None
        self.on_voice_note_double_clicked: Optional[Callable] = None
        self.on_image_selected: Optional[Callable] = None
        self.on_image_moved: Optional[Callable] = None
        self.on_seek: Optional[Callable] = None
        self.on_zoom_changed: Optional[Callable] = None

        # Dragging state
        self._dragging = False
        self._drag_chord: Optional[EditableChord] = None
        self._drag_note: Optional[EditableNote] = None
        self._drag_voice_note: Optional[EditableVoiceNote] = None
        self._drag_image: Optional[EditableImage] = None
        self._drag_offset: float = 0

        # Panning state
        self._panning = False
        self._pan_start_x: float = 0
        self._pan_start_offset: float = 0

        # Beat grid and snap
        self.bpm: Optional[float] = None
        self.show_grid: bool = True
        self.snap_to_beat: bool = False

        # Bind events
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)
        self.bind('<Double-Button-1>', self._on_double_click)
        self.bind('<Configure>', self._on_resize)

        # Mouse wheel zoom (macOS uses MouseWheel, Linux uses Button-4/5)
        self.bind('<MouseWheel>', self._on_mousewheel)
        self.bind('<Button-4>', self._on_mousewheel)
        self.bind('<Button-5>', self._on_mousewheel)

        # Middle-click panning
        self.bind('<Button-2>', self._on_pan_start)
        self.bind('<B2-Motion>', self._on_pan_drag)
        self.bind('<ButtonRelease-2>', self._on_pan_end)

        # Compact (sliver) mode
        self.compact_mode: bool = False

        # Colors
        self.bg_color = '#2b2b2b'
        self.chord_color = '#4a9eff'
        self.note_color = '#ffa500'  # Orange for text notes
        self.voice_note_color = '#9b59b6'  # Purple for voice notes
        self.image_color = '#2ecc71'  # Green for image markers
        self.selected_color = '#ff6b6b'
        self.playhead_color = '#00ff00'
        self.grid_color = '#444444'
        self.text_color = '#ffffff'

        self.configure(bg=self.bg_color)

    def set_duration(self, duration: float):
        """Set the timeline duration."""
        self.duration = duration
        self._update_zoom()
        self.redraw()

    def set_chords(self, chords: List[EditableChord]):
        """Set the chords to display."""
        self.chords = sorted(chords, key=lambda c: c.start_time)
        self.redraw()

    def set_notes(self, notes: List[EditableNote]):
        """Set the notes to display."""
        self.notes = sorted(notes, key=lambda n: n.start_time)
        self.redraw()

    def set_voice_notes(self, voice_notes: List[EditableVoiceNote]):
        """Set the voice notes to display."""
        self.voice_notes = sorted(voice_notes, key=lambda v: v.start_time)
        self.redraw()

    def set_images(self, images: List[EditableImage]):
        """Set the images to display."""
        self.images = sorted(images, key=lambda i: i.start_time)
        self.redraw()

    def set_bpm(self, bpm: Optional[float]):
        """Set the BPM for beat grid."""
        self.bpm = bpm
        self.redraw()

    def snap_time_to_beat(self, time: float) -> float:
        """Snap a time value to the nearest beat."""
        if not self.bpm or not self.snap_to_beat:
            return time
        beat_duration = 60.0 / self.bpm
        return round(time / beat_duration) * beat_duration

    def set_playhead(self, position: float, auto_scroll: bool = True):
        """Update playhead position."""
        self.playhead_pos = position

        # Auto-scroll to keep playhead visible during playback
        if auto_scroll:
            visible_start, visible_end = self.get_visible_range()
            # Add some margin (10% of visible width)
            margin = (visible_end - visible_start) * 0.1
            if position > visible_end - margin:
                # Scroll to keep playhead at 20% from right edge
                self.scroll_offset = position - (visible_end - visible_start) * 0.8
                self._clamp_scroll_offset()
            elif position < visible_start + margin:
                # Scroll to keep playhead at 20% from left edge
                self.scroll_offset = position - (visible_end - visible_start) * 0.2
                self._clamp_scroll_offset()

        self.redraw()

    def _update_zoom(self):
        """Calculate fit zoom level (for reference)."""
        width = self.winfo_width() or 800
        self.fit_zoom = (width - 100) / max(self.duration, 1)
        # Set min_zoom to fit the entire duration
        self.min_zoom = max(5.0, self.fit_zoom * 0.5)
        # If zoom hasn't been set manually, use fit_zoom
        if self.zoom < self.min_zoom:
            self.zoom = self.fit_zoom

    def zoom_in(self, factor: float = 1.5, center_time: float = None):
        """Zoom in by factor, optionally centered on a time position."""
        if center_time is None:
            center_time = self.playhead_pos

        old_zoom = self.zoom
        self.zoom = min(self.zoom * factor, self.max_zoom)

        # Adjust scroll offset to keep center_time at same screen position
        if old_zoom != self.zoom:
            self._adjust_scroll_for_zoom(center_time, old_zoom)
            self.redraw()
            if self.on_zoom_changed:
                self.on_zoom_changed(self.zoom)

    def zoom_out(self, factor: float = 1.5, center_time: float = None):
        """Zoom out by factor, optionally centered on a time position."""
        if center_time is None:
            center_time = self.playhead_pos

        old_zoom = self.zoom
        self.zoom = max(self.zoom / factor, self.min_zoom)

        # Adjust scroll offset to keep center_time at same screen position
        if old_zoom != self.zoom:
            self._adjust_scroll_for_zoom(center_time, old_zoom)
            self.redraw()
            if self.on_zoom_changed:
                self.on_zoom_changed(self.zoom)

    def zoom_to_fit(self):
        """Reset zoom to fit entire duration in view."""
        self.zoom = self.fit_zoom
        self.scroll_offset = 0
        self.redraw()
        if self.on_zoom_changed:
            self.on_zoom_changed(self.zoom)

    def set_zoom(self, zoom_level: float):
        """Set zoom to a specific level."""
        self.zoom = max(self.min_zoom, min(zoom_level, self.max_zoom))
        self._clamp_scroll_offset()
        self.redraw()
        if self.on_zoom_changed:
            self.on_zoom_changed(self.zoom)

    def _adjust_scroll_for_zoom(self, center_time: float, old_zoom: float):
        """Adjust scroll offset when zooming to keep center_time in place."""
        width = self.winfo_width() or 800
        # Calculate screen position of center_time at old zoom
        old_x = 50 + (center_time - self.scroll_offset) * old_zoom
        # Calculate new scroll offset to keep same screen position
        self.scroll_offset = center_time - (old_x - 50) / self.zoom
        self._clamp_scroll_offset()

    def _clamp_scroll_offset(self):
        """Ensure scroll offset stays within valid bounds."""
        width = self.winfo_width() or 800
        visible_duration = (width - 100) / self.zoom
        max_offset = max(0, self.duration - visible_duration)
        self.scroll_offset = max(0, min(self.scroll_offset, max_offset))

    def scroll_to_time(self, time: float):
        """Scroll to make a specific time visible (centered if possible)."""
        width = self.winfo_width() or 800
        visible_duration = (width - 100) / self.zoom
        # Center the time in view
        self.scroll_offset = time - visible_duration / 2
        self._clamp_scroll_offset()
        self.redraw()

    def get_visible_range(self) -> tuple:
        """Get the visible time range (start, end)."""
        width = self.winfo_width() or 800
        visible_duration = (width - 100) / self.zoom
        return (self.scroll_offset, self.scroll_offset + visible_duration)

    def _on_mousewheel(self, event):
        """Handle mouse wheel for zooming."""
        # Get mouse position for zoom center
        mouse_time = self._x_to_time(event.x)

        # Determine scroll direction (macOS vs Linux)
        if event.num == 4 or event.delta > 0:
            self.zoom_in(factor=1.2, center_time=mouse_time)
        elif event.num == 5 or event.delta < 0:
            self.zoom_out(factor=1.2, center_time=mouse_time)

    def _on_pan_start(self, event):
        """Start panning with middle mouse button."""
        self._panning = True
        self._pan_start_x = event.x
        self._pan_start_offset = self.scroll_offset

    def _on_pan_drag(self, event):
        """Handle panning drag."""
        if self._panning:
            delta_x = event.x - self._pan_start_x
            delta_time = delta_x / self.zoom
            self.scroll_offset = self._pan_start_offset - delta_time
            self._clamp_scroll_offset()
            self.redraw()

    def _on_pan_end(self, event):
        """End panning."""
        self._panning = False

    def _time_to_x(self, t: float) -> float:
        """Convert time to x coordinate."""
        return 50 + (t - self.scroll_offset) * self.zoom

    def _x_to_time(self, x: float) -> float:
        """Convert x coordinate to time."""
        return (x - 50) / self.zoom + self.scroll_offset

    def redraw(self):
        """Redraw the timeline."""
        self.delete('all')
        width = self.winfo_width() or 800
        height = self.winfo_height() or 100

        if self.compact_mode:
            self._redraw_compact(width, height)
        else:
            self._redraw_full(width, height)

    def _redraw_compact(self, width: int, height: int):
        """Redraw timeline in compact sliver mode - all items on one line."""
        mid_y = height // 2

        # Draw playhead
        px = self._time_to_x(self.playhead_pos)
        if 0 <= px <= width:
            self.create_line(px, 0, px, height, fill=self.playhead_color, width=2)

        # Draw all items as small icons on the center line
        icon_size = 5
        for chord in self.chords:
            x = self._time_to_x(chord.start_time)
            if 0 <= x <= width:
                color = self.selected_color if chord == self.selected_chord else self.chord_color
                self.create_oval(x - icon_size, mid_y - icon_size, x + icon_size, mid_y + icon_size,
                               fill=color, outline='')

        for note in self.notes:
            x = self._time_to_x(note.start_time)
            if 0 <= x <= width:
                color = self.selected_color if note == self.selected_note else self.note_color
                self.create_polygon(x, mid_y - icon_size, x + icon_size, mid_y, x, mid_y + icon_size, x - icon_size, mid_y,
                                   fill=color, outline='')

        for voice_note in self.voice_notes:
            x = self._time_to_x(voice_note.start_time)
            if 0 <= x <= width:
                color = self.selected_color if voice_note == self.selected_voice_note else self.voice_note_color
                self.create_rectangle(x - icon_size, mid_y - icon_size, x + icon_size, mid_y + icon_size,
                                     fill=color, outline='')

        for image in self.images:
            x = self._time_to_x(image.start_time)
            if 0 <= x <= width:
                color = self.selected_color if image == self.selected_image else self.image_color
                self.create_rectangle(x - icon_size, mid_y - icon_size, x + icon_size, mid_y + icon_size,
                                     fill=color, outline='')

    def _redraw_full(self, width: int, height: int):
        """Redraw timeline in full mode with all details."""
        # Draw beat grid (if BPM is set and grid is enabled)
        if self.bpm and self.show_grid:
            beat_duration = 60.0 / self.bpm
            t = 0
            beat_num = 0
            while t <= self.duration:
                x = self._time_to_x(t)
                if 0 <= x <= width:
                    # Measure lines (every 4 beats) are brighter
                    if beat_num % 4 == 0:
                        self.create_line(x, 0, x, height, fill='#555555', width=1)
                    else:
                        self.create_line(x, 0, x, height, fill='#333333', width=1, dash=(2, 4))
                t += beat_duration
                beat_num += 1

        # Draw time grid
        grid_interval = self._get_grid_interval()
        t = 0
        while t <= self.duration:
            x = self._time_to_x(t)
            if 0 <= x <= width:
                self.create_line(x, 0, x, height, fill=self.grid_color)
                self.create_text(x, height - 10, text=f"{t:.1f}s",
                               fill=self.text_color, font=('Arial', 8))
            t += grid_interval

        # Draw chords (upper half)
        chord_y = height // 3
        for chord in self.chords:
            x = self._time_to_x(chord.start_time)
            if 0 <= x <= width:
                color = self.selected_color if chord == self.selected_chord else self.chord_color
                # Draw chord marker (circle)
                self.create_oval(x - 8, chord_y - 8, x + 8, chord_y + 8, fill=color, outline='white')
                # Draw chord name
                self.create_text(x, chord_y - 20, text=chord.chord_name,
                               fill=self.text_color, font=('Arial', 10, 'bold'))

        # Draw notes (middle section)
        note_y = height // 2
        for note in self.notes:
            x = self._time_to_x(note.start_time)
            if 0 <= x <= width:
                color = self.selected_color if note == self.selected_note else self.note_color
                # Draw note marker (diamond shape)
                self.create_polygon(x, note_y - 10, x + 10, note_y, x, note_y + 10, x - 10, note_y,
                                   fill=color, outline='white')
                # Draw note text (truncate if too long)
                display_text = note.text[:15] + '...' if len(note.text) > 15 else note.text
                self.create_text(x, note_y + 20, text=display_text,
                               fill=self.text_color, font=('Arial', 9, 'italic'))

        # Draw voice notes (lower section)
        voice_y = (height * 3) // 4
        for voice_note in self.voice_notes:
            x = self._time_to_x(voice_note.start_time)
            if 0 <= x <= width:
                color = self.selected_color if voice_note == self.selected_voice_note else self.voice_note_color
                # Draw voice note marker (square with mic symbol)
                self.create_rectangle(x - 8, voice_y - 8, x + 8, voice_y + 8, fill=color, outline='white')
                self.create_text(x, voice_y, text='\U0001f3a4', font=('Arial', 10))
                # Draw filename (truncate)
                filename = Path(voice_note.file_path).stem[:12]
                self.create_text(x, voice_y + 18, text=filename,
                               fill=self.text_color, font=('Arial', 8))

        # Draw image markers (bottom section)
        image_y = (height * 7) // 8
        for image in self.images:
            x = self._time_to_x(image.start_time)
            if 0 <= x <= width:
                color = self.selected_color if image == self.selected_image else self.image_color
                # Draw image marker (rectangle with picture symbol)
                self.create_rectangle(x - 8, image_y - 8, x + 8, image_y + 8, fill=color, outline='white')
                self.create_text(x, image_y, text='\U0001f5bc', font=('Arial', 10))
                # Draw filename (truncate)
                filename = Path(image.file_path).stem[:12]
                self.create_text(x, image_y + 18, text=filename,
                               fill=self.text_color, font=('Arial', 8))

        # Draw playhead
        px = self._time_to_x(self.playhead_pos)
        if 0 <= px <= width:
            self.create_line(px, 0, px, height, fill=self.playhead_color, width=2)

    def _get_grid_interval(self) -> float:
        """Get appropriate grid interval based on zoom."""
        if self.zoom > 50:
            return 0.5
        elif self.zoom > 20:
            return 1
        elif self.zoom > 5:
            return 5
        else:
            return 10

    def _find_chord_at(self, x: float, y: float) -> Optional[EditableChord]:
        """Find chord at given coordinates."""
        height = self.winfo_height() or 100
        chord_y = height // 2 if self.compact_mode else height // 3
        for chord in self.chords:
            cx = self._time_to_x(chord.start_time)
            if abs(cx - x) < 15 and abs(chord_y - y) < 15:
                return chord
        return None

    def _find_note_at(self, x: float, y: float) -> Optional[EditableNote]:
        """Find note at given coordinates."""
        height = self.winfo_height() or 100
        note_y = height // 2
        for note in self.notes:
            nx = self._time_to_x(note.start_time)
            if abs(nx - x) < 15 and abs(note_y - y) < 15:
                return note
        return None

    def _find_voice_note_at(self, x: float, y: float) -> Optional[EditableVoiceNote]:
        """Find voice note at given coordinates."""
        height = self.winfo_height() or 100
        voice_y = height // 2 if self.compact_mode else (height * 3) // 4
        for voice_note in self.voice_notes:
            vx = self._time_to_x(voice_note.start_time)
            if abs(vx - x) < 15 and abs(voice_y - y) < 15:
                return voice_note
        return None

    def _find_image_at(self, x: float, y: float) -> Optional[EditableImage]:
        """Find image at given coordinates."""
        height = self.winfo_height() or 100
        image_y = height // 2 if self.compact_mode else (height * 7) // 8
        for image in self.images:
            ix = self._time_to_x(image.start_time)
            if abs(ix - x) < 15 and abs(image_y - y) < 15:
                return image
        return None

    def _on_click(self, event):
        """Handle click event."""
        # Check for chord first
        chord = self._find_chord_at(event.x, event.y)
        if chord:
            self.selected_chord = chord
            self.selected_note = None
            self.selected_voice_note = None
            self.selected_image = None
            self._dragging = True
            self._drag_chord = chord
            self._drag_note = None
            self._drag_voice_note = None
            self._drag_image = None
            self._drag_offset = self._time_to_x(chord.start_time) - event.x
            if self.on_chord_selected:
                self.on_chord_selected(chord)
            self.redraw()
            return

        # Check for note
        note = self._find_note_at(event.x, event.y)
        if note:
            self.selected_note = note
            self.selected_chord = None
            self.selected_voice_note = None
            self.selected_image = None
            self._dragging = True
            self._drag_note = note
            self._drag_chord = None
            self._drag_voice_note = None
            self._drag_image = None
            self._drag_offset = self._time_to_x(note.start_time) - event.x
            if self.on_note_selected:
                self.on_note_selected(note)
            self.redraw()
            return

        # Check for voice note
        voice_note = self._find_voice_note_at(event.x, event.y)
        if voice_note:
            self.selected_voice_note = voice_note
            self.selected_chord = None
            self.selected_note = None
            self.selected_image = None
            self._dragging = True
            self._drag_voice_note = voice_note
            self._drag_chord = None
            self._drag_note = None
            self._drag_image = None
            self._drag_offset = self._time_to_x(voice_note.start_time) - event.x
            if self.on_voice_note_selected:
                self.on_voice_note_selected(voice_note)
            self.redraw()
            return

        # Check for image
        image = self._find_image_at(event.x, event.y)
        if image:
            self.selected_image = image
            self.selected_chord = None
            self.selected_note = None
            self.selected_voice_note = None
            self._dragging = True
            self._drag_image = image
            self._drag_chord = None
            self._drag_note = None
            self._drag_voice_note = None
            self._drag_offset = self._time_to_x(image.start_time) - event.x
            if self.on_image_selected:
                self.on_image_selected(image)
            self.redraw()
            return

        # Nothing clicked - clear selection and seek
        self.selected_chord = None
        self.selected_note = None
        self.selected_voice_note = None
        self.selected_image = None
        t = self._x_to_time(event.x)
        if 0 <= t <= self.duration and self.on_seek:
            self.on_seek(t)
        self.redraw()

    def _on_drag(self, event):
        """Handle drag event."""
        if self._dragging:
            new_time = self._x_to_time(event.x + self._drag_offset)
            new_time = max(0, min(new_time, self.duration))
            if self._drag_chord:
                self._drag_chord.start_time = new_time
                self.chords.sort(key=lambda c: c.start_time)
            elif self._drag_note:
                self._drag_note.start_time = new_time
                self.notes.sort(key=lambda n: n.start_time)
            elif self._drag_voice_note:
                self._drag_voice_note.start_time = new_time
                self.voice_notes.sort(key=lambda v: v.start_time)
            elif self._drag_image:
                self._drag_image.start_time = new_time
                self.images.sort(key=lambda i: i.start_time)
            self.redraw()

    def _on_release(self, event):
        """Handle release event."""
        if self._dragging:
            # Snap to beat if enabled
            if self._drag_chord:
                self._drag_chord.start_time = self.snap_time_to_beat(self._drag_chord.start_time)
                if self.on_chord_moved:
                    self.on_chord_moved(self._drag_chord)
            elif self._drag_note:
                self._drag_note.start_time = self.snap_time_to_beat(self._drag_note.start_time)
                if self.on_note_moved:
                    self.on_note_moved(self._drag_note)
            elif self._drag_voice_note:
                self._drag_voice_note.start_time = self.snap_time_to_beat(self._drag_voice_note.start_time)
                if self.on_voice_note_moved:
                    self.on_voice_note_moved(self._drag_voice_note)
            elif self._drag_image:
                self._drag_image.start_time = self.snap_time_to_beat(self._drag_image.start_time)
                if self.on_image_moved:
                    self.on_image_moved(self._drag_image)
            self.redraw()
        self._dragging = False
        self._drag_chord = None
        self._drag_note = None
        self._drag_voice_note = None
        self._drag_image = None

    def _on_double_click(self, event):
        """Handle double-click to edit chord or note."""
        chord = self._find_chord_at(event.x, event.y)
        if chord:
            new_name = simpledialog.askstring("Edit Chord", "Chord name:",
                                             initialvalue=chord.chord_name)
            if new_name:
                chord.chord_name = new_name
                self.redraw()
                # Notify editor to load TTS for new chord
                if self.on_chord_edited:
                    self.on_chord_edited(new_name)
            return

        note = self._find_note_at(event.x, event.y)
        if note:
            new_text = simpledialog.askstring("Edit Note", "Note text:",
                                             initialvalue=note.text)
            if new_text:
                note.text = new_text
                self.redraw()
                # Notify editor to load TTS for new note
                if self.on_note_edited:
                    self.on_note_edited(new_text)
            return

        voice_note = self._find_voice_note_at(event.x, event.y)
        if voice_note:
            if self.on_voice_note_double_clicked:
                self.on_voice_note_double_clicked(voice_note)

    def _on_resize(self, event):
        """Handle resize event."""
        self._update_zoom()
        self.redraw()


class ChordEditor:
    """Main chord editor application."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chord Editor")
        self.root.geometry("1200x750")

        self.player = AudioPlayer()
        self.chords: List[EditableChord] = []
        self.notes: List[EditableNote] = []
        self.voice_notes: List[EditableVoiceNote] = []
        self.images: List[EditableImage] = []
        self.next_id = 0
        self.audio_file: Optional[str] = None
        self.clipboard_chord: Optional[EditableChord] = None
        self.clipboard_note: Optional[EditableNote] = None

        # BPM and key tracking
        self.bpm: Optional[float] = None
        self.estimated_key: Optional[str] = None
        self.snap_to_beat: bool = False

        # Real-time chord/note preview
        self.preview_with_chords: bool = True
        self._tts_clips: dict = {}  # chord_name -> pygame.Sound
        self._note_clips: dict = {}  # note_text -> pygame.Sound
        self._voice_note_clips: dict = {}  # file_path -> pygame.Sound
        self._announced_chords: set = set()  # chord IDs announced this playback
        self._announced_notes: set = set()  # note IDs announced this playback
        self._announced_voice_notes: set = set()  # voice note IDs announced this playback
        self._announced_images: set = set()  # image IDs announced this playback
        self._current_display_image = None  # Reference to PhotoImage to prevent GC
        self._last_position: float = -1  # Start at -1 so items at time 0 get announced

        # Pause-for-notes state
        self._paused_for_note: bool = False
        self._note_resume_time: float = 0  # When to resume after note
        self._resume_position: float = 0  # Position to resume playback from

        # Audio ducking state (lower music volume during voice notes)
        self._ducked: bool = False
        self._duck_restore_time: float = 0  # When to restore volume
        self._duck_volume: float = 0.2  # Volume during voice note
        self._normal_volume: float = 1.0  # Volume to restore to

        # Fuller screen mode
        self._fuller_mode: bool = False
        self._timeline_normal_height: int = 150
        self._timeline_compact_height: int = 30

        # Auto-save state
        self._save_path: Optional[str] = None  # Set on manual save or load
        self._change_count: int = 0  # Changes since last backup
        self._backup_interval: int = 3  # Backup every N changes
        self._auto_save_pending: Optional[str] = None  # Scheduled after() ID

        self._setup_ui()
        self._setup_bindings()
        self._update_loop()

    def _setup_ui(self):
        """Setup the UI components."""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Audio...", command=self._open_audio, accelerator="Cmd+O")
        file_menu.add_command(label="Load Chords...", command=self._load_chords)
        file_menu.add_command(label="Load Chords from MIDI...", command=self._load_chords_from_midi)
        file_menu.add_command(label="Save Chords...", command=self._save_chords, accelerator="Cmd+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export Voiced Audio...", command=self._export_audio)
        file_menu.add_separator()
        file_menu.add_command(label="Detect Chords", command=self._detect_chords)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Add Chord", command=self._add_chord, accelerator="A")
        edit_menu.add_command(label="Add Note", command=self._add_note, accelerator="N")
        edit_menu.add_command(label="Add Image from File...", command=self._add_image_from_file, accelerator="I")
        edit_menu.add_command(label="Add Image from Clipboard", command=self._add_image_from_clipboard, accelerator="Shift+I")
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete Selected", command=self._delete_selected, accelerator="Delete")
        edit_menu.add_command(label="Copy Selected", command=self._copy_selected, accelerator="Cmd+C")
        edit_menu.add_command(label="Paste", command=self._paste, accelerator="Cmd+V")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self._zoom_in, accelerator="+")
        view_menu.add_command(label="Zoom Out", command=self._zoom_out, accelerator="-")
        view_menu.add_command(label="Zoom to Fit", command=self._zoom_fit, accelerator="0")
        view_menu.add_separator()
        view_menu.add_command(label="Go to Playhead", command=self._scroll_to_playhead)
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Fuller View", command=self._toggle_fuller_mode, accelerator="F")

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.play_btn = ttk.Button(control_frame, text="▶ Play", command=self._toggle_play, takefocus=False)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="⏹ Stop", command=self._stop, takefocus=False).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🗑 Delete", command=self._delete_chord_at_playhead, takefocus=False).pack(side=tk.LEFT, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.preview_var = tk.BooleanVar(value=True)
        self.preview_check = ttk.Checkbutton(
            control_frame, text="🔊 Hear Chords",
            variable=self.preview_var,
            command=self._toggle_preview_mode,
            takefocus=False
        )
        self.preview_check.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Position:").pack(side=tk.LEFT, padx=(20, 5))
        self.position_var = tk.StringVar(value="0:00.0")
        ttk.Label(control_frame, textvariable=self.position_var, width=10).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Duration:").pack(side=tk.LEFT, padx=(20, 5))
        self.duration_var = tk.StringVar(value="0:00.0")
        ttk.Label(control_frame, textvariable=self.duration_var, width=10).pack(side=tk.LEFT)

        # Zoom controls
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(control_frame, text="Zoom:").pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(control_frame, text="-", width=3, command=self._zoom_out, takefocus=False).pack(side=tk.LEFT)
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(control_frame, textvariable=self.zoom_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="+", width=3, command=self._zoom_in, takefocus=False).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Fit", width=4, command=self._zoom_fit, takefocus=False).pack(side=tk.LEFT, padx=(5, 0))

        # BPM, Key, and Snap controls (second row)
        info_control_frame = ttk.Frame(main_frame)
        info_control_frame.pack(fill=tk.X, pady=(5, 5))

        ttk.Label(info_control_frame, text="BPM:").pack(side=tk.LEFT, padx=(5, 2))
        self.bpm_var = tk.StringVar(value="--")
        self.bpm_entry = ttk.Entry(info_control_frame, textvariable=self.bpm_var, width=6)
        self.bpm_entry.pack(side=tk.LEFT)
        self.bpm_entry.bind('<Return>', self._on_bpm_change)
        self.bpm_entry.bind('<FocusOut>', self._on_bpm_change)

        ttk.Label(info_control_frame, text="Key:").pack(side=tk.LEFT, padx=(15, 2))
        self.key_var = tk.StringVar(value="--")
        ttk.Label(info_control_frame, textvariable=self.key_var, width=8).pack(side=tk.LEFT)

        ttk.Separator(info_control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.snap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            info_control_frame, text="Snap to Beat",
            variable=self.snap_var,
            command=self._toggle_snap,
            takefocus=False
        ).pack(side=tk.LEFT, padx=5)

        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            info_control_frame, text="Show Beat Grid",
            variable=self.grid_var,
            command=self._toggle_grid,
            takefocus=False
        ).pack(side=tk.LEFT, padx=5)

        ttk.Separator(info_control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.pause_for_notes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            info_control_frame, text="Pause for Notes",
            variable=self.pause_for_notes_var,
            takefocus=False
        ).pack(side=tk.LEFT, padx=5)

        ttk.Separator(info_control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.fuller_btn = ttk.Button(
            info_control_frame, text="Fuller View",
            command=self._toggle_fuller_mode, takefocus=False
        )
        self.fuller_btn.pack(side=tk.LEFT, padx=5)

        # Timeline frame with scrollbar (fixed height, does NOT expand)
        self.timeline_frame = ttk.Frame(main_frame)
        self.timeline_frame.pack(fill=tk.X)

        # Horizontal scrollbar
        self.h_scrollbar = ttk.Scrollbar(self.timeline_frame, orient=tk.HORIZONTAL, command=self._on_scroll)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Timeline
        self.timeline = ChordTimeline(self.timeline_frame, height=self._timeline_normal_height)
        self.timeline.pack(fill=tk.X)
        self.timeline.on_chord_selected = self._on_chord_selected
        self.timeline.on_chord_moved = self._on_chord_moved
        self.timeline.on_chord_edited = self._on_chord_edited
        self.timeline.on_note_selected = self._on_note_selected
        self.timeline.on_note_moved = self._on_note_moved
        self.timeline.on_note_edited = self._on_note_edited
        self.timeline.on_voice_note_selected = self._on_voice_note_selected
        self.timeline.on_voice_note_moved = self._on_voice_note_moved
        self.timeline.on_voice_note_double_clicked = self._on_voice_note_double_clicked
        self.timeline.on_image_selected = self._on_image_selected
        self.timeline.on_image_moved = self._on_image_moved
        self.timeline.on_seek = self._on_seek
        self.timeline.on_zoom_changed = self._on_zoom_changed

        # Image display frame (below timeline, expands to fill available space)
        self.image_display_frame = ttk.LabelFrame(main_frame, text="Image")
        self.image_display_label = tk.Label(self.image_display_frame, bg='#1a1a1a')
        self.image_display_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_display_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self._current_source_image = None
        self.image_display_frame.bind('<Configure>', lambda e: self._resize_current_image())

        # Selection info frame
        info_frame = ttk.LabelFrame(main_frame, text="Selected Item")
        info_frame.pack(fill=tk.X, pady=(10, 0))

        # Type indicator
        ttk.Label(info_frame, text="Type:").grid(row=0, column=0, padx=5, pady=5)
        self.selection_type_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.selection_type_var, width=8).grid(row=0, column=1, padx=5, pady=5)

        # Text/Name field
        ttk.Label(info_frame, text="Text:").grid(row=0, column=2, padx=5, pady=5)
        self.item_text_var = tk.StringVar()
        self.item_text_entry = ttk.Entry(info_frame, textvariable=self.item_text_var, width=20)
        self.item_text_entry.grid(row=0, column=3, padx=5, pady=5)
        self.item_text_entry.bind('<Return>', self._update_selected_text)

        # Time field
        ttk.Label(info_frame, text="Time:").grid(row=0, column=4, padx=5, pady=5)
        self.item_time_var = tk.StringVar()
        self.item_time_entry = ttk.Entry(info_frame, textvariable=self.item_time_var, width=10)
        self.item_time_entry.grid(row=0, column=5, padx=5, pady=5)
        self.item_time_entry.bind('<Return>', self._update_selected_time)

        # Spoken preview (for chords)
        ttk.Label(info_frame, text="Spoken:").grid(row=0, column=6, padx=5, pady=5)
        self.item_spoken_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.item_spoken_var, width=15).grid(row=0, column=7, padx=5, pady=5)

        # Quick-add chord buttons frame
        self.chord_buttons_frame = ttk.LabelFrame(main_frame, text="Quick Add Chord")
        self.chord_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        self._chord_buttons: list = []

        # Status bar
        self.status_var = tk.StringVar(value="Load an audio file to begin")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_bindings(self):
        """Setup keyboard bindings."""
        # Spacebar toggles play/pause globally
        self.root.bind_all('<space>', self._on_space)
        self.root.bind('<a>', lambda e: self._add_chord())
        self.root.bind('<n>', lambda e: self._add_note())
        self.root.bind('<v>', lambda e: self._record_voice_note())
        self.root.bind('<i>', lambda e: self._add_image_from_file())
        self.root.bind('<Shift-I>', lambda e: self._add_image_from_clipboard())
        self.root.bind('<f>', lambda e: self._toggle_fuller_mode())
        self.root.bind('<Delete>', lambda e: self._delete_selected())
        self.root.bind('<BackSpace>', lambda e: self._delete_selected())
        self.root.bind('<Command-c>', lambda e: self._copy_selected())
        self.root.bind('<Command-v>', lambda e: self._paste())
        self.root.bind('<Command-o>', lambda e: self._open_audio())
        self.root.bind('<Command-s>', lambda e: self._save_chords())
        self.root.bind('<Left>', lambda e: self._nudge(-0.1))
        self.root.bind('<Right>', lambda e: self._nudge(0.1))
        self.root.bind('<Shift-Left>', lambda e: self._nudge(-1.0))
        self.root.bind('<Shift-Right>', lambda e: self._nudge(1.0))
        self.root.bind('<plus>', lambda e: self._zoom_in())
        self.root.bind('<equal>', lambda e: self._zoom_in())  # + without shift
        self.root.bind('<minus>', lambda e: self._zoom_out())
        self.root.bind('<0>', lambda e: self._zoom_fit())  # 0 to fit

    def _update_loop(self):
        """Update loop for playhead position."""
        # Check if ducked volume should be restored
        if self._ducked and time.time() >= self._duck_restore_time:
            self.player.set_volume(self._normal_volume)
            self._ducked = False

        # Check if we're paused for a note and should resume
        if self._paused_for_note:
            if time.time() >= self._note_resume_time:
                # Note finished, resume playback
                self._paused_for_note = False
                self.player.play(self._resume_position)
                self.play_btn.config(text="⏸ Pause")
            # Don't update position while paused for note
            self.root.after(50, self._update_loop)
            return

        if self.player.is_playing():
            pos = self.player.get_position()
            self.timeline.set_playhead(pos)
            self._update_position_display(pos)
            self._update_scrollbar()

            # Real-time chord announcement
            if self.preview_with_chords:
                self._check_chord_announcements(pos)
            self._last_position = pos

        self.root.after(50, self._update_loop)

    def _update_position_display(self, pos: float):
        """Update position display."""
        mins = int(pos // 60)
        secs = pos % 60
        self.position_var.set(f"{mins}:{secs:04.1f}")

    def _format_duration(self, dur: float) -> str:
        mins = int(dur // 60)
        secs = dur % 60
        return f"{mins}:{secs:04.1f}"

    def _open_audio(self):
        """Open an audio file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.ogg"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.player.load(filepath)
                self.audio_file = filepath

                # Clear existing chords, notes, voice notes, and images
                self.chords = []
                self.notes = []
                self.voice_notes = []
                self.images = []
                self.timeline.set_chords(self.chords)
                self.timeline.set_notes(self.notes)
                self.timeline.set_voice_notes(self.voice_notes)
                self.timeline.set_images(self.images)
                self._clear_selection_info()
                self._hide_image_display()

                self.timeline.set_duration(self.player.duration)
                self.duration_var.set(self._format_duration(self.player.duration))
                self._update_scrollbar()
                self._on_zoom_changed(self.timeline.zoom)
                self.status_var.set(f"Loaded: {Path(filepath).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio: {e}")

    def _detect_chords(self):
        """Run chord detection on loaded audio."""
        if not self.audio_file:
            messagebox.showwarning("Warning", "Load an audio file first")
            return

        self.status_var.set("Detecting chords...")
        self.root.update()

        try:
            detector = ChordDetector()
            events = detector.detect_with_duration(self.audio_file, self.player.duration)

            self.chords = []
            for event in events:
                self.chords.append(EditableChord(
                    start_time=event.start_time,
                    chord_name=event.chord_name,
                    id=self._get_next_id()
                ))

            self.timeline.set_chords(self.chords)
            self._update_chord_buttons()
            self.status_var.set(f"Detected {len(self.chords)} chords")
            self._mark_dirty()
            # Reload TTS clips if preview mode is enabled
            if self.preview_with_chords:
                self._load_tts_clips()
        except Exception as e:
            messagebox.showerror("Error", f"Chord detection failed: {e}")
            self.status_var.set("Detection failed")

    def _get_next_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def _on_space(self, event):
        """Handle spacebar press - toggle play/pause unless in text entry."""
        focused = str(self.root.focus_get())
        # Allow space in entry widgets
        if 'entry' in focused.lower():
            return
        self._toggle_play()
        return "break"

    def _toggle_play(self, event=None):
        """Toggle play/pause."""
        # If paused for a note, cancel and resume
        if self._paused_for_note:
            self._paused_for_note = False
            self.player.play(self._resume_position)
            self.play_btn.config(text="⏸ Pause")
        elif self.player.is_playing():
            self.player.pause()
            self.play_btn.config(text="▶ Play")
        else:
            if self.player.loaded_file:
                # Always use play() with current position - unpause only works after pause
                self.player.play(self.player.get_position())
                self.play_btn.config(text="⏸ Pause")
        # Return focus to timeline so spacebar keeps working
        self.timeline.focus_set()

    def _stop(self):
        """Stop playback."""
        self.player.stop()
        self.play_btn.config(text="▶ Play")
        self.timeline.set_playhead(0)
        self._update_position_display(0)
        self._reset_announcements()
        self._last_position = -1
        self._paused_for_note = False
        self._hide_image_display()

    def _on_seek(self, time: float):
        """Handle seek from timeline."""
        self.player.seek(time)
        self.timeline.set_playhead(time)
        self._update_position_display(time)
        # Reset announcements - mark items before seek position as announced
        self._announced_chords.clear()
        self._announced_notes.clear()
        self._announced_voice_notes.clear()
        self._announced_images.clear()
        for chord in self.chords:
            if chord.start_time < time:
                self._announced_chords.add(chord.id)
        for note in self.notes:
            if note.start_time < time:
                self._announced_notes.add(note.id)
        for voice_note in self.voice_notes:
            if voice_note.start_time < time:
                self._announced_voice_notes.add(voice_note.id)
        for image in self.images:
            if image.start_time < time:
                self._announced_images.add(image.id)
        self._last_position = time
        self._paused_for_note = False

    def _on_chord_selected(self, chord: EditableChord):
        """Handle chord selection."""
        self.selection_type_var.set("Chord")
        self.item_text_var.set(chord.chord_name)
        self.item_time_var.set(f"{chord.start_time:.2f}")
        spoken = format_chord(chord.chord_name) or "(unknown)"
        self.item_spoken_var.set(spoken)

    def _on_chord_moved(self, chord: EditableChord):
        """Handle chord moved."""
        self.item_time_var.set(f"{chord.start_time:.2f}")
        self._mark_dirty()

    def _on_note_selected(self, note: EditableNote):
        """Handle note selection."""
        self.selection_type_var.set("Note")
        self.item_text_var.set(note.text)
        self.item_time_var.set(f"{note.start_time:.2f}")
        self.item_spoken_var.set("(as typed)")

    def _on_note_moved(self, note: EditableNote):
        """Handle note moved."""
        self.item_time_var.set(f"{note.start_time:.2f}")
        self._mark_dirty()

    def _on_chord_edited(self, chord_name: str):
        """Handle chord edited via double-click - load TTS for new name."""
        if chord_name not in self._tts_clips:
            self._load_single_tts(chord_name)
        self._update_chord_buttons()
        self._mark_dirty()

    def _on_note_edited(self, note_text: str):
        """Handle note edited via double-click - load TTS for new text."""
        if note_text not in self._note_clips:
            self._load_single_note_tts(note_text)
        self._mark_dirty()

    def _update_selected_text(self, event=None):
        """Update text of selected chord or note."""
        new_text = self.item_text_var.get()
        if self.timeline.selected_chord:
            self.timeline.selected_chord.chord_name = new_text
            spoken = format_chord(new_text) or "(unknown)"
            self.item_spoken_var.set(spoken)
            self.timeline.redraw()
            # Always load TTS for new chord name
            if new_text not in self._tts_clips:
                self._load_single_tts(new_text)
            self._update_chord_buttons()
            self._mark_dirty()
        elif self.timeline.selected_note:
            self.timeline.selected_note.text = new_text
            self.timeline.redraw()
            # Always load TTS for new note text
            if new_text not in self._note_clips:
                self._load_single_note_tts(new_text)
            self._mark_dirty()

    def _update_selected_time(self, event=None):
        """Update time of selected chord or note."""
        try:
            new_time = float(self.item_time_var.get())
            new_time = max(0, min(new_time, self.player.duration))
            changed = False
            if self.timeline.selected_chord:
                self.timeline.selected_chord.start_time = new_time
                self.timeline.chords.sort(key=lambda c: c.start_time)
                changed = True
            elif self.timeline.selected_note:
                self.timeline.selected_note.start_time = new_time
                self.timeline.notes.sort(key=lambda n: n.start_time)
                changed = True
            elif self.timeline.selected_image:
                self.timeline.selected_image.start_time = new_time
                self.timeline.images.sort(key=lambda i: i.start_time)
                changed = True
            self.timeline.redraw()
            if changed:
                self._mark_dirty()
        except ValueError:
            pass

    def _clear_selection_info(self):
        """Clear the selection info panel."""
        self.selection_type_var.set("-")
        self.item_text_var.set("")
        self.item_time_var.set("")
        self.item_spoken_var.set("")

    def _add_chord(self):
        """Add a new chord at playhead position."""
        pos = self.player.get_position()
        new_chord = EditableChord(start_time=pos, chord_name="C", id=self._get_next_id())
        self.chords.append(new_chord)
        self.chords.sort(key=lambda c: c.start_time)
        self.timeline.set_chords(self.chords)
        self.timeline.selected_chord = new_chord
        self.timeline.selected_note = None
        self._on_chord_selected(new_chord)
        self._update_chord_buttons()
        self.status_var.set(f"Added chord at {pos:.2f}s")
        self._mark_dirty()
        # Load TTS for new chord if preview is enabled and not already loaded
        if self.preview_with_chords:
            if "C" not in self._tts_clips:
                self._load_single_tts("C")
            # Play it immediately so user hears it
            if "C" in self._tts_clips:
                self._tts_clips["C"].play()
            # Mark as announced so it doesn't double-play
            self._announced_chords.add(new_chord.id)

    def _add_note(self):
        """Add a new note at playhead position."""
        pos = self.player.get_position()
        # Prompt for note text
        note_text = simpledialog.askstring("Add Note", "Enter note text:")
        if not note_text:
            return
        new_note = EditableNote(start_time=pos, text=note_text, id=self._get_next_id())
        self.notes.append(new_note)
        self.notes.sort(key=lambda n: n.start_time)
        self.timeline.set_notes(self.notes)
        self.timeline.selected_note = new_note
        self.timeline.selected_chord = None
        self._on_note_selected(new_note)
        self.status_var.set(f"Added note at {pos:.2f}s")
        self._mark_dirty()
        # Load TTS for new note if preview is enabled
        if self.preview_with_chords:
            if note_text not in self._note_clips:
                self._load_single_note_tts(note_text)
            # Play it immediately so user hears it
            if note_text in self._note_clips:
                self._note_clips[note_text].play()
            # Mark as announced so it doesn't double-play
            self._announced_notes.add(new_note.id)

    def _record_voice_note(self):
        """Record a voice note at playhead position."""
        if not self.audio_file:
            messagebox.showwarning("Warning", "Load an audio file first")
            return

        pos = self.player.get_position()

        # Create voicerecordings directory
        audio_dir = Path(self.audio_file).parent
        voice_dir = audio_dir / "voicerecordings"
        voice_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = int(time.time() * 1000)
        filename = f"voice_{timestamp}.wav"
        filepath = voice_dir / filename

        # Show recording dialog
        self._show_recording_dialog(filepath, pos)

    def _show_recording_dialog(self, filepath: Path, position: float):
        """Show a dialog for recording voice note."""
        import sounddevice as sd
        import numpy as np
        from scipy.io import wavfile

        dialog = tk.Toplevel(self.root)
        dialog.title("Record Voice Note")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        # Recording state
        recording_data = []
        is_recording = [False]
        sample_rate = 44100

        ttk.Label(dialog, text="Press Record to start, Stop to finish").pack(pady=10)

        status_var = tk.StringVar(value="Ready")
        ttk.Label(dialog, textvariable=status_var).pack(pady=5)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        def start_recording():
            is_recording[0] = True
            recording_data.clear()
            status_var.set("Recording... 🔴")
            record_btn.config(state='disabled')
            stop_btn.config(state='normal')

            def callback(indata, frames, time_info, status):
                if is_recording[0]:
                    recording_data.append(indata.copy())

            stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
            stream.start()
            dialog.stream = stream

        def stop_recording():
            is_recording[0] = False
            if hasattr(dialog, 'stream'):
                dialog.stream.stop()
                dialog.stream.close()
            status_var.set("Stopped")
            record_btn.config(state='normal')
            stop_btn.config(state='disabled')
            save_btn.config(state='normal')

        def save_recording():
            if recording_data:
                audio_data = np.concatenate(recording_data, axis=0)
                wavfile.write(str(filepath), sample_rate, audio_data)

                # Create voice note
                new_voice_note = EditableVoiceNote(
                    start_time=position,
                    file_path=str(filepath),
                    id=self._get_next_id()
                )
                self.voice_notes.append(new_voice_note)
                self.voice_notes.sort(key=lambda v: v.start_time)
                self.timeline.set_voice_notes(self.voice_notes)
                self.timeline.selected_voice_note = new_voice_note
                self._on_voice_note_selected(new_voice_note)

                # Load the clip for playback
                if self.preview_with_chords:
                    self._load_single_voice_clip(str(filepath))
                    if str(filepath) in self._voice_note_clips:
                        self._voice_note_clips[str(filepath)].play()
                    self._announced_voice_notes.add(new_voice_note.id)

                self.status_var.set(f"Voice note saved at {position:.2f}s")
                self._mark_dirty()
                dialog.destroy()
            else:
                messagebox.showwarning("Warning", "No recording to save")

        def cancel():
            if is_recording[0]:
                is_recording[0] = False
                if hasattr(dialog, 'stream'):
                    dialog.stream.stop()
                    dialog.stream.close()
            dialog.destroy()

        record_btn = ttk.Button(btn_frame, text="🔴 Record", command=start_recording)
        record_btn.pack(side=tk.LEFT, padx=5)

        stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=stop_recording, state='disabled')
        stop_btn.pack(side=tk.LEFT, padx=5)

        save_btn = ttk.Button(btn_frame, text="💾 Save", command=save_recording, state='disabled')
        save_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(dialog, text="Cancel", command=cancel).pack(pady=5)

        dialog.protocol("WM_DELETE_WINDOW", cancel)

    def _load_single_voice_clip(self, filepath: str):
        """Load a voice note clip for playback."""
        try:
            self._voice_note_clips[filepath] = pygame.mixer.Sound(filepath)
        except Exception as e:
            print(f"Failed to load voice clip {filepath}: {e}")

    def _on_voice_note_selected(self, voice_note: EditableVoiceNote):
        """Handle voice note selection."""
        self.selection_type_var.set("Voice")
        filename = Path(voice_note.file_path).stem
        self.item_text_var.set(filename)
        self.item_time_var.set(f"{voice_note.start_time:.2f}")
        self.item_spoken_var.set("(recorded)")

    def _on_voice_note_moved(self, voice_note: EditableVoiceNote):
        """Handle voice note moved."""
        self.item_time_var.set(f"{voice_note.start_time:.2f}")
        self._mark_dirty()

    def _on_voice_note_double_clicked(self, voice_note: EditableVoiceNote):
        """Handle double-click on voice note - play it if not currently playing audio."""
        if not self.player.is_playing():
            # Ensure clip is loaded
            if voice_note.file_path not in self._voice_note_clips:
                self._load_single_voice_clip(voice_note.file_path)
            if voice_note.file_path in self._voice_note_clips:
                self._voice_note_clips[voice_note.file_path].play()
                self.status_var.set(f"Playing voice note: {Path(voice_note.file_path).stem}")

    def _on_image_selected(self, image: EditableImage):
        """Handle image selection."""
        self.selection_type_var.set("Image")
        filename = Path(image.file_path).stem
        self.item_text_var.set(filename)
        self.item_time_var.set(f"{image.start_time:.2f}")
        self.item_spoken_var.set("(image)")
        # Show the image in the display frame
        self._show_image(image)

    def _on_image_moved(self, image: EditableImage):
        """Handle image moved."""
        self.item_time_var.set(f"{image.start_time:.2f}")
        self._mark_dirty()

    def _paste_clipboard_image(self, clip_image, pos: float):
        """Save a clipboard image and add it as an image marker."""
        if not self.audio_file:
            messagebox.showwarning("Warning", "Load an audio file first")
            return

        # Create images directory
        audio_dir = Path(self.audio_file).parent
        image_dir = audio_dir / "images"
        image_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = int(time.time() * 1000)
        filename = f"image_{timestamp}.png"
        filepath = image_dir / filename

        # Save the image
        clip_image.save(str(filepath), 'PNG')

        # Create image marker
        new_image = EditableImage(
            start_time=pos,
            file_path=str(filepath),
            id=self._get_next_id()
        )
        self.images.append(new_image)
        self.images.sort(key=lambda i: i.start_time)
        self.timeline.set_images(self.images)
        self.timeline.selected_image = new_image
        self._on_image_selected(new_image)
        self._announced_images.add(new_image.id)
        self.status_var.set(f"Pasted image at {pos:.2f}s")
        self._mark_dirty()

    def _add_image_from_clipboard(self):
        """Add an image from the system clipboard at the playhead position."""
        if not self.audio_file:
            messagebox.showwarning("Warning", "Load an audio file first")
            return

        try:
            from PIL import ImageGrab, Image as PILImage
            clip = ImageGrab.grabclipboard()

            if clip is None:
                messagebox.showinfo("Clipboard", "No image found on clipboard")
                return

            # On macOS, grabclipboard() can return a list of file paths
            if isinstance(clip, list):
                # Filter to image files
                image_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
                image_paths = [p for p in clip if Path(p).suffix.lower() in image_exts]
                if not image_paths:
                    messagebox.showinfo("Clipboard", "No image files found on clipboard")
                    return
                # Use the first image file directly
                pos = self.player.get_position()
                new_image = EditableImage(
                    start_time=pos,
                    file_path=image_paths[0],
                    id=self._get_next_id()
                )
                self.images.append(new_image)
                self.images.sort(key=lambda i: i.start_time)
                self.timeline.set_images(self.images)
                self.timeline.selected_image = new_image
                self._on_image_selected(new_image)
                self._announced_images.add(new_image.id)
                self.status_var.set(f"Added image from clipboard at {pos:.2f}s")
                self._mark_dirty()
                return

            if isinstance(clip, PILImage.Image):
                pos = self.player.get_position()
                self._paste_clipboard_image(clip, pos)
                return

            messagebox.showinfo("Clipboard", "No image found on clipboard")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get clipboard image: {e}")

    def _add_image_from_file(self):
        """Add an image from a file dialog at the playhead position."""
        if not self.audio_file:
            messagebox.showwarning("Warning", "Load an audio file first")
            return

        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not filepath:
            return

        pos = self.player.get_position()
        new_image = EditableImage(
            start_time=pos,
            file_path=filepath,
            id=self._get_next_id()
        )
        self.images.append(new_image)
        self.images.sort(key=lambda i: i.start_time)
        self.timeline.set_images(self.images)
        self.timeline.selected_image = new_image
        self._on_image_selected(new_image)
        self._announced_images.add(new_image.id)
        self.status_var.set(f"Added image at {pos:.2f}s")
        self._mark_dirty()

    def _show_image(self, image: EditableImage):
        """Display an image in the image display frame, scaled to best fill the area."""
        try:
            from PIL import Image, ImageTk

            img = Image.open(image.file_path)
            self._current_source_image = img.copy()

            self._resize_current_image()
        except Exception as e:
            self.status_var.set(f"Failed to display image: {e}")

    def _resize_current_image(self):
        """Resize the current source image to best fill the display area."""
        if not hasattr(self, '_current_source_image') or self._current_source_image is None:
            return
        try:
            from PIL import Image, ImageTk

            img = self._current_source_image
            avail_w = max(self.image_display_frame.winfo_width() - 20, 100)
            avail_h = max(self.image_display_frame.winfo_height() - 30, 100)
            img_w, img_h = img.size

            # Scale to best fill available area while keeping aspect ratio
            scale = min(avail_w / img_w, avail_h / img_h)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))

            resized = img.resize((new_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            self.image_display_label.configure(image=photo)
            self._current_display_image = photo
        except Exception:
            pass

    def _hide_image_display(self):
        """Clear the image display."""
        self.image_display_label.configure(image='')
        self._current_display_image = None
        self._current_source_image = None

    def _delete_selected(self):
        """Delete selected chord, note, or voice note."""
        if self.timeline.selected_chord:
            self.chords.remove(self.timeline.selected_chord)
            self.timeline.selected_chord = None
            self.timeline.set_chords(self.chords)
            self._update_chord_buttons()
            self._clear_selection_info()
            self.status_var.set("Chord deleted")
            self._mark_dirty()
        elif self.timeline.selected_note:
            self.notes.remove(self.timeline.selected_note)
            self.timeline.selected_note = None
            self.timeline.set_notes(self.notes)
            self._clear_selection_info()
            self.status_var.set("Note deleted")
            self._mark_dirty()
        elif self.timeline.selected_voice_note:
            self.voice_notes.remove(self.timeline.selected_voice_note)
            self.timeline.selected_voice_note = None
            self.timeline.set_voice_notes(self.voice_notes)
            self._clear_selection_info()
            self.status_var.set("Voice note deleted")
            self._mark_dirty()
        elif self.timeline.selected_image:
            self.images.remove(self.timeline.selected_image)
            self.timeline.selected_image = None
            self.timeline.set_images(self.images)
            self._clear_selection_info()
            self._hide_image_display()
            self.status_var.set("Image deleted")
            self._mark_dirty()

    def _delete_chord_at_playhead(self):
        """Delete selected chord, or closest chord to playhead if none selected."""
        # If a chord is selected, delete it
        if self.timeline.selected_chord:
            self._delete_selected()
            return

        # Otherwise find closest chord to playhead
        if not self.chords:
            self.status_var.set("No chords to delete")
            return

        pos = self.player.get_position()
        closest = min(self.chords, key=lambda c: abs(c.start_time - pos))
        self.chords.remove(closest)
        self.timeline.set_chords(self.chords)
        self._update_chord_buttons()
        self._clear_selection_info()
        self.status_var.set(f"Deleted {closest.chord_name} at {closest.start_time:.2f}s")
        self._mark_dirty()

    def _copy_selected(self):
        """Copy selected chord or note."""
        if self.timeline.selected_chord:
            self.clipboard_chord = EditableChord(
                start_time=self.timeline.selected_chord.start_time,
                chord_name=self.timeline.selected_chord.chord_name,
                id=0
            )
            self.clipboard_note = None
            self.status_var.set(f"Copied chord: {self.clipboard_chord.chord_name}")
        elif self.timeline.selected_note:
            self.clipboard_note = EditableNote(
                start_time=self.timeline.selected_note.start_time,
                text=self.timeline.selected_note.text,
                id=0
            )
            self.clipboard_chord = None
            self.status_var.set(f"Copied note: {self.clipboard_note.text[:20]}...")

    def _paste(self):
        """Paste chord or note at playhead position."""
        pos = self.player.get_position()
        if self.clipboard_chord:
            new_chord = EditableChord(
                start_time=pos,
                chord_name=self.clipboard_chord.chord_name,
                id=self._get_next_id()
            )
            self.chords.append(new_chord)
            self.chords.sort(key=lambda c: c.start_time)
            self.timeline.set_chords(self.chords)
            self._update_chord_buttons()
            self.status_var.set(f"Pasted chord: {new_chord.chord_name} at {pos:.2f}s")
            self._mark_dirty()
        elif self.clipboard_note:
            new_note = EditableNote(
                start_time=pos,
                text=self.clipboard_note.text,
                id=self._get_next_id()
            )
            self.notes.append(new_note)
            self.notes.sort(key=lambda n: n.start_time)
            self.timeline.set_notes(self.notes)
            self.status_var.set(f"Pasted note at {pos:.2f}s")
            self._mark_dirty()

    def _nudge(self, delta: float):
        """Nudge selected chord or note by delta seconds."""
        if self.timeline.selected_chord:
            new_time = self.timeline.selected_chord.start_time + delta
            new_time = max(0, min(new_time, self.player.duration))
            self.timeline.selected_chord.start_time = new_time
            self.item_time_var.set(f"{new_time:.2f}")
            self.timeline.chords.sort(key=lambda c: c.start_time)
            self.timeline.redraw()
            self._mark_dirty()
        elif self.timeline.selected_note:
            new_time = self.timeline.selected_note.start_time + delta
            new_time = max(0, min(new_time, self.player.duration))
            self.timeline.selected_note.start_time = new_time
            self.item_time_var.set(f"{new_time:.2f}")
            self.timeline.notes.sort(key=lambda n: n.start_time)
            self.timeline.redraw()
            self._mark_dirty()
        elif self.timeline.selected_image:
            new_time = self.timeline.selected_image.start_time + delta
            new_time = max(0, min(new_time, self.player.duration))
            self.timeline.selected_image.start_time = new_time
            self.item_time_var.set(f"{new_time:.2f}")
            self.timeline.images.sort(key=lambda i: i.start_time)
            self.timeline.redraw()
            self._mark_dirty()

    def _zoom_in(self):
        """Zoom in on the timeline."""
        self.timeline.zoom_in(factor=1.5)
        self._update_scrollbar()

    def _zoom_out(self):
        """Zoom out on the timeline."""
        self.timeline.zoom_out(factor=1.5)
        self._update_scrollbar()

    def _zoom_fit(self):
        """Zoom to fit entire duration."""
        self.timeline.zoom_to_fit()
        self._update_scrollbar()

    def _on_bpm_change(self, event=None):
        """Handle BPM entry change."""
        try:
            bpm_text = self.bpm_var.get().strip()
            if bpm_text and bpm_text != "--":
                self.bpm = float(bpm_text)
                self.timeline.set_bpm(self.bpm)
                self.timeline.redraw()
        except ValueError:
            pass  # Ignore invalid input

    def _toggle_snap(self):
        """Toggle snap to beat."""
        self.snap_to_beat = self.snap_var.get()
        self.timeline.snap_to_beat = self.snap_to_beat

    def _toggle_grid(self):
        """Toggle beat grid visibility."""
        self.timeline.show_grid = self.grid_var.get()
        self.timeline.redraw()

    def _toggle_fuller_mode(self):
        """Toggle fuller screen mode - shrinks timeline to a sliver."""
        self._fuller_mode = not self._fuller_mode
        if self._fuller_mode:
            self.fuller_btn.config(text="Normal View")
            self.timeline.compact_mode = True
            self.timeline.configure(height=self._timeline_compact_height)
        else:
            self.fuller_btn.config(text="Fuller View")
            self.timeline.compact_mode = False
            self.timeline.configure(height=self._timeline_normal_height)
        self.timeline.redraw()

    def _estimate_key(self):
        """Estimate the musical key from chord roots."""
        if not self.chords:
            self.estimated_key = None
            self.key_var.set("--")
            return

        from collections import Counter

        # Extract root notes from chord names
        roots = []
        for chord in self.chords:
            name = chord.chord_name
            if not name:
                continue
            # Get root: first letter + optional # or b
            root = name[0]
            if len(name) > 1 and name[1] in '#b':
                root += name[1]
            roots.append(root)

        if not roots:
            self.estimated_key = None
            self.key_var.set("--")
            return

        root_counts = Counter(roots)
        most_common = root_counts.most_common(3)

        # Simple heuristic: if minor chords dominate, suggest minor key
        minor_count = sum(1 for c in self.chords if 'm' in c.chord_name.lower() and 'maj' not in c.chord_name.lower())
        total = len(self.chords)

        top_root = most_common[0][0]
        if minor_count > total * 0.4:
            self.estimated_key = f"{top_root}m"
        else:
            self.estimated_key = top_root

        self.key_var.set(self.estimated_key)

    def _on_zoom_changed(self, zoom: float):
        """Handle zoom level change."""
        # Update zoom percentage display
        if self.timeline.fit_zoom > 0:
            pct = int(100 * zoom / self.timeline.fit_zoom)
            self.zoom_var.set(f"{pct}%")
        self._update_scrollbar()

    def _update_scrollbar(self):
        """Update scrollbar position and size."""
        if self.player.duration <= 0:
            return
        visible_start, visible_end = self.timeline.get_visible_range()
        visible_duration = visible_end - visible_start
        # Scrollbar position as fraction of total duration
        start_frac = max(0, visible_start / self.player.duration)
        end_frac = min(1, visible_end / self.player.duration)
        self.h_scrollbar.set(start_frac, end_frac)

    def _on_scroll(self, *args):
        """Handle scrollbar movement."""
        if args[0] == 'moveto':
            frac = float(args[1])
            # Convert fraction to time offset
            self.timeline.scroll_offset = frac * self.player.duration
            self.timeline._clamp_scroll_offset()
            self.timeline.redraw()
        elif args[0] == 'scroll':
            amount = int(args[1])
            # Scroll by a portion of visible width
            visible_start, visible_end = self.timeline.get_visible_range()
            visible_duration = visible_end - visible_start
            scroll_amount = visible_duration * 0.1 * amount
            self.timeline.scroll_offset += scroll_amount
            self.timeline._clamp_scroll_offset()
            self.timeline.redraw()
            self._update_scrollbar()

    def _scroll_to_playhead(self):
        """Scroll to center the playhead in view."""
        self.timeline.scroll_to_time(self.player.get_position())
        self._update_scrollbar()

    def _update_chord_buttons(self):
        """Update quick-add chord buttons based on unique chords in the song."""
        # Clear existing buttons
        for btn in self._chord_buttons:
            btn.destroy()
        self._chord_buttons.clear()

        # Get unique chord names, sorted
        unique_chords = sorted(set(c.chord_name for c in self.chords))

        # Create a button for each chord
        for chord_name in unique_chords:
            btn = ttk.Button(
                self.chord_buttons_frame,
                text=chord_name,
                command=lambda cn=chord_name: self._quick_add_chord(cn),
                takefocus=False
            )
            btn.pack(side=tk.LEFT, padx=2, pady=5)
            self._chord_buttons.append(btn)

    def _quick_add_chord(self, chord_name: str):
        """Add a chord at the current playhead position."""
        pos = self.player.get_position()
        new_chord = EditableChord(start_time=pos, chord_name=chord_name, id=self._get_next_id())
        self.chords.append(new_chord)
        self.chords.sort(key=lambda c: c.start_time)
        self.timeline.set_chords(self.chords)
        self.timeline.selected_chord = new_chord
        self._on_chord_selected(new_chord)
        # Load TTS if needed
        if chord_name not in self._tts_clips:
            self._load_single_tts(chord_name)
        self.status_var.set(f"Added {chord_name} at {pos:.2f}s")
        self._mark_dirty()

    def _toggle_preview_mode(self):
        """Toggle chord preview mode."""
        self.preview_with_chords = self.preview_var.get()
        if self.preview_with_chords:
            self._load_tts_clips()
            self.status_var.set("Chord preview enabled")
        else:
            self.status_var.set("Chord preview disabled")

    def _load_tts_clips(self):
        """Load TTS clips for all current chords and notes as pygame Sound objects."""
        if not self.chords and not self.notes and not self.voice_notes:
            print("[TTS] No chords, notes, or voice notes to load")
            return

        self.status_var.set("Loading sounds...")
        self.root.update()

        try:
            from .chord_formatter import format_chord
            from .tts_generator import TTSGenerator, VOICE_SAMANTHA
            import tempfile
            import os

            # Use same voice for chords and notes
            tts = TTSGenerator(cache_dir="cache", rate=175, voice_id=VOICE_SAMANTHA)
            temp_dir = tempfile.mkdtemp()

            # Load chord TTS clips
            unique_chord_names = set(c.chord_name for c in self.chords)
            self._tts_clips = {}
            print(f"[TTS] Loading {len(unique_chord_names)} unique chord names...")

            for chord_name in unique_chord_names:
                spoken = format_chord(chord_name)
                if not spoken:
                    print(f"[TTS]   Skipping chord '{chord_name}' - no spoken form")
                    continue
                clip = tts.generate_clip(spoken)
                safe_name = chord_name.replace('/', '_').replace('#', 'sharp')
                temp_path = os.path.join(temp_dir, f"chord_{safe_name}.wav")
                clip.export(temp_path, format="wav")
                self._tts_clips[chord_name] = pygame.mixer.Sound(temp_path)
                print(f"[TTS]   Loaded chord '{chord_name}' -> '{spoken}'")

            # Load note TTS clips (slower rate for better comprehension)
            note_tts = TTSGenerator(cache_dir="cache", rate=130, voice_id=VOICE_SAMANTHA)
            unique_note_texts = set(n.text for n in self.notes)
            self._note_clips = {}

            for note_text in unique_note_texts:
                # Add pauses after periods for better pacing
                spoken_text = note_text.replace('. ', '... ').replace('.', '...')
                clip = note_tts.generate_clip(spoken_text)
                safe_name = "".join(c if c.isalnum() else '_' for c in note_text[:20])
                temp_path = os.path.join(temp_dir, f"note_{safe_name}.wav")
                clip.export(temp_path, format="wav")
                self._note_clips[note_text] = pygame.mixer.Sound(temp_path)
                print(f"[TTS]   Loaded note '{note_text[:30]}'")

            # Load voice note clips
            self._voice_note_clips = {}
            for voice_note in self.voice_notes:
                if Path(voice_note.file_path).exists():
                    try:
                        self._voice_note_clips[voice_note.file_path] = pygame.mixer.Sound(voice_note.file_path)
                        print(f"[TTS]   Loaded voice note: {voice_note.file_path}")
                    except Exception as e:
                        print(f"[TTS]   FAILED voice clip {voice_note.file_path}: {e}")
                else:
                    print(f"[TTS]   Voice note file NOT FOUND: {voice_note.file_path}")

            total = len(self._tts_clips) + len(self._note_clips) + len(self._voice_note_clips)
            print(f"[TTS] Total loaded: {total} ({len(self._tts_clips)} chords, {len(self._note_clips)} notes, {len(self._voice_note_clips)} voice)")
            self.status_var.set(f"Loaded {total} sounds ({len(self._tts_clips)} chords, {len(self._note_clips)} notes, {len(self._voice_note_clips)} voice)")
        except Exception as e:
            print(f"[TTS] FAILED to load sounds: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Failed to load sounds: {e}")
            self.preview_var.set(False)
            self.preview_with_chords = False

    def _check_chord_announcements(self, current_pos: float):
        """Check if we need to announce any chords, notes, or voice notes at current position."""
        # Check chords
        for chord in self.chords:
            if (self._last_position < chord.start_time <= current_pos and
                chord.id not in self._announced_chords):
                self._announce_chord(chord)
                self._announced_chords.add(chord.id)

        # Check notes
        for note in self.notes:
            if (self._last_position < note.start_time <= current_pos and
                note.id not in self._announced_notes):
                self._announce_note(note)
                self._announced_notes.add(note.id)

        # Check voice notes
        for voice_note in self.voice_notes:
            if (self._last_position < voice_note.start_time <= current_pos and
                voice_note.id not in self._announced_voice_notes):
                self._announce_voice_note(voice_note)
                self._announced_voice_notes.add(voice_note.id)

        # Check images — display when playhead passes, image stays until next image
        for image in self.images:
            if (self._last_position < image.start_time <= current_pos and
                image.id not in self._announced_images):
                self._show_image(image)
                self._announced_images.add(image.id)

    def _announce_chord(self, chord: EditableChord):
        """Play the TTS clip for a chord."""
        if chord.chord_name in self._tts_clips:
            print(f"[ANNOUNCE] Playing chord '{chord.chord_name}' at {chord.start_time:.2f}s")
            self._tts_clips[chord.chord_name].play()
        else:
            print(f"[ANNOUNCE] No clip for chord '{chord.chord_name}' - available: {list(self._tts_clips.keys())[:5]}")

    def _announce_note(self, note: EditableNote):
        """Play the TTS clip for a note."""
        if note.text in self._note_clips:
            clip = self._note_clips[note.text]

            # If pause-for-notes is enabled, pause audio and resume after note
            if self.pause_for_notes_var.get() and self.player.is_playing():
                # Get current position before pausing
                self._resume_position = self.player.get_position()

                # Pause the audio
                self.player.pause()
                self.play_btn.config(text="▶ Play")

                # Play the note
                clip.play()

                # Calculate when to resume (clip duration + 0.5s pause after note)
                clip_duration = clip.get_length()
                self._note_resume_time = time.time() + clip_duration + 0.5
                self._paused_for_note = True
            else:
                # Just play the note without pausing
                clip.play()

    def _announce_voice_note(self, voice_note: EditableVoiceNote):
        """Play a voice note recording."""
        if voice_note.file_path in self._voice_note_clips:
            clip = self._voice_note_clips[voice_note.file_path]

            # If pause-for-notes is enabled, pause audio and resume after voice note
            if self.pause_for_notes_var.get() and self.player.is_playing():
                # Get current position before pausing
                self._resume_position = self.player.get_position()

                # Pause the audio
                self.player.pause()
                self.play_btn.config(text="▶ Play")

                # Play the voice note
                clip.play()

                # Calculate when to resume (clip duration + 0.5s pause)
                clip_duration = clip.get_length()
                self._note_resume_time = time.time() + clip_duration + 0.5
                self._paused_for_note = True
            else:
                # Duck the music volume while voice note plays
                if self.player.is_playing():
                    self._normal_volume = self.player.get_volume()
                    self.player.set_volume(self._duck_volume)
                    self._ducked = True
                    self._duck_restore_time = time.time() + clip.get_length() + 0.3
                clip.play()

    def _load_single_tts(self, chord_name: str):
        """Load TTS clip for a single chord name."""
        try:
            from .chord_formatter import format_chord
            from .tts_generator import TTSGenerator, VOICE_SAMANTHA
            import tempfile
            import os

            spoken = format_chord(chord_name)
            if not spoken:
                return

            tts = TTSGenerator(cache_dir="cache", rate=175, voice_id=VOICE_SAMANTHA)
            clip = tts.generate_clip(spoken)

            temp_dir = tempfile.mkdtemp()
            safe_name = chord_name.replace('/', '_').replace('#', 'sharp')
            temp_path = os.path.join(temp_dir, f"{safe_name}.wav")
            clip.export(temp_path, format="wav")
            self._tts_clips[chord_name] = pygame.mixer.Sound(temp_path)
        except Exception as e:
            print(f"Failed to load TTS for chord {chord_name}: {e}")

    def _load_single_note_tts(self, note_text: str):
        """Load TTS clip for a single note text (slower rate for comprehension)."""
        try:
            from .tts_generator import TTSGenerator, VOICE_SAMANTHA
            import tempfile
            import os

            # Use slower rate for notes
            tts = TTSGenerator(cache_dir="cache", rate=130, voice_id=VOICE_SAMANTHA)
            # Add pauses after periods
            spoken_text = note_text.replace('. ', '... ').replace('.', '...')
            clip = tts.generate_clip(spoken_text)

            temp_dir = tempfile.mkdtemp()
            safe_name = "".join(c if c.isalnum() else '_' for c in note_text[:20])
            temp_path = os.path.join(temp_dir, f"{safe_name}.wav")
            clip.export(temp_path, format="wav")
            self._note_clips[note_text] = pygame.mixer.Sound(temp_path)
        except Exception as e:
            print(f"Failed to load TTS for note {note_text}: {e}")

    def _reset_announcements(self):
        """Reset announced chords, notes, voice notes, and images (called on seek/stop)."""
        self._announced_chords.clear()
        self._announced_notes.clear()
        self._announced_voice_notes.clear()
        self._announced_images.clear()

    def _get_save_data(self) -> dict:
        """Build the save data dictionary."""
        return {
            'audio_file': self.audio_file,
            'duration': self.player.duration,
            'chords': [c.to_dict() for c in self.chords],
            'notes': [n.to_dict() for n in self.notes],
            'voice_notes': [v.to_dict() for v in self.voice_notes],
            'images': [i.to_dict() for i in self.images]
        }

    def _save_to_file(self, filepath: str):
        """Write save data to a file."""
        with open(filepath, 'w') as f:
            json.dump(self._get_save_data(), f, indent=2)

    def _rotate_backups(self):
        """Rotate backup files: .2backup -> .3backup, .1backup -> .2backup, current -> .1backup."""
        if not self._save_path:
            return
        p = Path(self._save_path)
        b3 = p.with_suffix(p.suffix + '.3backup')
        b2 = p.with_suffix(p.suffix + '.2backup')
        b1 = p.with_suffix(p.suffix + '.1backup')

        # Rotate: .2backup -> .3backup
        if b2.exists():
            import shutil
            shutil.copy2(str(b2), str(b3))
        # .1backup -> .2backup
        if b1.exists():
            import shutil
            shutil.copy2(str(b1), str(b2))
        # current -> .1backup
        if p.exists():
            import shutil
            shutil.copy2(str(p), str(b1))

    def _mark_dirty(self):
        """Mark data as changed. Debounces auto-save to 1 second after last change."""
        if not self._save_path:
            # Derive save path from audio file if possible
            if self.audio_file:
                audio_path = Path(self.audio_file)
                self._save_path = str(audio_path.parent / f"{audio_path.stem}_chords.json")
            else:
                return

        self._change_count += 1

        # Cancel any pending auto-save and schedule a new one
        if self._auto_save_pending:
            self.root.after_cancel(self._auto_save_pending)
        self._auto_save_pending = self.root.after(1000, self._do_auto_save)

    def _do_auto_save(self):
        """Perform the actual auto-save (called after debounce delay)."""
        self._auto_save_pending = None
        if not self._save_path:
            return

        # Rotate backups every N changes
        if self._change_count % self._backup_interval == 0:
            self._rotate_backups()

        try:
            self._save_to_file(self._save_path)
        except Exception as e:
            print(f"[AUTO-SAVE] Failed: {e}")

    def _save_chords(self):
        """Save chords, notes, and voice notes to JSON file."""
        initial_dir = Path(self.audio_file).parent if self.audio_file else None
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=initial_dir,
            initialfile="chords.json"
        )
        if filepath:
            self._save_path = filepath
            self._save_to_file(filepath)
            self.status_var.set(f"Saved to {Path(filepath).name}")

    def _load_chords(self):
        """Load chords, notes, and voice notes from JSON file."""
        initial_dir = Path(self.audio_file).parent if self.audio_file else None
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        if filepath:
            try:
                self._save_path = filepath
                self._change_count = 0
                with open(filepath, 'r') as f:
                    data = json.load(f)

                self.chords = []
                for c in data.get('chords', []):
                    self.chords.append(EditableChord.from_dict(c, self._get_next_id()))

                self.notes = []
                for n in data.get('notes', []):
                    self.notes.append(EditableNote.from_dict(n, self._get_next_id()))

                self.voice_notes = []
                for v in data.get('voice_notes', []):
                    self.voice_notes.append(EditableVoiceNote.from_dict(v, self._get_next_id()))

                self.images = []
                for i in data.get('images', []):
                    self.images.append(EditableImage.from_dict(i, self._get_next_id()))

                self.timeline.set_chords(self.chords)
                self.timeline.set_notes(self.notes)
                self.timeline.set_voice_notes(self.voice_notes)
                self.timeline.set_images(self.images)
                self._update_chord_buttons()

                # Reset announcement tracking
                self._reset_announcements()
                self._last_position = -1

                # Load TTS clips for preview
                tts_status = ""
                if self.preview_with_chords:
                    self._load_tts_clips()
                    if self.preview_with_chords:
                        # TTS succeeded - include clip counts
                        tts_status = f" | Sounds: {len(self._tts_clips)} chords, {len(self._note_clips)} notes, {len(self._voice_note_clips)} voice"
                    else:
                        # TTS failed and disabled preview - show warning
                        tts_status = " | WARNING: Sound loading failed - check console"

                self.status_var.set(f"Loaded {len(self.chords)} chords, {len(self.notes)} notes, {len(self.voice_notes)} voice notes, {len(self.images)} images from {Path(filepath).name}{tts_status}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")

    def _load_chords_from_midi(self):
        """Load chords from a MIDI file."""
        initial_dir = Path(self.audio_file).parent if self.audio_file else None
        filepath = filedialog.askopenfilename(
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        if filepath:
            try:
                from .midi_loader import load_midi_info

                midi_info = load_midi_info(filepath)
                midi_chords = midi_info['chords']

                if not midi_chords:
                    messagebox.showwarning("Warning", "No chords found in MIDI file")
                    return

                # Convert to EditableChord objects
                self.chords = []
                for time, chord_name in midi_chords:
                    self.chords.append(EditableChord(
                        start_time=time,
                        chord_name=chord_name,
                        id=self._get_next_id()
                    ))

                self.timeline.set_chords(self.chords)
                self._update_chord_buttons()

                # Set BPM if found in MIDI
                if midi_info['bpm']:
                    self.bpm = midi_info['bpm']
                    self.bpm_var.set(f"{self.bpm:.1f}")
                    self.timeline.set_bpm(self.bpm)

                # Estimate key from chords
                self._estimate_key()

                status_parts = [f"Loaded {len(self.chords)} chords"]
                if midi_info['bpm']:
                    status_parts.append(f"BPM: {midi_info['bpm']:.0f}")
                if self.estimated_key:
                    status_parts.append(f"Key: {self.estimated_key}")
                self.status_var.set(f"{' | '.join(status_parts)} from {Path(filepath).name}")

                self._mark_dirty()
                # Load TTS for new chords if preview is enabled
                if self.preview_with_chords:
                    self._load_tts_clips()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load MIDI: {e}")

    def _export_audio(self):
        """Export voiced audio (mixed) and voice-only track."""
        if not self.audio_file or (not self.chords and not self.notes and not self.voice_notes):
            messagebox.showwarning("Warning", "Load audio and add chords/notes first")
            return

        # Generate default filename based on original audio
        original_name = Path(self.audio_file).stem
        default_name = f"voiced{original_name}.mp3"
        initial_dir = Path(self.audio_file).parent

        filepath = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")],
            initialdir=initial_dir,
            initialfile=default_name
        )
        if filepath:
            self.status_var.set("Exporting audio...")
            self.root.update()

            try:
                from .chord_formatter import format_chord
                from .tts_generator import TTSGenerator, VOICE_SAMANTHA
                from .audio_mixer import AudioMixer

                # Convert chords to ChordEvents
                events = []
                for i, c in enumerate(self.chords):
                    end_time = self.chords[i + 1].start_time if i + 1 < len(self.chords) else self.player.duration
                    events.append(ChordEvent(
                        start_time=c.start_time,
                        end_time=end_time,
                        chord_name=c.chord_name
                    ))

                # Add notes as ChordEvents (using note text directly as "chord name")
                for note in self.notes:
                    events.append(ChordEvent(
                        start_time=note.start_time,
                        end_time=note.start_time + 1.0,  # Notes don't have end times
                        chord_name=note.text  # Use text directly - mixer will find it in tts_clips
                    ))

                # Sort all events by start time
                events.sort(key=lambda e: e.start_time)

                # Generate TTS clips for chords and notes (same voice)
                tts = TTSGenerator(cache_dir="cache", rate=175, voice_id=VOICE_SAMANTHA)
                tts_clips = {}

                for chord in self.chords:
                    spoken = format_chord(chord.chord_name)
                    if spoken and spoken not in tts_clips:
                        tts_clips[spoken] = tts.generate_clip(spoken)

                # Generate TTS clips for notes (slower rate for comprehension)
                note_tts = TTSGenerator(cache_dir="cache", rate=130, voice_id=VOICE_SAMANTHA)
                for note in self.notes:
                    if note.text not in tts_clips:
                        # Add pauses after periods
                        spoken_text = note.text.replace('. ', '... ').replace('.', '...')
                        tts_clips[note.text] = note_tts.generate_clip(spoken_text)

                # Load voice note clips
                from pydub import AudioSegment as PydubSegment
                for voice_note in self.voice_notes:
                    if Path(voice_note.file_path).exists():
                        tts_clips[voice_note.file_path] = PydubSegment.from_file(voice_note.file_path)
                        # Add voice note as event
                        events.append(ChordEvent(
                            start_time=voice_note.start_time,
                            end_time=voice_note.start_time + 1.0,
                            chord_name=voice_note.file_path
                        ))

                # Re-sort events after adding voice notes
                events.sort(key=lambda e: e.start_time)

                # Debug: print what we're exporting
                print(f"Exporting {len(self.chords)} chords, {len(self.notes)} notes, {len(self.voice_notes)} voice notes")
                print(f"TTS clips generated: {list(tts_clips.keys())}")
                print(f"Events: {[(e.start_time, e.chord_name) for e in events]}")

                # Mix audio (negative gap allows overlapping - don't skip anything)
                mixer = AudioMixer(min_gap_seconds=-10.0)
                original = mixer.load_audio(self.audio_file)

                # Check if pause-for-notes is enabled
                if self.pause_for_notes_var.get() and (self.notes or self.voice_notes):
                    # Build audio with pauses for notes and voice notes
                    from pydub import AudioSegment

                    # Combine notes and voice notes into one list with their clips
                    note_info = []  # [(original_time, duration_ms, note_clip)]

                    # Add text notes
                    for note in self.notes:
                        if note.text in tts_clips:
                            clip = tts_clips[note.text]
                            duration_ms = len(clip)
                            note_info.append((note.start_time, duration_ms, clip))

                    # Add voice notes
                    for voice_note in self.voice_notes:
                        if voice_note.file_path in tts_clips:
                            clip = tts_clips[voice_note.file_path]
                            duration_ms = len(clip)
                            note_info.append((voice_note.start_time, duration_ms, clip))

                    # Sort by time
                    note_info.sort(key=lambda x: x[0])

                    # Build the modified audio with pauses
                    modified_audio = AudioSegment.empty()
                    voiced_with_pauses = AudioSegment.empty()
                    last_pos_ms = 0
                    time_offset = 0  # Accumulated offset from inserted pauses

                    # Also build adjusted chord events
                    adjusted_events = []

                    for orig_time, duration_ms, note_clip in note_info:
                        pos_ms = int(orig_time * 1000)

                        # Add audio from last position to this note
                        if pos_ms > last_pos_ms:
                            segment = original[last_pos_ms:pos_ms]
                            modified_audio += segment
                            # Add silence for the voiced track (chords will be overlaid later)
                            voiced_with_pauses += AudioSegment.silent(duration=len(segment), frame_rate=original.frame_rate)

                        # Add the note (with silence in original, note in voiced)
                        # Include 500ms pause after note for better pacing
                        pause_after_ms = 500
                        total_note_duration = duration_ms + pause_after_ms
                        modified_audio += AudioSegment.silent(duration=total_note_duration, frame_rate=original.frame_rate)
                        voiced_with_pauses += note_clip
                        voiced_with_pauses += AudioSegment.silent(duration=pause_after_ms, frame_rate=original.frame_rate)

                        # Update offset for events after this note
                        time_offset += total_note_duration / 1000.0
                        last_pos_ms = pos_ms

                    # Add remaining audio
                    if last_pos_ms < len(original):
                        modified_audio += original[last_pos_ms:]
                        voiced_with_pauses += AudioSegment.silent(duration=len(original) - last_pos_ms, frame_rate=original.frame_rate)

                    # Adjust chord events based on accumulated offsets
                    for event in events:
                        # Skip note and voice note events (already handled)
                        if any(n.text == event.chord_name for n in self.notes):
                            continue
                        if any(v.file_path == event.chord_name for v in self.voice_notes):
                            continue

                        # Calculate time offset for this chord
                        offset = 0
                        for orig_time, duration_ms, _ in note_info:
                            if event.start_time > orig_time:
                                offset += duration_ms / 1000.0

                        adjusted_events.append(ChordEvent(
                            start_time=event.start_time + offset,
                            end_time=event.end_time + offset,
                            chord_name=event.chord_name
                        ))

                    # Create voiced track for chords only (notes already in voiced_with_pauses)
                    chord_voiced, _ = mixer.create_voiced_track(
                        duration_ms=len(modified_audio),
                        chord_events=adjusted_events,
                        tts_clips=tts_clips,
                        sample_rate=original.frame_rate,
                        channels=original.channels
                    )

                    # Ensure consistent audio format
                    voiced_with_pauses = voiced_with_pauses.set_frame_rate(original.frame_rate)
                    voiced_with_pauses = voiced_with_pauses.set_channels(original.channels)
                    chord_voiced = chord_voiced.set_frame_rate(original.frame_rate)
                    chord_voiced = chord_voiced.set_channels(original.channels)

                    # Combine note voices with chord voices
                    voiced_track = voiced_with_pauses.overlay(chord_voiced)

                    # Mix with modified original
                    original_adjusted = modified_audio + (- 3.0)  # Apply volume reduction
                    mixed = original_adjusted.overlay(voiced_track)

                    print(f"Pause-for-notes: inserted {len(note_info)} pauses")
                else:
                    # Standard export without pauses
                    voiced_track, voiced_items = mixer.create_voiced_track(
                        duration_ms=len(original),
                        chord_events=events,
                        tts_clips=tts_clips,
                        sample_rate=original.frame_rate,
                        channels=original.channels
                    )

                    # Debug: print what was actually voiced
                    print(f"Actually voiced: {voiced_items}")

                    # Export mixed (voiced + original)
                    mixed = mixer.mix_tracks(original, voiced_track, original_volume_db=-3.0)

                mixer.export(mixed, filepath)

                # Also export voice-only track
                filepath_path = Path(filepath)
                voice_only_name = f"voicedOnly{original_name}{filepath_path.suffix}"
                voice_only_path = filepath_path.parent / voice_only_name
                mixer.export(voiced_track, str(voice_only_path))

                # Auto-save chords/notes/voice notes/images JSON
                json_path = filepath_path.parent / f"{original_name}_chords.json"
                data = {
                    'audio_file': self.audio_file,
                    'duration': self.player.duration,
                    'chords': [c.to_dict() for c in self.chords],
                    'notes': [n.to_dict() for n in self.notes],
                    'voice_notes': [v.to_dict() for v in self.voice_notes],
                    'images': [i.to_dict() for i in self.images]
                }
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)

                self.status_var.set(f"Exported to {filepath_path.name}, {voice_only_name}, and {json_path.name}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
                self.status_var.set("Export failed")

    def run(self):
        """Run the editor."""
        self.root.mainloop()
        self.player.cleanup()


def main():
    """Entry point for the chord editor."""
    editor = ChordEditor()
    editor.run()


if __name__ == "__main__":
    main()
