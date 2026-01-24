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

    def cleanup(self):
        pygame.mixer.quit()


class ChordTimeline(tk.Canvas):
    """Timeline canvas showing chords."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.duration: float = 60  # Default duration
        self.chords: List[EditableChord] = []
        self.notes: List[EditableNote] = []
        self.selected_chord: Optional[EditableChord] = None
        self.selected_note: Optional[EditableNote] = None
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
        self.on_note_selected: Optional[Callable] = None
        self.on_note_moved: Optional[Callable] = None
        self.on_seek: Optional[Callable] = None
        self.on_zoom_changed: Optional[Callable] = None

        # Dragging state
        self._dragging = False
        self._drag_chord: Optional[EditableChord] = None
        self._drag_note: Optional[EditableNote] = None
        self._drag_offset: float = 0

        # Panning state
        self._panning = False
        self._pan_start_x: float = 0
        self._pan_start_offset: float = 0

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

        # Colors
        self.bg_color = '#2b2b2b'
        self.chord_color = '#4a9eff'
        self.note_color = '#ffa500'  # Orange for notes
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

        # Draw notes (lower half)
        note_y = (height * 2) // 3
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
        chord_y = height // 3
        for chord in self.chords:
            cx = self._time_to_x(chord.start_time)
            if abs(cx - x) < 15 and abs(chord_y - y) < 15:
                return chord
        return None

    def _find_note_at(self, x: float, y: float) -> Optional[EditableNote]:
        """Find note at given coordinates."""
        height = self.winfo_height() or 100
        note_y = (height * 2) // 3
        for note in self.notes:
            nx = self._time_to_x(note.start_time)
            if abs(nx - x) < 15 and abs(note_y - y) < 15:
                return note
        return None

    def _on_click(self, event):
        """Handle click event."""
        # Check for chord first
        chord = self._find_chord_at(event.x, event.y)
        if chord:
            self.selected_chord = chord
            self.selected_note = None
            self._dragging = True
            self._drag_chord = chord
            self._drag_note = None
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
            self._dragging = True
            self._drag_note = note
            self._drag_chord = None
            self._drag_offset = self._time_to_x(note.start_time) - event.x
            if self.on_note_selected:
                self.on_note_selected(note)
            self.redraw()
            return

        # Nothing clicked - clear selection and seek
        self.selected_chord = None
        self.selected_note = None
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
            self.redraw()

    def _on_release(self, event):
        """Handle release event."""
        if self._dragging:
            if self._drag_chord and self.on_chord_moved:
                self.on_chord_moved(self._drag_chord)
            elif self._drag_note and self.on_note_moved:
                self.on_note_moved(self._drag_note)
        self._dragging = False
        self._drag_chord = None
        self._drag_note = None

    def _on_double_click(self, event):
        """Handle double-click to edit chord or note."""
        chord = self._find_chord_at(event.x, event.y)
        if chord:
            new_name = simpledialog.askstring("Edit Chord", "Chord name:",
                                             initialvalue=chord.chord_name)
            if new_name:
                chord.chord_name = new_name
                self.redraw()
            return

        note = self._find_note_at(event.x, event.y)
        if note:
            new_text = simpledialog.askstring("Edit Note", "Note text:",
                                             initialvalue=note.text)
            if new_text:
                note.text = new_text
                self.redraw()

    def _on_resize(self, event):
        """Handle resize event."""
        self._update_zoom()
        self.redraw()


class ChordEditor:
    """Main chord editor application."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chord Editor")
        self.root.geometry("1200x400")

        self.player = AudioPlayer()
        self.chords: List[EditableChord] = []
        self.notes: List[EditableNote] = []
        self.next_id = 0
        self.audio_file: Optional[str] = None
        self.clipboard_chord: Optional[EditableChord] = None
        self.clipboard_note: Optional[EditableNote] = None

        # Real-time chord/note preview
        self.preview_with_chords: bool = False
        self._tts_clips: dict = {}  # chord_name -> pygame.Sound
        self._note_clips: dict = {}  # note_text -> pygame.Sound
        self._announced_chords: set = set()  # chord IDs announced this playback
        self._announced_notes: set = set()  # note IDs announced this playback
        self._last_position: float = -1  # Start at -1 so items at time 0 get announced

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
        file_menu.add_command(label="Save Chords...", command=self._save_chords, accelerator="Cmd+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export Voiced Audio...", command=self._export_audio)
        file_menu.add_separator()
        file_menu.add_command(label="Detect Chords", command=self._detect_chords)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Add Chord", command=self._add_chord, accelerator="A")
        edit_menu.add_command(label="Add Note", command=self._add_note, accelerator="N")
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

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.play_btn = ttk.Button(control_frame, text="▶ Play", command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="⏹ Stop", command=self._stop).pack(side=tk.LEFT, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.preview_var = tk.BooleanVar(value=False)
        self.preview_check = ttk.Checkbutton(
            control_frame, text="🔊 Hear Chords",
            variable=self.preview_var,
            command=self._toggle_preview_mode
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
        ttk.Button(control_frame, text="-", width=3, command=self._zoom_out).pack(side=tk.LEFT)
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(control_frame, textvariable=self.zoom_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="+", width=3, command=self._zoom_in).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Fit", width=4, command=self._zoom_fit).pack(side=tk.LEFT, padx=(5, 0))

        # Timeline frame with scrollbar
        timeline_frame = ttk.Frame(main_frame)
        timeline_frame.pack(fill=tk.BOTH, expand=True)

        # Horizontal scrollbar
        self.h_scrollbar = ttk.Scrollbar(timeline_frame, orient=tk.HORIZONTAL, command=self._on_scroll)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Timeline
        self.timeline = ChordTimeline(timeline_frame, height=150)
        self.timeline.pack(fill=tk.BOTH, expand=True)
        self.timeline.on_chord_selected = self._on_chord_selected
        self.timeline.on_chord_moved = self._on_chord_moved
        self.timeline.on_note_selected = self._on_note_selected
        self.timeline.on_note_moved = self._on_note_moved
        self.timeline.on_seek = self._on_seek
        self.timeline.on_zoom_changed = self._on_zoom_changed

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

        # Status bar
        self.status_var = tk.StringVar(value="Load an audio file to begin")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_bindings(self):
        """Setup keyboard bindings."""
        # Spacebar toggles play/pause globally
        self.root.bind_all('<space>', self._on_space)
        self.root.bind('<a>', lambda e: self._add_chord())
        self.root.bind('<n>', lambda e: self._add_note())
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

                # Clear existing chords and notes
                self.chords = []
                self.notes = []
                self.timeline.set_chords(self.chords)
                self.timeline.set_notes(self.notes)
                self._clear_selection_info()

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
            self.status_var.set(f"Detected {len(self.chords)} chords")
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
        if self.player.is_playing():
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

    def _on_seek(self, time: float):
        """Handle seek from timeline."""
        self.player.seek(time)
        self.timeline.set_playhead(time)
        self._update_position_display(time)
        # Reset announcements - mark items before seek position as announced
        self._announced_chords.clear()
        self._announced_notes.clear()
        for chord in self.chords:
            if chord.start_time < time:
                self._announced_chords.add(chord.id)
        for note in self.notes:
            if note.start_time < time:
                self._announced_notes.add(note.id)
        self._last_position = time

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

    def _on_note_selected(self, note: EditableNote):
        """Handle note selection."""
        self.selection_type_var.set("Note")
        self.item_text_var.set(note.text)
        self.item_time_var.set(f"{note.start_time:.2f}")
        self.item_spoken_var.set("(as typed)")

    def _on_note_moved(self, note: EditableNote):
        """Handle note moved."""
        self.item_time_var.set(f"{note.start_time:.2f}")

    def _update_selected_text(self, event=None):
        """Update text of selected chord or note."""
        new_text = self.item_text_var.get()
        if self.timeline.selected_chord:
            self.timeline.selected_chord.chord_name = new_text
            spoken = format_chord(new_text) or "(unknown)"
            self.item_spoken_var.set(spoken)
            self.timeline.redraw()
            # Load TTS for new chord if preview is enabled
            if self.preview_with_chords and new_text not in self._tts_clips:
                self._load_single_tts(new_text)
        elif self.timeline.selected_note:
            self.timeline.selected_note.text = new_text
            self.timeline.redraw()
            # Load TTS for new note if preview is enabled
            if self.preview_with_chords and new_text not in self._note_clips:
                self._load_single_note_tts(new_text)

    def _update_selected_time(self, event=None):
        """Update time of selected chord or note."""
        try:
            new_time = float(self.item_time_var.get())
            new_time = max(0, min(new_time, self.player.duration))
            if self.timeline.selected_chord:
                self.timeline.selected_chord.start_time = new_time
                self.timeline.chords.sort(key=lambda c: c.start_time)
            elif self.timeline.selected_note:
                self.timeline.selected_note.start_time = new_time
                self.timeline.notes.sort(key=lambda n: n.start_time)
            self.timeline.redraw()
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
        self.status_var.set(f"Added chord at {pos:.2f}s")
        # Load TTS for new chord if preview is enabled and not already loaded
        if self.preview_with_chords and "C" not in self._tts_clips:
            self._load_single_tts("C")

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
        # Load TTS for new note if preview is enabled
        if self.preview_with_chords and note_text not in self._note_clips:
            self._load_single_note_tts(note_text)

    def _delete_selected(self):
        """Delete selected chord or note."""
        if self.timeline.selected_chord:
            self.chords.remove(self.timeline.selected_chord)
            self.timeline.selected_chord = None
            self.timeline.set_chords(self.chords)
            self._clear_selection_info()
            self.status_var.set("Chord deleted")
        elif self.timeline.selected_note:
            self.notes.remove(self.timeline.selected_note)
            self.timeline.selected_note = None
            self.timeline.set_notes(self.notes)
            self._clear_selection_info()
            self.status_var.set("Note deleted")

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
            self.status_var.set(f"Pasted chord: {new_chord.chord_name} at {pos:.2f}s")
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

    def _nudge(self, delta: float):
        """Nudge selected chord or note by delta seconds."""
        if self.timeline.selected_chord:
            new_time = self.timeline.selected_chord.start_time + delta
            new_time = max(0, min(new_time, self.player.duration))
            self.timeline.selected_chord.start_time = new_time
            self.item_time_var.set(f"{new_time:.2f}")
            self.timeline.chords.sort(key=lambda c: c.start_time)
            self.timeline.redraw()
        elif self.timeline.selected_note:
            new_time = self.timeline.selected_note.start_time + delta
            new_time = max(0, min(new_time, self.player.duration))
            self.timeline.selected_note.start_time = new_time
            self.item_time_var.set(f"{new_time:.2f}")
            self.timeline.notes.sort(key=lambda n: n.start_time)
            self.timeline.redraw()

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
        if not self.chords and not self.notes:
            return

        self.status_var.set("Loading sounds...")
        self.root.update()

        try:
            from .chord_formatter import format_chord
            from .tts_generator import TTSGenerator, VOICE_SAMANTHA, VOICE_MOIRA
            import tempfile
            import os

            # Use different voices for chords vs notes
            chord_tts = TTSGenerator(cache_dir="cache", rate=175, voice_id=VOICE_SAMANTHA)
            note_tts = TTSGenerator(cache_dir="cache", rate=175, voice_id=VOICE_MOIRA)
            temp_dir = tempfile.mkdtemp()

            # Load chord TTS clips (Samantha voice - female)
            unique_chord_names = set(c.chord_name for c in self.chords)
            self._tts_clips = {}

            for chord_name in unique_chord_names:
                spoken = format_chord(chord_name)
                if not spoken:
                    continue
                clip = chord_tts.generate_clip(spoken)
                safe_name = chord_name.replace('/', '_').replace('#', 'sharp')
                temp_path = os.path.join(temp_dir, f"chord_{safe_name}.wav")
                clip.export(temp_path, format="wav")
                self._tts_clips[chord_name] = pygame.mixer.Sound(temp_path)

            # Load note TTS clips (Daniel voice - male, British)
            unique_note_texts = set(n.text for n in self.notes)
            self._note_clips = {}

            for note_text in unique_note_texts:
                clip = note_tts.generate_clip(note_text)
                safe_name = "".join(c if c.isalnum() else '_' for c in note_text[:20])
                temp_path = os.path.join(temp_dir, f"note_{safe_name}.wav")
                clip.export(temp_path, format="wav")
                self._note_clips[note_text] = pygame.mixer.Sound(temp_path)

            total = len(self._tts_clips) + len(self._note_clips)
            self.status_var.set(f"Loaded {total} sounds ({len(self._tts_clips)} chords, {len(self._note_clips)} notes)")
        except Exception as e:
            self.status_var.set(f"Failed to load sounds: {e}")
            self.preview_var.set(False)
            self.preview_with_chords = False

    def _check_chord_announcements(self, current_pos: float):
        """Check if we need to announce any chords or notes at current position."""
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

    def _announce_chord(self, chord: EditableChord):
        """Play the TTS clip for a chord."""
        if chord.chord_name in self._tts_clips:
            self._tts_clips[chord.chord_name].play()

    def _announce_note(self, note: EditableNote):
        """Play the TTS clip for a note."""
        if note.text in self._note_clips:
            self._note_clips[note.text].play()

    def _load_single_tts(self, chord_name: str):
        """Load TTS clip for a single chord name (Samantha voice)."""
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
        """Load TTS clip for a single note text (Daniel voice)."""
        try:
            from .tts_generator import TTSGenerator, VOICE_MOIRA
            import tempfile
            import os

            tts = TTSGenerator(cache_dir="cache", rate=175, voice_id=VOICE_MOIRA)
            clip = tts.generate_clip(note_text)

            temp_dir = tempfile.mkdtemp()
            safe_name = "".join(c if c.isalnum() else '_' for c in note_text[:20])
            temp_path = os.path.join(temp_dir, f"{safe_name}.wav")
            clip.export(temp_path, format="wav")
            self._note_clips[note_text] = pygame.mixer.Sound(temp_path)
        except Exception as e:
            print(f"Failed to load TTS for note {note_text}: {e}")

    def _reset_announcements(self):
        """Reset announced chords and notes (called on seek/stop)."""
        self._announced_chords.clear()
        self._announced_notes.clear()

    def _save_chords(self):
        """Save chords and notes to JSON file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            data = {
                'audio_file': self.audio_file,
                'duration': self.player.duration,
                'chords': [c.to_dict() for c in self.chords],
                'notes': [n.to_dict() for n in self.notes]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.status_var.set(f"Saved to {Path(filepath).name}")

    def _load_chords(self):
        """Load chords and notes from JSON file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                self.chords = []
                for c in data.get('chords', []):
                    self.chords.append(EditableChord.from_dict(c, self._get_next_id()))

                self.notes = []
                for n in data.get('notes', []):
                    self.notes.append(EditableNote.from_dict(n, self._get_next_id()))

                self.timeline.set_chords(self.chords)
                self.timeline.set_notes(self.notes)
                self.status_var.set(f"Loaded {len(self.chords)} chords, {len(self.notes)} notes from {Path(filepath).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")

    def _export_audio(self):
        """Export voiced audio."""
        if not self.audio_file or (not self.chords and not self.notes):
            messagebox.showwarning("Warning", "Load audio and add chords/notes first")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")]
        )
        if filepath:
            self.status_var.set("Exporting audio...")
            self.root.update()

            try:
                from .chord_formatter import format_chord
                from .tts_generator import TTSGenerator, VOICE_SAMANTHA, VOICE_MOIRA
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

                # Generate TTS clips for chords (Samantha voice)
                # Key must be the SPOKEN text since that's what audio_mixer looks up
                chord_tts = TTSGenerator(cache_dir="cache", rate=175, voice_id=VOICE_SAMANTHA)
                note_tts = TTSGenerator(cache_dir="cache", rate=175, voice_id=VOICE_MOIRA)
                tts_clips = {}

                for chord in self.chords:
                    spoken = format_chord(chord.chord_name)
                    if spoken and spoken not in tts_clips:
                        tts_clips[spoken] = chord_tts.generate_clip(spoken)

                # Generate TTS clips for notes (Daniel voice)
                # Notes use the raw text as both the chord_name and spoken form
                for note in self.notes:
                    # The note text IS the spoken text
                    if note.text not in tts_clips:
                        tts_clips[note.text] = note_tts.generate_clip(note.text)

                # Mix audio
                mixer = AudioMixer()
                original = mixer.load_audio(self.audio_file)

                voiced_track, _ = mixer.create_voiced_track(
                    duration_ms=len(original),
                    chord_events=events,
                    tts_clips=tts_clips,
                    sample_rate=original.frame_rate,
                    channels=original.channels
                )

                mixed = mixer.mix_tracks(original, voiced_track, original_volume_db=-3.0)
                mixer.export(mixed, filepath)

                self.status_var.set(f"Exported to {Path(filepath).name}")
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
