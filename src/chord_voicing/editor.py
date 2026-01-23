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
        self.selected_chord: Optional[EditableChord] = None
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
        self.on_seek: Optional[Callable] = None
        self.on_zoom_changed: Optional[Callable] = None

        # Dragging state
        self._dragging = False
        self._drag_chord: Optional[EditableChord] = None
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

        # Draw chords
        chord_y = height // 2
        for chord in self.chords:
            x = self._time_to_x(chord.start_time)
            if 0 <= x <= width:
                color = self.selected_color if chord == self.selected_chord else self.chord_color
                # Draw chord marker
                self.create_oval(x - 8, chord_y - 8, x + 8, chord_y + 8, fill=color, outline='white')
                # Draw chord name
                self.create_text(x, chord_y - 20, text=chord.chord_name,
                               fill=self.text_color, font=('Arial', 10, 'bold'))

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
        chord_y = height // 2
        for chord in self.chords:
            cx = self._time_to_x(chord.start_time)
            if abs(cx - x) < 15 and abs(chord_y - y) < 15:
                return chord
        return None

    def _on_click(self, event):
        """Handle click event."""
        chord = self._find_chord_at(event.x, event.y)
        if chord:
            self.selected_chord = chord
            self._dragging = True
            self._drag_chord = chord
            self._drag_offset = self._time_to_x(chord.start_time) - event.x
            if self.on_chord_selected:
                self.on_chord_selected(chord)
        else:
            self.selected_chord = None
            # Seek to clicked position
            t = self._x_to_time(event.x)
            if 0 <= t <= self.duration and self.on_seek:
                self.on_seek(t)
        self.redraw()

    def _on_drag(self, event):
        """Handle drag event."""
        if self._dragging and self._drag_chord:
            new_time = self._x_to_time(event.x + self._drag_offset)
            new_time = max(0, min(new_time, self.duration))
            self._drag_chord.start_time = new_time
            self.chords.sort(key=lambda c: c.start_time)
            self.redraw()

    def _on_release(self, event):
        """Handle release event."""
        if self._dragging and self._drag_chord and self.on_chord_moved:
            self.on_chord_moved(self._drag_chord)
        self._dragging = False
        self._drag_chord = None

    def _on_double_click(self, event):
        """Handle double-click to edit chord name."""
        chord = self._find_chord_at(event.x, event.y)
        if chord:
            new_name = simpledialog.askstring("Edit Chord", "Chord name:",
                                             initialvalue=chord.chord_name)
            if new_name:
                chord.chord_name = new_name
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
        self.next_id = 0
        self.audio_file: Optional[str] = None
        self.clipboard: Optional[EditableChord] = None

        # Real-time chord preview
        self.preview_with_chords: bool = False
        self._tts_clips: dict = {}  # chord_name -> pygame.Sound
        self._announced_chords: set = set()  # chord IDs announced this playback
        self._last_position: float = -1  # Start at -1 so chords at time 0 get announced

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
        edit_menu.add_command(label="Delete Chord", command=self._delete_chord, accelerator="Delete")
        edit_menu.add_command(label="Copy Chord", command=self._copy_chord, accelerator="Cmd+C")
        edit_menu.add_command(label="Paste Chord", command=self._paste_chord, accelerator="Cmd+V")

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
        self.timeline.on_seek = self._on_seek
        self.timeline.on_zoom_changed = self._on_zoom_changed

        # Chord info frame
        info_frame = ttk.LabelFrame(main_frame, text="Selected Chord")
        info_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(info_frame, text="Name:").grid(row=0, column=0, padx=5, pady=5)
        self.chord_name_var = tk.StringVar()
        self.chord_name_entry = ttk.Entry(info_frame, textvariable=self.chord_name_var, width=15)
        self.chord_name_entry.grid(row=0, column=1, padx=5, pady=5)
        self.chord_name_entry.bind('<Return>', self._update_chord_name)

        ttk.Label(info_frame, text="Time:").grid(row=0, column=2, padx=5, pady=5)
        self.chord_time_var = tk.StringVar()
        self.chord_time_entry = ttk.Entry(info_frame, textvariable=self.chord_time_var, width=10)
        self.chord_time_entry.grid(row=0, column=3, padx=5, pady=5)
        self.chord_time_entry.bind('<Return>', self._update_chord_time)

        ttk.Label(info_frame, text="Spoken:").grid(row=0, column=4, padx=5, pady=5)
        self.chord_spoken_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.chord_spoken_var, width=20).grid(row=0, column=5, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Load an audio file to begin")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_bindings(self):
        """Setup keyboard bindings."""
        self.root.bind('<space>', lambda e: self._toggle_play())
        self.root.bind('<a>', lambda e: self._add_chord())
        self.root.bind('<Delete>', lambda e: self._delete_chord())
        self.root.bind('<BackSpace>', lambda e: self._delete_chord())
        self.root.bind('<Command-c>', lambda e: self._copy_chord())
        self.root.bind('<Command-v>', lambda e: self._paste_chord())
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

    def _toggle_play(self):
        """Toggle play/pause."""
        if self.player.is_playing():
            self.player.pause()
            self.play_btn.config(text="▶ Play")
        else:
            if self.player.loaded_file:
                # Always use play() with current position - unpause only works after pause
                self.player.play(self.player.get_position())
                self.play_btn.config(text="⏸ Pause")

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
        # Reset announcements - mark chords before seek position as announced
        self._announced_chords.clear()
        for chord in self.chords:
            if chord.start_time < time:
                self._announced_chords.add(chord.id)
        self._last_position = time

    def _on_chord_selected(self, chord: EditableChord):
        """Handle chord selection."""
        self.chord_name_var.set(chord.chord_name)
        self.chord_time_var.set(f"{chord.start_time:.2f}")
        spoken = format_chord(chord.chord_name) or "(unknown)"
        self.chord_spoken_var.set(spoken)

    def _on_chord_moved(self, chord: EditableChord):
        """Handle chord moved."""
        self.chord_time_var.set(f"{chord.start_time:.2f}")

    def _update_chord_name(self, event=None):
        """Update selected chord name."""
        if self.timeline.selected_chord:
            new_name = self.chord_name_var.get()
            self.timeline.selected_chord.chord_name = new_name
            spoken = format_chord(new_name) or "(unknown)"
            self.chord_spoken_var.set(spoken)
            self.timeline.redraw()
            # Load TTS for new chord if preview is enabled
            if self.preview_with_chords and new_name not in self._tts_clips:
                self._load_single_tts(new_name)

    def _update_chord_time(self, event=None):
        """Update selected chord time."""
        if self.timeline.selected_chord:
            try:
                new_time = float(self.chord_time_var.get())
                self.timeline.selected_chord.start_time = max(0, min(new_time, self.player.duration))
                self.timeline.chords.sort(key=lambda c: c.start_time)
                self.timeline.redraw()
            except ValueError:
                pass

    def _add_chord(self):
        """Add a new chord at playhead position."""
        pos = self.player.get_position()
        new_chord = EditableChord(start_time=pos, chord_name="C", id=self._get_next_id())
        self.chords.append(new_chord)
        self.chords.sort(key=lambda c: c.start_time)
        self.timeline.set_chords(self.chords)
        self.timeline.selected_chord = new_chord
        self._on_chord_selected(new_chord)
        self.status_var.set(f"Added chord at {pos:.2f}s")
        # Load TTS for new chord if preview is enabled and not already loaded
        if self.preview_with_chords and "C" not in self._tts_clips:
            self._load_single_tts("C")

    def _delete_chord(self):
        """Delete selected chord."""
        if self.timeline.selected_chord:
            self.chords.remove(self.timeline.selected_chord)
            self.timeline.selected_chord = None
            self.timeline.set_chords(self.chords)
            self.chord_name_var.set("")
            self.chord_time_var.set("")
            self.chord_spoken_var.set("")
            self.status_var.set("Chord deleted")

    def _copy_chord(self):
        """Copy selected chord."""
        if self.timeline.selected_chord:
            self.clipboard = EditableChord(
                start_time=self.timeline.selected_chord.start_time,
                chord_name=self.timeline.selected_chord.chord_name,
                id=0
            )
            self.status_var.set(f"Copied: {self.clipboard.chord_name}")

    def _paste_chord(self):
        """Paste chord at playhead position."""
        if self.clipboard:
            pos = self.player.get_position()
            new_chord = EditableChord(
                start_time=pos,
                chord_name=self.clipboard.chord_name,
                id=self._get_next_id()
            )
            self.chords.append(new_chord)
            self.chords.sort(key=lambda c: c.start_time)
            self.timeline.set_chords(self.chords)
            self.status_var.set(f"Pasted: {new_chord.chord_name} at {pos:.2f}s")

    def _nudge(self, delta: float):
        """Nudge selected chord by delta seconds."""
        if self.timeline.selected_chord:
            new_time = self.timeline.selected_chord.start_time + delta
            new_time = max(0, min(new_time, self.player.duration))
            self.timeline.selected_chord.start_time = new_time
            self.chord_time_var.set(f"{new_time:.2f}")
            self.timeline.chords.sort(key=lambda c: c.start_time)
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
        """Load TTS clips for all current chords as pygame Sound objects."""
        if not self.chords:
            return

        self.status_var.set("Loading chord sounds...")
        self.root.update()

        try:
            from .chord_formatter import format_chord
            from .tts_generator import TTSGenerator
            import tempfile
            import os

            # Get unique chord names (original names, not spoken)
            unique_chord_names = set(c.chord_name for c in self.chords)

            # Generate TTS clips keyed by ORIGINAL chord name
            tts = TTSGenerator(cache_dir="cache", rate=175)
            self._tts_clips = {}
            temp_dir = tempfile.mkdtemp()

            for chord_name in unique_chord_names:
                spoken = format_chord(chord_name)
                if not spoken:
                    continue

                # Generate clip for this chord
                clip = tts.generate_clip(spoken)

                # Export to temporary WAV file
                safe_name = chord_name.replace('/', '_').replace('#', 'sharp')
                temp_path = os.path.join(temp_dir, f"{safe_name}.wav")
                clip.export(temp_path, format="wav")

                # Load as pygame Sound, keyed by ORIGINAL chord name
                self._tts_clips[chord_name] = pygame.mixer.Sound(temp_path)

            self.status_var.set(f"Loaded {len(self._tts_clips)} chord sounds")
        except Exception as e:
            self.status_var.set(f"Failed to load sounds: {e}")
            self.preview_var.set(False)
            self.preview_with_chords = False

    def _check_chord_announcements(self, current_pos: float):
        """Check if we need to announce any chords at current position."""
        # Look for chords that we've just crossed
        for chord in self.chords:
            # Check if we just crossed this chord's start time
            if (self._last_position < chord.start_time <= current_pos and
                chord.id not in self._announced_chords):
                self._announce_chord(chord)
                self._announced_chords.add(chord.id)

    def _announce_chord(self, chord: EditableChord):
        """Play the TTS clip for a chord."""
        if chord.chord_name in self._tts_clips:
            self._tts_clips[chord.chord_name].play()

    def _load_single_tts(self, chord_name: str):
        """Load TTS clip for a single chord name."""
        try:
            from .chord_formatter import format_chord
            from .tts_generator import TTSGenerator
            import tempfile
            import os

            spoken = format_chord(chord_name)
            if not spoken:
                return

            tts = TTSGenerator(cache_dir="cache", rate=175)
            clip = tts.generate_clip(spoken)

            # Export to temp file and load as pygame Sound
            temp_dir = tempfile.mkdtemp()
            safe_name = chord_name.replace('/', '_').replace('#', 'sharp')
            temp_path = os.path.join(temp_dir, f"{safe_name}.wav")
            clip.export(temp_path, format="wav")
            self._tts_clips[chord_name] = pygame.mixer.Sound(temp_path)
        except Exception as e:
            print(f"Failed to load TTS for {chord_name}: {e}")

    def _reset_announcements(self):
        """Reset announced chords (called on seek/stop)."""
        self._announced_chords.clear()

    def _save_chords(self):
        """Save chords to JSON file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            data = {
                'audio_file': self.audio_file,
                'duration': self.player.duration,
                'chords': [c.to_dict() for c in self.chords]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.status_var.set(f"Saved to {Path(filepath).name}")

    def _load_chords(self):
        """Load chords from JSON file."""
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

                self.timeline.set_chords(self.chords)
                self.status_var.set(f"Loaded {len(self.chords)} chords from {Path(filepath).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load chords: {e}")

    def _export_audio(self):
        """Export voiced audio."""
        if not self.audio_file or not self.chords:
            messagebox.showwarning("Warning", "Load audio and add chords first")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")]
        )
        if filepath:
            self.status_var.set("Exporting audio...")
            self.root.update()

            try:
                from .chord_formatter import get_unique_chords
                from .tts_generator import TTSGenerator
                from .audio_mixer import AudioMixer

                # Convert to ChordEvents
                events = []
                for i, c in enumerate(self.chords):
                    end_time = self.chords[i + 1].start_time if i + 1 < len(self.chords) else self.player.duration
                    events.append(ChordEvent(
                        start_time=c.start_time,
                        end_time=end_time,
                        chord_name=c.chord_name
                    ))

                # Generate TTS
                chord_names = [c.chord_name for c in self.chords]
                unique_spoken = get_unique_chords(chord_names)

                tts = TTSGenerator(cache_dir="cache", rate=175)
                tts_clips = tts.generate_clips(unique_spoken)

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
