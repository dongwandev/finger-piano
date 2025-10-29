#!/usr/bin/env python3
"""
Finger Piano - A virtual instrument that plays piano sounds based on finger movements.
Uses MediaPipe for hand landmark detection and pygame for audio playback.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
from typing import Optional, Tuple, List
import sys
from config import Config
from screens import LobbyScreen, SettingsScreen, PlayScreen


class FingerPiano:
    """Main class for the finger piano virtual instrument."""
    
    # Piano key mapping for fingers (chords instead of single notes)
    # Index: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    NOTES = ['C Major', 'G Major', 'A Minor', 'F Major', 'D Major']
    
    # Chord definitions (note names for each chord)
    CHORDS = {
        'C Major': ['C4', 'E4', 'G4'],       # C, E, G
        'G Major': ['G4', 'B4', 'D5'],       # G, B, D
        'A Minor': ['A4', 'C5', 'E5'],       # A, C, E
        'F Major': ['F4', 'A4', 'C5'],       # F, A, C
        'D Major': ['D4', 'F#4', 'A4'],      # D, F#, A
        'E Minor': ['E4', 'G4', 'B4'],       # Em
        'E Major': ['E4', 'G#4', 'B4'],      # E
        'A Sus4': ['A4', 'D5', 'E5']         # Asus4
    }
    
    # Chord presets - each preset defines which chords are assigned to each finger
    # Format: preset_name -> list of chord names (empty string for unassigned fingers)
    CHORD_PRESETS = {
        'default': ['C Major', 'G Major', 'A Minor', 'F Major', 'D Major'],
        'preset1': ['G Major', 'D Major', 'E Minor', 'C Major', ''],
        'preset2': ['A Minor', 'C Major', 'G Major', 'D Major', ''],
        'preset3': ['E Major', 'A Minor', 'A Sus4', '', '']
    }
    
    # MIDI note frequencies (expanded to include all notes needed for chords)
    NOTE_FREQUENCIES = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'B4': 493.88, 
        'C5': 523.25, 'D5': 587.33, 'E5': 659.25
    }
    
    # Colors for visualization
    COLORS = {
        'active': (0, 255, 0),      # Green for active finger
        'inactive': (255, 0, 0),    # Red for inactive finger
        'landmark': (255, 255, 0),  # Yellow for landmarks
        'connection': (0, 255, 255) # Cyan for connections
    }
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the finger piano.
        
        Args:
            config: Configuration object (creates default if None)
        """
        # Load or create configuration
        self.config = config if config is not None else Config()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5)
        )
        
        # Initialize camera
        camera_id = self.config.get('camera_id', 0)
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize pygame for audio
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Generate piano sounds
        self.sounds = self._generate_piano_sounds()
        
        # Set current chord preset
        preset_name = self.config.get('chord_preset', 'default')
        self.current_preset = self.CHORD_PRESETS.get(preset_name, self.CHORD_PRESETS['default'])
        
        # Track finger states (whether each finger is currently playing)
        self.finger_states = [False] * 5
        self.finger_extensions = [1.0] * 5  # Track finger extension (distance from tip to MCP)
        self.finger_base_extensions = [0.0] * 5  # Track maximum (fully extended) finger extension for each finger
        self.finger_flexion_percent = [0.0] * 5  # Track flexion percentage (0-100%)
        
        # Threshold for triggering notes (relative change in finger extension)
        # When finger bends, extension decreases
        self.trigger_threshold = self.config.get('trigger_threshold', 0.15)
        
        # GUI state
        self.current_screen = None
        self.screens = {}
        
    def _generate_piano_sounds(self) -> dict:
        """Generate synthesized piano sounds for each chord.
        
        Returns:
            Dictionary mapping chord names to pygame Sound objects
        """
        sounds = {}
        sample_rate = 22050
        duration = 0.5  # seconds
        
        # First, generate individual note waves
        note_waves = {}
        for note, frequency in self.NOTE_FREQUENCIES.items():
            # Generate a simple sine wave with ADSR envelope
            samples = int(sample_rate * duration)
            wave = np.zeros(samples)
            
            for i in range(samples):
                t = i / sample_rate
                
                # ADSR envelope
                if t < 0.05:  # Attack
                    envelope = t / 0.05
                elif t < 0.1:  # Decay
                    envelope = 1.0 - 0.3 * (t - 0.05) / 0.05
                elif t < 0.4:  # Sustain
                    envelope = 0.7
                else:  # Release
                    envelope = 0.7 * (1.0 - (t - 0.4) / 0.1)
                
                # Generate sine wave with envelope
                wave[i] = envelope * np.sin(2 * np.pi * frequency * t)
                
                # Add harmonics for richer sound
                wave[i] += 0.3 * envelope * np.sin(4 * np.pi * frequency * t)
                wave[i] += 0.1 * envelope * np.sin(6 * np.pi * frequency * t)
            
            note_waves[note] = wave
        
        # Now combine notes into chords
        for chord_name, chord_notes in self.CHORDS.items():
            samples = int(sample_rate * duration)
            chord_wave = np.zeros(samples)
            
            # Add all notes in the chord together
            for note in chord_notes:
                if note not in note_waves:
                    raise ValueError(f"Note '{note}' in chord '{chord_name}' not found in NOTE_FREQUENCIES")
                chord_wave += note_waves[note] * 0.7  # Scale to prevent clipping while maintaining volume
            
            # Normalize and convert to 16-bit integers
            if np.max(np.abs(chord_wave)) > 0:
                chord_wave = chord_wave / np.max(np.abs(chord_wave))
            chord_wave = (chord_wave * 32767).astype(np.int16)
            
            # Create stereo sound
            stereo_wave = np.column_stack((chord_wave, chord_wave))
            
            # Create pygame Sound object
            sounds[chord_name] = pygame.sndarray.make_sound(stereo_wave)
        
        return sounds
    
    def _get_finger_data(self, hand_landmarks) -> List[Tuple[int, float]]:
        """Extract finger extension data from hand landmarks.
        
        Calculates the distance between fingertip and MCP (knuckle) for each finger.
        When a finger bends/curls, this distance decreases.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            List of tuples (finger_id, extension_distance) for each finger
        """
        # MediaPipe hand landmark indices
        # Format: (tip_id, mcp_id) for each finger
        finger_landmarks = [
            (4, 2),   # Thumb: TIP to MCP
            (8, 5),   # Index: TIP to MCP
            (12, 9),  # Middle: TIP to MCP
            (16, 13), # Ring: TIP to MCP
            (20, 17)  # Pinky: TIP to MCP
        ]
        
        finger_data = []
        for i, (tip_id, mcp_id) in enumerate(finger_landmarks):
            tip = hand_landmarks.landmark[tip_id]
            mcp = hand_landmarks.landmark[mcp_id]
            
            # Calculate Euclidean distance between tip and MCP
            # This represents how extended/straight the finger is
            dx = tip.x - mcp.x
            dy = tip.y - mcp.y
            dz = tip.z - mcp.z
            extension = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            finger_data.append((i, extension))
        
        return finger_data
    
    def _detect_finger_bending(self, finger_id: int, current_extension: float) -> bool:
        """Detect if a finger has bent/curled (like pressing a piano key).
        
        When a finger bends, the distance between tip and MCP decreases.
        
        Args:
            finger_id: Index of the finger (0-4)
            current_extension: Current extension distance (tip to MCP)
            
        Returns:
            True if the finger has bent significantly
        """
        previous_extension = self.finger_extensions[finger_id]
        
        # Calculate relative change in extension
        # Negative change means finger is bending (getting shorter)
        if previous_extension > 0:
            relative_change = (current_extension - previous_extension) / previous_extension
        else:
            relative_change = 0
        
        # Trigger if finger bent (extension decreased) and not already playing
        # Negative relative_change means bending
        if relative_change < -self.trigger_threshold and not self.finger_states[finger_id]:
            return True
        
        return False
    
    def _play_note(self, finger_id: int):
        """Play the chord associated with a finger.
        
        Args:
            finger_id: Index of the finger (0-4)
        """
        if finger_id < len(self.current_preset):
            chord_name = self.current_preset[finger_id]
            # Skip if finger is unassigned (empty string)
            if chord_name and chord_name in self.sounds:
                self.sounds[chord_name].play()
                self.finger_states[finger_id] = True
    
    def _reset_finger_state(self, finger_id: int, current_extension: float) -> bool:
        """Reset finger state when it extends back (straightens).
        
        Args:
            finger_id: Index of the finger (0-4)
            current_extension: Current extension distance (tip to MCP)
            
        Returns:
            True if the finger state was reset
        """
        previous_extension = self.finger_extensions[finger_id]
        
        # Calculate relative change in extension
        if previous_extension > 0:
            relative_change = (current_extension - previous_extension) / previous_extension
        else:
            relative_change = 0
        
        # Reset if finger extended (straightened) significantly
        # Positive relative_change means extending
        if relative_change > self.trigger_threshold:
            self.finger_states[finger_id] = False
            return True
        
        return False
    
    def _update_finger_extension(self, finger_id: int, current_extension: float):
        """Update the tracked extension of a finger and calculate flexion percentage.
        
        Args:
            finger_id: Index of the finger (0-4)
            current_extension: Current extension distance (tip to MCP)
        """
        # Update base extension if this is the most extended we've seen this finger
        # This automatically calibrates to each user's hand size and range of motion
        # Note: Users should extend their fingers fully at least once for accurate readings
        if current_extension > self.finger_base_extensions[finger_id]:
            self.finger_base_extensions[finger_id] = current_extension
        
        self.finger_extensions[finger_id] = current_extension
        
        # Calculate flexion percentage
        # 0% = fully extended (max distance), 100% = fully flexed (min distance)
        base = self.finger_base_extensions[finger_id]
        if base > 0:
            # Flexion % = (base - current) / base * 100
            # When current == base (extended), flexion = 0%
            # When current < base (bent), flexion > 0%
            flexion_percent = ((base - current_extension) / base) * 100
            # Clamp to 0-100 range
            self.finger_flexion_percent[finger_id] = max(0.0, min(100.0, flexion_percent))
        else:
            self.finger_flexion_percent[finger_id] = 0.0
    
    def _draw_ui(self, image, hand_landmarks, image_width: int, image_height: int):
        """Draw UI elements on the image.
        
        Args:
            image: The image to draw on
            hand_landmarks: MediaPipe hand landmarks
            image_width: Width of the image
            image_height: Height of the image
        """
        # Draw hand landmarks
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw finger state indicators
        y_offset = 30
        for i in range(5):
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            note = self.NOTES[i] if i < len(self.NOTES) else 'N/A'
            color = self.COLORS['active'] if self.finger_states[i] else self.COLORS['inactive']
            
            text = f"{finger_names[i]}: {note}"
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)
            y_offset += 30
        
        # Draw instructions
        cv2.putText(image, "Bend fingers to play notes", 
                   (10, image_height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press 'q' to quit", 
                   (10, image_height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
    
    def run(self):
        """Main loop for the finger piano application with GUI."""
        print("Finger Piano started.")
        print("Navigate using arrow keys or W/A/S/D")
        
        # Initialize screens
        self.screens = {
            'lobby': LobbyScreen(self.config),
            'settings': SettingsScreen(self.config),
            'play': PlayScreen(self.config, self)
        }
        
        self.current_screen = self.screens['lobby']
        self.current_screen.on_enter()
        
        while True:
            # Handle different screens
            if self.current_screen.name == 'play':
                # For play screen, read from camera
                success, frame = self.cap.read()
                if success:
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process hand landmarks
                    results = self.hands.process(rgb_frame)
                    
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        # Get finger extension data
                        finger_data = self._get_finger_data(hand_landmarks)
                        
                        # Check each finger for bending
                        for finger_id, extension in finger_data:
                            # Reset state if finger extended (check before updating)
                            self._reset_finger_state(finger_id, extension)
                            
                            # Detect finger bending and play note
                            if self._detect_finger_bending(finger_id, extension):
                                self._play_note(finger_id)
                            
                            # Update extension after all checks
                            self._update_finger_extension(finger_id, extension)
                        
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    else:
                        # No hand detected - reset all states
                        self.finger_states = [False] * 5
                    
                    # Render play screen with camera frame
                    rendered_frame = self.current_screen.render(frame)
                else:
                    # No camera frame available
                    rendered_frame = self.current_screen.render(None)
            else:
                # For other screens, render without camera
                rendered_frame = self.current_screen.render()
            
            # Display the rendered frame
            cv2.imshow('Finger Piano', rendered_frame)
            
            # Handle events
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Key was pressed
                next_screen_name = self.current_screen.handle_event(key)
                
                if next_screen_name == 'quit':
                    break
                elif next_screen_name and next_screen_name in self.screens:
                    # Navigate to new screen
                    self.current_screen.on_exit()
                    self.current_screen = self.screens[next_screen_name]
                    self.current_screen.on_enter()
                    
                    # Reinitialize camera and hands if settings changed
                    if next_screen_name == 'play':
                        self._reinitialize_from_config()
        
        # Cleanup
        self.cleanup()
    
    def _reinitialize_from_config(self):
        """Reinitialize camera and hand detection based on current config."""
        # Close existing camera
        if self.cap.isOpened():
            self.cap.release()
        
        # Reopen with new camera ID
        camera_id = self.config.get('camera_id', 0)
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"Warning: Cannot open camera {camera_id}, using default")
            self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Update hand detection parameters
        self.hands.close()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5)
        )
        
        # Update trigger threshold
        self.trigger_threshold = self.config.get('trigger_threshold', 0.15)
        
        # Update chord preset
        preset_name = self.config.get('chord_preset', 'default')
        self.current_preset = self.CHORD_PRESETS.get(preset_name, self.CHORD_PRESETS['default'])
        
        # Reset finger states
        self.finger_states = [False] * 5
        self.finger_extensions = [1.0] * 5
        self.finger_base_extensions = [0.0] * 5
        self.finger_flexion_percent = [0.0] * 5
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        pygame.mixer.quit()
        print("Finger Piano stopped.")


def main():
    """Main entry point for the application."""
    try:
        config = Config()
        piano = FingerPiano(config)
        piano.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
