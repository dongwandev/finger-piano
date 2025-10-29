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


class FingerPiano:
    """Main class for the finger piano virtual instrument."""
    
    # Piano key mapping for fingers (C major scale)
    # Index: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    NOTES = ['C4', 'D4', 'E4', 'F4', 'G4']
    
    # MIDI note frequencies
    NOTE_FREQUENCIES = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25
    }
    
    # Colors for visualization
    COLORS = {
        'active': (0, 255, 0),      # Green for active finger
        'inactive': (255, 0, 0),    # Red for inactive finger
        'landmark': (255, 255, 0),  # Yellow for landmarks
        'connection': (0, 255, 255) # Cyan for connections
    }
    
    def __init__(self, camera_id: int = 0):
        """Initialize the finger piano.
        
        Args:
            camera_id: Camera device ID (default: 0)
        """
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
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
        
        # Track finger states (whether each finger is currently playing)
        self.finger_states = [False] * 5
        self.finger_y_positions = [0.0] * 5
        
        # Threshold for triggering notes (finger tip y-position change)
        self.trigger_threshold = 0.05
        
    def _generate_piano_sounds(self) -> dict:
        """Generate synthesized piano sounds for each note.
        
        Returns:
            Dictionary mapping note names to pygame Sound objects
        """
        sounds = {}
        sample_rate = 22050
        duration = 0.5  # seconds
        
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
            
            # Normalize and convert to 16-bit integers
            if np.max(np.abs(wave)) > 0:
                wave = wave / np.max(np.abs(wave))
            wave = (wave * 32767).astype(np.int16)
            
            # Create stereo sound
            stereo_wave = np.column_stack((wave, wave))
            
            # Create pygame Sound object
            sounds[note] = pygame.sndarray.make_sound(stereo_wave)
            
        return sounds
    
    def _get_finger_tips(self, hand_landmarks) -> List[Tuple[int, float, float]]:
        """Extract finger tip positions from hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            List of tuples (finger_id, x, y) for each finger tip
        """
        # MediaPipe hand landmark indices for finger tips
        finger_tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        tips = []
        for i, tip_id in enumerate(finger_tip_ids):
            landmark = hand_landmarks.landmark[tip_id]
            tips.append((i, landmark.x, landmark.y))
        
        return tips
    
    def _detect_finger_movement(self, finger_id: int, current_y: float) -> bool:
        """Detect if a finger has moved up (like folding/bending a finger).
        
        Args:
            finger_id: Index of the finger (0-4)
            current_y: Current y-position of the finger tip
            
        Returns:
            True if the finger has moved up significantly (folded)
        """
        previous_y = self.finger_y_positions[finger_id]
        
        # Check if finger moved up (y decreases upward in image coordinates, finger folding)
        movement = previous_y - current_y
        
        # Update position
        self.finger_y_positions[finger_id] = current_y
        
        # Trigger if moved up (folded) and not already playing
        if movement > self.trigger_threshold and not self.finger_states[finger_id]:
            return True
        
        return False
    
    def _play_note(self, finger_id: int):
        """Play the note associated with a finger.
        
        Args:
            finger_id: Index of the finger (0-4)
        """
        if finger_id < len(self.NOTES):
            note = self.NOTES[finger_id]
            if note in self.sounds:
                self.sounds[note].play()
                self.finger_states[finger_id] = True
    
    def _reset_finger_state(self, finger_id: int, current_y: float):
        """Reset finger state when it extends back down.
        
        Args:
            finger_id: Index of the finger (0-4)
            current_y: Current y-position of the finger tip
        """
        previous_y = self.finger_y_positions[finger_id]
        
        # Reset if finger extended down significantly
        if current_y > previous_y + self.trigger_threshold:
            self.finger_states[finger_id] = False
    
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
        cv2.putText(image, "Fold fingers to play notes", 
                   (10, image_height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press 'q' to quit", 
                   (10, image_height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
    
    def run(self):
        """Main loop for the finger piano application."""
        print("Finger Piano started. Press 'q' to quit.")
        print("Fold your fingers to play piano notes!")
        
        while True:
            # Read frame from camera
            success, frame = self.cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Flip the frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = self.hands.process(rgb_frame)
            
            # Process hand landmarks and detect finger movements
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get finger tip positions
                finger_tips = self._get_finger_tips(hand_landmarks)
                
                # Check each finger for movement
                for finger_id, x, y in finger_tips:
                    # Detect downward movement and play note
                    if self._detect_finger_movement(finger_id, y):
                        self._play_note(finger_id)
                    
                    # Reset state if finger moved up
                    self._reset_finger_state(finger_id, y)
                
                # Draw UI elements
                self._draw_ui(frame, hand_landmarks, frame.shape[1], frame.shape[0])
            else:
                # No hand detected - reset all states
                self.finger_states = [False] * 5
                
                # Draw UI without hand landmarks
                self._draw_ui(frame, None, frame.shape[1], frame.shape[0])
            
            # Display the frame
            cv2.imshow('Finger Piano', frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cleanup()
    
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
        piano = FingerPiano()
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
