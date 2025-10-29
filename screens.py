#!/usr/bin/env python3
"""
GUI screens for the Finger Piano application.
Implements lobby, settings, and play screens.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Callable
from config import Config


class Screen:
    """Base class for all GUI screens."""
    
    def __init__(self, name: str, config: Config):
        """Initialize screen.
        
        Args:
            name: Name of the screen
            config: Application configuration
        """
        self.name = name
        self.config = config
        self.next_screen = None
        self.running = True
    
    def handle_event(self, event: int) -> Optional[str]:
        """Handle keyboard or mouse events.
        
        Args:
            event: Event code from cv2.waitKey
            
        Returns:
            Name of next screen to navigate to, or None
        """
        return None
    
    def render(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Render the screen.
        
        Args:
            frame: Optional camera frame
            
        Returns:
            Rendered frame as numpy array
        """
        raise NotImplementedError
    
    def on_enter(self):
        """Called when entering this screen."""
        pass
    
    def on_exit(self):
        """Called when leaving this screen."""
        pass


class LobbyScreen(Screen):
    """Lobby screen with main menu options."""
    
    BUTTON_WIDTH = 300
    BUTTON_HEIGHT = 60
    BUTTON_SPACING = 30
    
    def __init__(self, config: Config):
        """Initialize lobby screen."""
        super().__init__('lobby', config)
        self.selected_button = 0
        self.buttons = [
            {'text': 'Start Playing', 'action': 'play'},
            {'text': 'Settings', 'action': 'settings'},
            {'text': 'Quit', 'action': 'quit'}
        ]
    
    def handle_event(self, event: int) -> Optional[str]:
        """Handle keyboard events."""
        if event == ord('q') or event == 27:  # q or ESC
            return 'quit'
        elif event == ord('w') or event == 82:  # w or UP arrow
            self.selected_button = (self.selected_button - 1) % len(self.buttons)
        elif event == ord('s') or event == 84:  # s or DOWN arrow
            self.selected_button = (self.selected_button + 1) % len(self.buttons)
        elif event == 13 or event == 32:  # ENTER or SPACE
            return self.buttons[self.selected_button]['action']
        elif event == ord('1'):
            return 'play'
        elif event == ord('2'):
            return 'settings'
        
        return None
    
    def render(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Render lobby screen."""
        # Create blank canvas
        height, width = 720, 1280
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw title
        title = "Finger Piano"
        title_emoji = "🎹✋"
        cv2.putText(canvas, title, (width // 2 - 200, 150),
                   cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 4)
        
        # Draw subtitle
        subtitle = "Play piano with your fingers!"
        cv2.putText(canvas, subtitle, (width // 2 - 250, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        # Calculate button positions
        total_height = len(self.buttons) * self.BUTTON_HEIGHT + \
                      (len(self.buttons) - 1) * self.BUTTON_SPACING
        start_y = (height - total_height) // 2 + 100
        
        # Draw buttons
        for i, button in enumerate(self.buttons):
            x = (width - self.BUTTON_WIDTH) // 2
            y = start_y + i * (self.BUTTON_HEIGHT + self.BUTTON_SPACING)
            
            # Button background
            color = (0, 200, 100) if i == self.selected_button else (70, 70, 70)
            cv2.rectangle(canvas, (x, y), 
                         (x + self.BUTTON_WIDTH, y + self.BUTTON_HEIGHT),
                         color, -1)
            
            # Button border
            border_color = (255, 255, 255) if i == self.selected_button else (150, 150, 150)
            cv2.rectangle(canvas, (x, y),
                         (x + self.BUTTON_WIDTH, y + self.BUTTON_HEIGHT),
                         border_color, 2)
            
            # Button text
            text = button['text']
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = x + (self.BUTTON_WIDTH - text_size[0]) // 2
            text_y = y + (self.BUTTON_HEIGHT + text_size[1]) // 2
            cv2.putText(canvas, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "Use UP/DOWN arrows or W/S to navigate",
            "Press ENTER or SPACE to select",
            "Or press 1 for Play, 2 for Settings"
        ]
        
        y_offset = height - 100
        for instruction in instructions:
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x = (width - text_size[0]) // 2
            cv2.putText(canvas, instruction, (x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_offset += 25
        
        return canvas


class SettingsScreen(Screen):
    """Settings screen for configuration."""
    
    def __init__(self, config: Config):
        """Initialize settings screen."""
        super().__init__('settings', config)
        self.selected_option = 0
        self.camera_test_active = False
        self.test_cap = None
        
        # Define menu options
        self.options = [
            {'label': 'Camera ID', 'type': 'camera', 'values': [0, 1, 2]},
            {'label': 'Test Camera', 'type': 'test_camera'},
            {'label': 'Instrument', 'type': 'instrument', 
             'values': ['Piano', 'Guitar', 'Electric Guitar', 'Violin']},
            {'label': 'Detection Confidence', 'type': 'detection', 
             'values': [0.5, 0.6, 0.7, 0.8, 0.9]},
            {'label': 'Tracking Confidence', 'type': 'tracking',
             'values': [0.3, 0.4, 0.5, 0.6, 0.7]},
            {'label': 'Trigger Threshold', 'type': 'trigger',
             'values': [0.10, 0.12, 0.15, 0.18, 0.20]},
            {'label': 'Save & Return', 'type': 'save'},
            {'label': 'Cancel', 'type': 'cancel'}
        ]
    
    def handle_event(self, event: int) -> Optional[str]:
        """Handle keyboard events."""
        if self.camera_test_active:
            if event == ord('q') or event == 27:  # q or ESC
                self._stop_camera_test()
            return None
        
        if event == ord('q') or event == 27:  # q or ESC
            return 'lobby'
        elif event == ord('w') or event == 82:  # w or UP arrow
            self.selected_option = (self.selected_option - 1) % len(self.options)
        elif event == ord('s') or event == 84:  # s or DOWN arrow
            self.selected_option = (self.selected_option + 1) % len(self.options)
        elif event == ord('a') or event == 81:  # a or LEFT arrow
            self._change_value(-1)
        elif event == ord('d') or event == 83:  # d or RIGHT arrow
            self._change_value(1)
        elif event == 13 or event == 32:  # ENTER or SPACE
            return self._activate_option()
        
        return None
    
    def _change_value(self, direction: int):
        """Change the value of current option."""
        option = self.options[self.selected_option]
        opt_type = option['type']
        
        if 'values' not in option:
            return
        
        values = option['values']
        
        if opt_type == 'camera':
            current = self.config.get('camera_id', 0)
            try:
                current_idx = values.index(current)
            except ValueError:
                current_idx = 0
            new_idx = (current_idx + direction) % len(values)
            self.config.set('camera_id', values[new_idx])
        
        elif opt_type == 'instrument':
            instruments_map = ['piano', 'guitar', 'electric_guitar', 'violin']
            current = self.config.get('instrument', 'piano')
            try:
                current_idx = instruments_map.index(current)
            except ValueError:
                current_idx = 0
            new_idx = (current_idx + direction) % len(values)
            self.config.set('instrument', instruments_map[new_idx])
        
        elif opt_type == 'detection':
            current = self.config.get('min_detection_confidence', 0.7)
            try:
                current_idx = values.index(current)
            except ValueError:
                current_idx = 2
            new_idx = (current_idx + direction) % len(values)
            self.config.set('min_detection_confidence', values[new_idx])
        
        elif opt_type == 'tracking':
            current = self.config.get('min_tracking_confidence', 0.5)
            try:
                current_idx = values.index(current)
            except ValueError:
                current_idx = 2
            new_idx = (current_idx + direction) % len(values)
            self.config.set('min_tracking_confidence', values[new_idx])
        
        elif opt_type == 'trigger':
            current = self.config.get('trigger_threshold', 0.15)
            try:
                current_idx = values.index(current)
            except ValueError:
                current_idx = 2
            new_idx = (current_idx + direction) % len(values)
            self.config.set('trigger_threshold', values[new_idx])
    
    def _activate_option(self) -> Optional[str]:
        """Activate the selected option."""
        option = self.options[self.selected_option]
        opt_type = option['type']
        
        if opt_type == 'test_camera':
            self._start_camera_test()
            return None
        elif opt_type == 'save':
            self.config.save()
            return 'lobby'
        elif opt_type == 'cancel':
            return 'lobby'
        
        return None
    
    def _start_camera_test(self):
        """Start camera test mode."""
        camera_id = self.config.get('camera_id', 0)
        self.test_cap = cv2.VideoCapture(camera_id)
        if self.test_cap.isOpened():
            self.camera_test_active = True
        else:
            self.test_cap = None
    
    def _stop_camera_test(self):
        """Stop camera test mode."""
        if self.test_cap is not None:
            self.test_cap.release()
            self.test_cap = None
        self.camera_test_active = False
    
    def on_exit(self):
        """Cleanup when leaving settings."""
        self._stop_camera_test()
    
    def render(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Render settings screen."""
        if self.camera_test_active and self.test_cap is not None:
            # Show camera test
            success, test_frame = self.test_cap.read()
            if success:
                test_frame = cv2.flip(test_frame, 1)
                # Add overlay text
                cv2.putText(test_frame, "Camera Test - Press 'q' or ESC to close",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                return test_frame
            else:
                self._stop_camera_test()
        
        # Create settings canvas
        height, width = 720, 1280
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw title
        cv2.putText(canvas, "Settings", (50, 80),
                   cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3)
        
        # Draw options
        y_offset = 160
        for i, option in enumerate(self.options):
            color = (0, 200, 100) if i == self.selected_option else (200, 200, 200)
            
            # Draw option label
            label = option['label']
            cv2.putText(canvas, label, (100, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw current value
            value_text = self._get_value_text(option)
            if value_text:
                cv2.putText(canvas, value_text, (500, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw selection indicator
            if i == self.selected_option:
                cv2.putText(canvas, ">", (60, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            y_offset += 50
        
        # Draw instructions
        instructions = [
            "UP/DOWN or W/S: Navigate options",
            "LEFT/RIGHT or A/D: Change values",
            "ENTER or SPACE: Activate",
            "ESC or Q: Back to lobby"
        ]
        
        y_offset = height - 120
        for instruction in instructions:
            cv2.putText(canvas, instruction, (80, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_offset += 25
        
        return canvas
    
    def _get_value_text(self, option: dict) -> str:
        """Get current value text for an option."""
        opt_type = option['type']
        
        if opt_type == 'camera':
            return f"Camera {self.config.get('camera_id', 0)}"
        elif opt_type == 'test_camera':
            return "[Press ENTER to test]"
        elif opt_type == 'instrument':
            instruments_map = {
                'piano': 'Piano',
                'guitar': 'Guitar',
                'electric_guitar': 'Electric Guitar',
                'violin': 'Violin'
            }
            inst = self.config.get('instrument', 'piano')
            return instruments_map.get(inst, 'Piano')
        elif opt_type == 'detection':
            return f"{self.config.get('min_detection_confidence', 0.7):.1f}"
        elif opt_type == 'tracking':
            return f"{self.config.get('min_tracking_confidence', 0.5):.1f}"
        elif opt_type == 'trigger':
            return f"{self.config.get('trigger_threshold', 0.05):.2f}"
        elif opt_type in ['save', 'cancel']:
            return "[Press ENTER]"
        
        return ""


class PlayScreen(Screen):
    """Play screen for performing with the instrument."""
    
    def __init__(self, config: Config, piano_instance):
        """Initialize play screen.
        
        Args:
            config: Application configuration
            piano_instance: Instance of FingerPiano
        """
        super().__init__('play', config)
        self.piano = piano_instance
    
    def handle_event(self, event: int) -> Optional[str]:
        """Handle keyboard events."""
        if event == ord('q') or event == 27:  # q or ESC
            return 'lobby'
        return None
    
    def render(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Render play screen with camera and overlays."""
        if frame is None:
            # Create blank frame if no camera feed
            height, width = 720, 1280
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, "No camera feed", (width // 2 - 150, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        height, width = frame.shape[:2]
        
        # Draw finger status in top-left
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        y_offset = 30
        
        # Draw background for finger status (wider to accommodate chord names)
        cv2.rectangle(frame, (5, 5), (300, 175), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (300, 175), (255, 255, 255), 2)
        
        for i in range(5):
            chord = self.piano.NOTES[i] if i < len(self.piano.NOTES) else 'N/A'
            is_active = self.piano.finger_states[i] if i < len(self.piano.finger_states) else False
            color = (0, 255, 0) if is_active else (100, 100, 100)
            
            text = f"{finger_names[i]}: {chord}"
            cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)
            y_offset += 30
        
        # Draw instrument in top-right
        instruments_map = {
            'piano': 'Piano',
            'guitar': 'Guitar',
            'electric_guitar': 'Electric Guitar',
            'violin': 'Violin'
        }
        instrument = instruments_map.get(self.config.get('instrument', 'piano'), 'Piano')
        
        text = f"Instrument: {instrument}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Draw background for instrument display
        cv2.rectangle(frame, (width - text_size[0] - 30, 5), 
                     (width - 5, 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - text_size[0] - 30, 5),
                     (width - 5, 50), (255, 255, 255), 2)
        
        cv2.putText(frame, text, (width - text_size[0] - 20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        
        # Draw bottom instructions
        cv2.putText(frame, "Press 'q' or ESC to return to lobby",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        
        return frame
