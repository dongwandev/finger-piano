#!/usr/bin/env python3
"""
Unit tests for the Finger Piano application.
Tests core functionality without requiring a camera.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path is set
from finger_piano import FingerPiano
from config import Config


class TestFingerPiano(unittest.TestCase):
    """Test cases for FingerPiano class."""
    
    @patch('finger_piano.cv2.VideoCapture')
    @patch('finger_piano.pygame.mixer.init')
    @patch('finger_piano.pygame.sndarray.make_sound')
    def setUp(self, mock_make_sound, mock_mixer, mock_capture):
        """Set up test fixtures."""
        # Mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        # Mock sound creation
        mock_sound = Mock()
        mock_make_sound.return_value = mock_sound
        
        # Create config
        config = Config()
        
        # Create instance
        self.piano = FingerPiano(config)
    
    def test_note_frequencies(self):
        """Test that all expected notes have frequencies defined."""
        # Check that all notes needed for chords are in NOTE_FREQUENCIES
        all_needed_notes = set()
        for chord_notes in FingerPiano.CHORDS.values():
            all_needed_notes.update(chord_notes)
        
        for note in all_needed_notes:
            self.assertIn(note, FingerPiano.NOTE_FREQUENCIES)
            self.assertIsInstance(FingerPiano.NOTE_FREQUENCIES[note], (int, float))
            self.assertGreater(FingerPiano.NOTE_FREQUENCIES[note], 0)
    
    def test_sound_generation(self):
        """Test that piano sounds are generated for all chords."""
        self.assertIsNotNone(self.piano.sounds)
        
        # Check that sounds were generated for all chords
        for chord_name in self.piano.CHORDS.keys():
            self.assertIn(chord_name, self.piano.sounds)
    
    def test_finger_tips_extraction(self):
        """Test finger tip extraction from hand landmarks."""
        # Mock hand landmarks
        mock_landmarks = Mock()
        mock_landmarks.landmark = []
        
        # Create 21 landmarks (standard for MediaPipe hands)
        for i in range(21):
            landmark = Mock()
            landmark.x = 0.1 * i
            landmark.y = 0.2 * i
            mock_landmarks.landmark.append(landmark)
        
        tips = self.piano._get_finger_tips(mock_landmarks)
        
        # Should return 5 finger tips
        self.assertEqual(len(tips), 5)
        
        # Each tip should have (finger_id, x, y)
        for i, (finger_id, x, y) in enumerate(tips):
            self.assertEqual(finger_id, i)
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)
    
    def test_finger_movement_detection(self):
        """Test finger movement detection logic."""
        # Initialize positions
        self.piano.finger_y_positions = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.piano.finger_states = [False, False, False, False, False]
        
        # Test downward movement (should trigger)
        result = self.piano._detect_finger_movement(0, 0.6)
        self.assertTrue(result)
        
        # Test small movement (should not trigger)
        self.piano.finger_states[1] = False
        result = self.piano._detect_finger_movement(1, 0.52)
        self.assertFalse(result)
        
        # Test upward movement (should not trigger)
        self.piano.finger_states[2] = False
        result = self.piano._detect_finger_movement(2, 0.4)
        self.assertFalse(result)
    
    def test_finger_state_reset(self):
        """Test finger state reset on upward movement."""
        # Set initial state
        self.piano.finger_y_positions = [0.6, 0.6, 0.6, 0.6, 0.6]
        self.piano.finger_states = [True, True, True, True, True]
        
        # Test upward movement (should reset)
        self.piano._reset_finger_state(0, 0.5)
        self.assertFalse(self.piano.finger_states[0])
        
        # Test small upward movement (should not reset)
        self.piano.finger_states[1] = True
        self.piano._reset_finger_state(1, 0.58)
        self.assertTrue(self.piano.finger_states[1])
    
    def test_initial_state(self):
        """Test that initial state is correct."""
        self.assertEqual(len(self.piano.finger_states), 5)
        self.assertEqual(len(self.piano.finger_y_positions), 5)
        
        # All fingers should start inactive
        for state in self.piano.finger_states:
            self.assertFalse(state)
        
        # All positions should start at 0
        for pos in self.piano.finger_y_positions:
            self.assertEqual(pos, 0.0)
    
    def test_play_note(self):
        """Test note playing functionality."""
        # Mock the sound playing for chords
        for chord_name in self.piano.sounds:
            self.piano.sounds[chord_name] = Mock()
        
        # Play chord for finger 0 (C Major)
        self.piano._play_note(0)
        
        # Check that state is updated
        self.assertTrue(self.piano.finger_states[0])
        
        # Check that chord sound was played
        chord = self.piano.NOTES[0]
        self.piano.sounds[chord].play.assert_called_once()
    
    def test_press_release_press_cycle(self):
        """Test complete press-release-press cycle for finger recognition.
        
        This test verifies that:
        1. Finger press triggers chord
        2. Finger release resets state
        3. Finger can be pressed again to re-trigger chord
        """
        # Mock the sound playing for chords
        for chord_name in self.piano.sounds:
            self.piano.sounds[chord_name] = Mock()
        
        finger_id = 0
        
        # Initial state: finger at rest position
        self.piano.finger_y_positions[finger_id] = 0.5
        self.piano.finger_states[finger_id] = False
        
        # Step 1: Finger moves down (press) - should trigger
        current_y = 0.6  # moved down by 0.1 > threshold (0.05)
        
        # Check for reset (shouldn't happen, finger moved down)
        self.piano._reset_finger_state(finger_id, current_y)
        self.assertFalse(self.piano.finger_states[finger_id])
        
        # Check for downward movement (should trigger)
        should_play = self.piano._detect_finger_movement(finger_id, current_y)
        self.assertTrue(should_play)
        
        # Play chord
        if should_play:
            self.piano._play_note(finger_id)
        
        # Update position
        self.piano._update_finger_position(finger_id, current_y)
        
        # Verify state after press
        self.assertTrue(self.piano.finger_states[finger_id])
        self.assertEqual(self.piano.finger_y_positions[finger_id], 0.6)
        self.piano.sounds['C Major'].play.assert_called_once()
        
        # Step 2: Finger held down - should NOT re-trigger
        current_y = 0.6  # still at same position
        
        self.piano._reset_finger_state(finger_id, current_y)
        should_play = self.piano._detect_finger_movement(finger_id, current_y)
        self.assertFalse(should_play)  # Should not trigger again
        self.piano._update_finger_position(finger_id, current_y)
        
        # State should remain active
        self.assertTrue(self.piano.finger_states[finger_id])
        
        # Step 3: Finger moves up (release) - should reset state
        current_y = 0.4  # moved up by 0.2 > threshold (0.05)
        
        # Check for reset (should happen now)
        reset_happened = self.piano._reset_finger_state(finger_id, current_y)
        self.assertTrue(reset_happened)
        self.assertFalse(self.piano.finger_states[finger_id])
        
        # Check for downward movement (shouldn't trigger, moved up)
        should_play = self.piano._detect_finger_movement(finger_id, current_y)
        self.assertFalse(should_play)
        
        # Update position
        self.piano._update_finger_position(finger_id, current_y)
        
        # Verify state after release
        self.assertFalse(self.piano.finger_states[finger_id])
        self.assertEqual(self.piano.finger_y_positions[finger_id], 0.4)
        
        # Step 4: Finger moves down again (second press) - should re-trigger
        current_y = 0.55  # moved down by 0.15 > threshold (0.05)
        
        self.piano._reset_finger_state(finger_id, current_y)
        should_play = self.piano._detect_finger_movement(finger_id, current_y)
        self.assertTrue(should_play)  # Should trigger again!
        
        if should_play:
            self.piano._play_note(finger_id)
        
        self.piano._update_finger_position(finger_id, current_y)
        
        # Verify state after second press
        self.assertTrue(self.piano.finger_states[finger_id])
        self.assertEqual(self.piano.finger_y_positions[finger_id], 0.55)
        
        # Sound should have been played twice total
        self.assertEqual(self.piano.sounds['C Major'].play.call_count, 2)


class TestSoundGeneration(unittest.TestCase):
    """Test sound synthesis functionality."""
    
    def test_wave_generation(self):
        """Test that generated waves are valid numpy arrays."""
        sample_rate = 22050
        duration = 0.1
        frequency = 440.0
        
        samples = int(sample_rate * duration)
        wave = np.zeros(samples)
        
        for i in range(samples):
            t = i / sample_rate
            wave[i] = np.sin(2 * np.pi * frequency * t)
        
        # Check wave properties
        self.assertEqual(len(wave), samples)
        self.assertIsInstance(wave, np.ndarray)
        
        # Check wave is bounded between -1 and 1
        self.assertLessEqual(np.max(wave), 1.0)
        self.assertGreaterEqual(np.min(wave), -1.0)


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config_file = '/tmp/test_finger_piano_config.json'
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config(self.test_config_file)
        
        self.assertEqual(config.get('camera_id'), 0)
        self.assertEqual(config.get('instrument'), 'piano')
        self.assertEqual(config.get('min_detection_confidence'), 0.7)
        self.assertEqual(config.get('min_tracking_confidence'), 0.5)
        self.assertEqual(config.get('trigger_threshold'), 0.05)
    
    def test_save_and_load(self):
        """Test saving and loading configuration."""
        config = Config(self.test_config_file)
        
        # Modify config
        config.set('camera_id', 1)
        config.set('instrument', 'guitar')
        config.set('min_detection_confidence', 0.8)
        
        # Save config
        self.assertTrue(config.save())
        
        # Load config in new instance
        config2 = Config(self.test_config_file)
        
        self.assertEqual(config2.get('camera_id'), 1)
        self.assertEqual(config2.get('instrument'), 'guitar')
        self.assertEqual(config2.get('min_detection_confidence'), 0.8)
    
    def test_config_reset(self):
        """Test resetting configuration to defaults."""
        config = Config(self.test_config_file)
        
        # Modify config
        config.set('camera_id', 2)
        config.set('instrument', 'violin')
        
        # Reset to defaults
        config.reset_to_defaults()
        
        self.assertEqual(config.get('camera_id'), 0)
        self.assertEqual(config.get('instrument'), 'piano')
    
    def test_instrument_options(self):
        """Test that all supported instruments are defined."""
        config = Config(self.test_config_file)
        
        # Test that INSTRUMENTS list contains all expected instruments
        expected_instruments = ['piano', 'guitar', 'electric_guitar', 'violin']
        self.assertEqual(Config.INSTRUMENTS, expected_instruments)
        
        # Test setting each instrument
        for instrument in expected_instruments:
            config.set('instrument', instrument)
            self.assertEqual(config.get('instrument'), instrument)


class TestInstruments(unittest.TestCase):
    """Test instrument-specific functionality."""
    
    @patch('finger_piano.cv2.VideoCapture')
    @patch('finger_piano.pygame.mixer.init')
    @patch('finger_piano.pygame.sndarray.make_sound')
    def test_sound_generation_for_all_instruments(self, mock_make_sound, mock_mixer, mock_capture):
        """Test that sounds are generated for all supported instruments."""
        # Mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        # Mock sound creation
        mock_sound = Mock()
        mock_make_sound.return_value = mock_sound
        
        instruments = ['piano', 'guitar', 'electric_guitar', 'violin']
        
        for instrument in instruments:
            # Create config with specific instrument
            config = Config()
            config.set('instrument', instrument)
            
            # Create instance
            piano = FingerPiano(config)
            
            # Verify sounds were generated
            self.assertIsNotNone(piano.sounds)
            self.assertGreater(len(piano.sounds), 0)
            
            # Verify all chords have sounds
            for chord_name in piano.CHORDS.keys():
                self.assertIn(chord_name, piano.sounds)
    
    @patch('finger_piano.cv2.VideoCapture')
    @patch('finger_piano.pygame.mixer.init')
    @patch('finger_piano.pygame.sndarray.make_sound')
    def test_instrument_selection(self, mock_make_sound, mock_mixer, mock_capture):
        """Test that instrument selection is properly stored in config."""
        # Mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        # Mock sound creation
        mock_sound = Mock()
        mock_make_sound.return_value = mock_sound
        
        # Test each instrument
        for instrument in ['piano', 'guitar', 'electric_guitar', 'violin']:
            config = Config()
            config.set('instrument', instrument)
            
            # Verify instrument is set correctly
            self.assertEqual(config.get('instrument'), instrument)
            
            # Create FingerPiano instance
            piano = FingerPiano(config)
            
            # Verify it uses the correct instrument
            self.assertEqual(piano.config.get('instrument'), instrument)


def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
