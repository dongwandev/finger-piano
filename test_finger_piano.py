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
    
    def test_finger_data_extraction(self):
        """Test finger extension data extraction from hand landmarks."""
        # Mock hand landmarks
        mock_landmarks = Mock()
        mock_landmarks.landmark = []
        
        # Create 21 landmarks (standard for MediaPipe hands)
        for i in range(21):
            landmark = Mock()
            landmark.x = 0.1 * i
            landmark.y = 0.2 * i
            landmark.z = 0.05 * i
            mock_landmarks.landmark.append(landmark)
        
        finger_data = self.piano._get_finger_data(mock_landmarks)
        
        # Should return 5 finger extension values
        self.assertEqual(len(finger_data), 5)
        
        # Each entry should have (finger_id, extension)
        for i, (finger_id, extension) in enumerate(finger_data):
            self.assertEqual(finger_id, i)
            self.assertIsInstance(extension, float)
            self.assertGreater(extension, 0)  # Extension should be positive
    
    def test_finger_bending_detection(self):
        """Test finger bending detection logic."""
        # Initialize extensions (fully extended fingers)
        self.piano.finger_extensions = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.piano.finger_states = [False, False, False, False, False]
        
        # Test significant bending (20% decrease, should trigger with 15% threshold)
        result = self.piano._detect_finger_bending(0, 0.16)
        self.assertTrue(result)
        
        # Test small bending (10% decrease, should not trigger with 15% threshold)
        self.piano.finger_states[1] = False
        result = self.piano._detect_finger_bending(1, 0.18)
        self.assertFalse(result)
        
        # Test extending (should not trigger)
        self.piano.finger_states[2] = False
        result = self.piano._detect_finger_bending(2, 0.25)
        self.assertFalse(result)
    
    def test_finger_state_reset(self):
        """Test finger state reset on extending (straightening)."""
        # Set initial state (fingers bent)
        self.piano.finger_extensions = [0.16, 0.16, 0.16, 0.16, 0.16]
        self.piano.finger_states = [True, True, True, True, True]
        
        # Test significant extension (25% increase, should reset with 15% threshold)
        self.piano._reset_finger_state(0, 0.20)
        self.assertFalse(self.piano.finger_states[0])
        
        # Test small extension (10% increase, should not reset with 15% threshold)
        self.piano.finger_states[1] = True
        self.piano._reset_finger_state(1, 0.176)
        self.assertTrue(self.piano.finger_states[1])
    
    def test_initial_state(self):
        """Test that initial state is correct."""
        self.assertEqual(len(self.piano.finger_states), 5)
        self.assertEqual(len(self.piano.finger_extensions), 5)
        
        # All fingers should start inactive
        for state in self.piano.finger_states:
            self.assertFalse(state)
        
        # All extensions should start at 1.0
        for ext in self.piano.finger_extensions:
            self.assertEqual(ext, 1.0)
    
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
        """Test complete bend-extend-bend cycle for finger recognition.
        
        This test verifies that:
        1. Finger bending triggers chord
        2. Finger extending resets state
        3. Finger can be bent again to re-trigger chord
        """
        # Mock the sound playing for chords
        for chord_name in self.piano.sounds:
            self.piano.sounds[chord_name] = Mock()
        
        finger_id = 0
        
        # Initial state: finger at extended position
        self.piano.finger_extensions[finger_id] = 0.2
        self.piano.finger_states[finger_id] = False
        
        # Step 1: Finger bends (extension decreases by 20%) - should trigger
        current_extension = 0.16  # 20% decrease > threshold (15%)
        
        # Check for reset (shouldn't happen, finger bent)
        self.piano._reset_finger_state(finger_id, current_extension)
        self.assertFalse(self.piano.finger_states[finger_id])
        
        # Check for bending (should trigger)
        should_play = self.piano._detect_finger_bending(finger_id, current_extension)
        self.assertTrue(should_play)
        
        # Play chord
        if should_play:
            self.piano._play_note(finger_id)
        
        # Update extension
        self.piano._update_finger_extension(finger_id, current_extension)
        
        # Verify state after bending
        self.assertTrue(self.piano.finger_states[finger_id])
        self.assertEqual(self.piano.finger_extensions[finger_id], 0.16)
        self.piano.sounds['C Major'].play.assert_called_once()
        
        # Step 2: Finger held bent - should NOT re-trigger
        current_extension = 0.16  # still at same extension
        
        self.piano._reset_finger_state(finger_id, current_extension)
        should_play = self.piano._detect_finger_bending(finger_id, current_extension)
        self.assertFalse(should_play)  # Should not trigger again
        self.piano._update_finger_extension(finger_id, current_extension)
        
        # State should remain active
        self.assertTrue(self.piano.finger_states[finger_id])
        
        # Step 3: Finger extends (straightens) - should reset state
        current_extension = 0.20  # 25% increase > threshold (15%)
        
        # Check for reset (should happen now)
        reset_happened = self.piano._reset_finger_state(finger_id, current_extension)
        self.assertTrue(reset_happened)
        self.assertFalse(self.piano.finger_states[finger_id])
        
        # Check for bending (shouldn't trigger, finger extended)
        should_play = self.piano._detect_finger_bending(finger_id, current_extension)
        self.assertFalse(should_play)
        
        # Update extension
        self.piano._update_finger_extension(finger_id, current_extension)
        
        # Verify state after extending
        self.assertFalse(self.piano.finger_states[finger_id])
        self.assertEqual(self.piano.finger_extensions[finger_id], 0.20)
        
        # Step 4: Finger bends again (second bend) - should re-trigger
        current_extension = 0.16  # 20% decrease > threshold (15%)
        
        self.piano._reset_finger_state(finger_id, current_extension)
        should_play = self.piano._detect_finger_bending(finger_id, current_extension)
        self.assertTrue(should_play)  # Should trigger again!
        
        if should_play:
            self.piano._play_note(finger_id)
        
        self.piano._update_finger_extension(finger_id, current_extension)
        
        # Verify state after second bend
        self.assertTrue(self.piano.finger_states[finger_id])
        self.assertEqual(self.piano.finger_extensions[finger_id], 0.16)
        
        # Sound should have been played twice total
        self.assertEqual(self.piano.sounds['C Major'].play.call_count, 2)
    
    def test_thumb_bending_detection(self):
        """Test that thumb bending is properly detected with new algorithm.
        
        This specifically tests that thumb (finger_id=0) can trigger chords
        by bending, regardless of x-axis or y-axis movement. This addresses
        the issue where y-axis-only detection failed for thumbs.
        """
        # Mock the sound playing
        for chord_name in self.piano.sounds:
            self.piano.sounds[chord_name] = Mock()
        
        finger_id = 0  # Thumb
        
        # Initial state: thumb extended
        self.piano.finger_extensions[finger_id] = 0.25
        self.piano.finger_states[finger_id] = False
        
        # Thumb bends (curls) - extension decreases by 20%
        # This could happen from ANY direction (x, y, or z movement)
        # because we measure 3D Euclidean distance
        bent_extension = 0.20  # 20% decrease
        
        should_play = self.piano._detect_finger_bending(finger_id, bent_extension)
        self.assertTrue(should_play, "Thumb bending should trigger chord")
        
        if should_play:
            self.piano._play_note(finger_id)
        
        self.piano._update_finger_extension(finger_id, bent_extension)
        
        # Verify thumb chord (C Major) was played
        self.assertTrue(self.piano.finger_states[finger_id])
        self.piano.sounds['C Major'].play.assert_called_once()
        
        # Thumb extends back - should reset
        extended_extension = 0.25  # Back to original
        self.piano._reset_finger_state(finger_id, extended_extension)
        self.assertFalse(self.piano.finger_states[finger_id], 
                        "Thumb extending should reset state")


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
        self.assertEqual(config.get('trigger_threshold'), 0.15)
        self.assertEqual(config.get('chord_preset'), 'default')
    
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


class TestChordPresets(unittest.TestCase):
    """Test chord preset functionality."""
    
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
    
    def test_chord_presets_defined(self):
        """Test that all chord presets are defined."""
        self.assertIn('default', FingerPiano.CHORD_PRESETS)
        self.assertIn('preset1', FingerPiano.CHORD_PRESETS)
        self.assertIn('preset2', FingerPiano.CHORD_PRESETS)
        self.assertIn('preset3', FingerPiano.CHORD_PRESETS)
        
        # Check default preset
        default = FingerPiano.CHORD_PRESETS['default']
        self.assertEqual(len(default), 5)
        self.assertEqual(default, ['C Major', 'G Major', 'A Minor', 'F Major', 'D Major'])
        
        # Check preset1
        preset1 = FingerPiano.CHORD_PRESETS['preset1']
        self.assertEqual(len(preset1), 5)
        self.assertEqual(preset1, ['G Major', 'D Major', 'E Minor', 'C Major', ''])
        
        # Check preset2
        preset2 = FingerPiano.CHORD_PRESETS['preset2']
        self.assertEqual(len(preset2), 5)
        self.assertEqual(preset2, ['A Minor', 'C Major', 'G Major', 'D Major', ''])
        
        # Check preset3
        preset3 = FingerPiano.CHORD_PRESETS['preset3']
        self.assertEqual(len(preset3), 5)
        self.assertEqual(preset3, ['E Major', 'A Minor', 'A Sus4', '', ''])
    
    def test_new_chords_have_definitions(self):
        """Test that new chords have proper definitions."""
        self.assertIn('E Minor', FingerPiano.CHORDS)
        self.assertIn('E Major', FingerPiano.CHORDS)
        self.assertIn('A Sus4', FingerPiano.CHORDS)
        
        # Check chord notes
        self.assertEqual(FingerPiano.CHORDS['E Minor'], ['E4', 'G4', 'B4'])
        self.assertEqual(FingerPiano.CHORDS['E Major'], ['E4', 'G#4', 'B4'])
        self.assertEqual(FingerPiano.CHORDS['A Sus4'], ['A4', 'D5', 'E5'])
    
    def test_new_notes_have_frequencies(self):
        """Test that new notes have frequencies defined."""
        self.assertIn('G#4', FingerPiano.NOTE_FREQUENCIES)
        self.assertGreater(FingerPiano.NOTE_FREQUENCIES['G#4'], 0)
    
    def test_default_preset_loaded(self):
        """Test that default preset is loaded by default."""
        self.assertEqual(self.piano.current_preset, 
                        FingerPiano.CHORD_PRESETS['default'])
    
    def test_unassigned_finger_no_sound(self):
        """Test that unassigned fingers don't play sound."""
        # Mock the sound playing
        for chord_name in self.piano.sounds:
            self.piano.sounds[chord_name] = Mock()
        
        # Switch to preset3 which has unassigned fingers
        self.piano.current_preset = FingerPiano.CHORD_PRESETS['preset3']
        
        # Try to play unassigned finger (finger 3 and 4 are unassigned in preset3)
        self.piano.finger_states[3] = False
        self.piano._play_note(3)
        
        # Finger state should not be set to True
        self.assertFalse(self.piano.finger_states[3])
        
        # Try to play assigned finger (finger 0 is E Major in preset3)
        self.piano.finger_states[0] = False
        self.piano._play_note(0)
        
        # Finger state should be set to True
        self.assertTrue(self.piano.finger_states[0])
        self.piano.sounds['E Major'].play.assert_called_once()
    
    def test_preset_change_in_config(self):
        """Test changing preset through config."""
        # Change to preset1
        self.piano.config.set('chord_preset', 'preset1')
        self.piano._reinitialize_from_config()
        
        self.assertEqual(self.piano.current_preset, 
                        FingerPiano.CHORD_PRESETS['preset1'])
        
        # Change to preset2
        self.piano.config.set('chord_preset', 'preset2')
        self.piano._reinitialize_from_config()
        
        self.assertEqual(self.piano.current_preset,
                        FingerPiano.CHORD_PRESETS['preset2'])


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
