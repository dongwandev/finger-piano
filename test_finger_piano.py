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
        
        # Create instance
        self.piano = FingerPiano()
    
    def test_note_frequencies(self):
        """Test that all expected notes have frequencies defined."""
        expected_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        
        for note in expected_notes:
            self.assertIn(note, FingerPiano.NOTE_FREQUENCIES)
            self.assertIsInstance(FingerPiano.NOTE_FREQUENCIES[note], (int, float))
            self.assertGreater(FingerPiano.NOTE_FREQUENCIES[note], 0)
    
    def test_sound_generation(self):
        """Test that piano sounds are generated for all notes."""
        self.assertIsNotNone(self.piano.sounds)
        
        for note in self.piano.NOTE_FREQUENCIES.keys():
            self.assertIn(note, self.piano.sounds)
    
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
        # Mock the sound playing
        for note in self.piano.sounds:
            self.piano.sounds[note] = Mock()
        
        # Play note for finger 0
        self.piano._play_note(0)
        
        # Check that state is updated
        self.assertTrue(self.piano.finger_states[0])
        
        # Check that sound was played
        note = self.piano.NOTES[0]
        self.piano.sounds[note].play.assert_called_once()


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
