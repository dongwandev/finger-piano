# Finger Piano Examples and Technical Details

## Quick Start Example

```python
from finger_piano import FingerPiano

# Create and run the finger piano
piano = FingerPiano()
piano.run()
```

## Advanced Usage

### Custom Camera Selection

```python
from finger_piano import FingerPiano

# Use camera device 1 instead of default 0
piano = FingerPiano(camera_id=1)
piano.run()
```

### Understanding the Hand Tracking

The application uses MediaPipe's hand landmark detection which identifies 21 key points on your hand:

- Wrist (0)
- Thumb: 1, 2, 3, 4 (tip)
- Index: 5, 6, 7, 8 (tip)
- Middle: 9, 10, 11, 12 (tip)
- Ring: 13, 14, 15, 16 (tip)
- Pinky: 17, 18, 19, 20 (tip)

The finger piano specifically tracks the 5 fingertip landmarks (4, 8, 12, 16, 20).

## How the Music Works

### Note Triggering

Notes are triggered based on vertical finger movement:

1. **Detection**: The system monitors the Y-coordinate of each fingertip
2. **Threshold**: When a finger moves down by more than 5% of screen height
3. **Trigger**: The corresponding piano note plays
4. **Reset**: When the finger moves up, it resets and can trigger again

### Sound Synthesis

Each piano note is synthesized using:
- **Base frequency**: Standard piano frequencies (e.g., A4 = 440 Hz)
- **ADSR Envelope**: 
  - Attack: 50ms rise time
  - Decay: 50ms fall to sustain
  - Sustain: 300ms at 70% volume
  - Release: 100ms fade out
- **Harmonics**: Additional overtones at 2x and 3x frequency for richer sound

### Note Mapping

| Finger | Landmark ID | Note | Frequency |
|--------|-------------|------|-----------|
| Thumb  | 4           | C4   | 261.63 Hz |
| Index  | 8           | D4   | 293.66 Hz |
| Middle | 12          | E4   | 329.63 Hz |
| Ring   | 16          | F4   | 349.23 Hz |
| Pinky  | 20          | G4   | 392.00 Hz |

## Performance Tips

### Optimal Conditions

- **Lighting**: Bright, even lighting works best
- **Background**: Plain background helps hand detection
- **Distance**: Keep hand 30-60cm from camera
- **Position**: Hand should fill about 30-50% of frame

### Improving Response

If notes aren't triggering reliably:

1. **Adjust lighting**: Ensure hand is well-lit
2. **Check hand visibility**: Make sure all fingers are visible
3. **Increase movement**: Make more deliberate finger motions
4. **Check camera**: Ensure camera is focused properly

## Technical Architecture

```
Camera Input (OpenCV)
    ↓
Hand Detection (MediaPipe)
    ↓
Landmark Extraction
    ↓
Movement Analysis
    ↓
Note Triggering
    ↓
Sound Playback (Pygame)
```

## Customization Ideas

### Changing the Scale

Modify the `NOTES` and `NOTE_FREQUENCIES` in the `FingerPiano` class to use different scales:

```python
# Example: Pentatonic scale
NOTES = ['C4', 'D4', 'E4', 'G4', 'A4']
```

### Adjusting Sensitivity

Modify `trigger_threshold` in the `__init__` method:

```python
self.trigger_threshold = 0.03  # More sensitive (lower value)
self.trigger_threshold = 0.08  # Less sensitive (higher value)
```

### Adding More Notes

You could extend the system to use two hands (10 fingers) for more notes:

```python
# In __init__, change max_num_hands to track both hands
self.hands = self.mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Track both hands instead of 1
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
```

## Troubleshooting

### "Cannot open camera"
- Check if camera is being used by another application
- Try different camera_id values (0, 1, 2, etc.)
- Verify camera permissions

### No sound
- Check system volume
- Verify pygame mixer initialized correctly
- Try restarting the application

### Poor hand detection
- Improve lighting conditions
- Remove gloves or hand accessories
- Ensure hand contrasts with background

### High CPU usage
- Lower camera resolution in the code
- Reduce frame processing frequency
- Close other applications
