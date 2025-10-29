# Finger Piano 🎹✋

A virtual instrument program that recognizes finger joints as landmarks and plays piano sounds as your fingers move.

## Description

Finger Piano uses computer vision and hand tracking to create an interactive musical instrument. By detecting your hand landmarks through your webcam, it maps finger movements to piano keys, allowing you to play music with natural hand gestures.

### Features

- **Real-time hand tracking** using MediaPipe
- **Gesture-based piano playing** - fold fingers to play notes
- **Visual feedback** showing which fingers are active
- **Synthesized piano sounds** for a rich musical experience
- **5 fingers mapped to 5 piano notes** (C major scale: C4, D4, E4, F4, G4)

## Requirements

- Python 3.7 or higher
- Webcam
- Operating System: Windows, macOS, or Linux

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dongwandev/finger-piano.git
cd finger-piano
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the finger piano application:
```bash
python finger_piano.py
```

### How to Play

1. Position your hand in front of the webcam
2. Keep your palm facing the camera with fingers extended (this is the default state)
3. Fold/bend your fingers to play notes
4. Each finger corresponds to a different note:
   - **Thumb** → C4
   - **Index finger** → D4
   - **Middle finger** → E4
   - **Ring finger** → F4
   - **Pinky** → G4

5. Press 'q' to quit the application

### Tips

- Ensure good lighting for better hand detection
- Keep your hand clearly visible to the camera with fingers extended as the default state
- Fold fingers deliberately for best note triggering
- Experiment with different hand positions and movements

## Technical Details

### Technology Stack

- **OpenCV**: Camera input and image processing
- **MediaPipe**: Hand landmark detection and tracking
- **Pygame**: Audio playback system
- **NumPy**: Signal processing for sound synthesis

### How It Works

1. **Hand Detection**: MediaPipe detects hand landmarks in real-time from webcam feed
2. **Finger Tracking**: Tracks finger tip positions (landmarks 4, 8, 12, 16, 20)
3. **Movement Detection**: Monitors vertical movement of finger tips
4. **Note Triggering**: When a finger folds (moves up) past a threshold, plays the corresponding note
5. **Sound Synthesis**: Generates piano-like sounds using sine waves with ADSR envelope

## License

MIT License - see [LICENSE](LICENSE) file for details

## Author

Hong Dongwan

## Acknowledgments

- MediaPipe by Google for hand tracking
- Pygame community for audio support