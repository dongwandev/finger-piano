# Finger Piano 🎹✋

A virtual instrument program that recognizes finger joints as landmarks and plays piano sounds as your fingers move.

## Description

Finger Piano uses computer vision and hand tracking to create an interactive musical instrument. By detecting your hand landmarks through your webcam, it maps finger movements to piano keys, allowing you to play music with natural hand gestures.

The application now features a multi-screen GUI with lobby, settings, and play screens for an enhanced user experience.

### Features

- **Multi-screen GUI** with easy navigation
  - 🏠 **Lobby Screen** - Main menu to start playing or adjust settings
  - ⚙️ **Settings Screen** - Configure camera, instrument, and detection sensitivity
  - 🎮 **Play Screen** - Interactive performance interface
- **Real-time hand tracking** using MediaPipe
- **Gesture-based piano playing** - move fingers down to play notes
- **Visual feedback** showing which fingers are active
- **Synthesized piano sounds** for a rich musical experience
- **5 fingers mapped to 5 chords** (C Major, G Major, A Minor, F Major, D Major)
- **Configurable settings** with persistent storage
- **Multiple instrument support** (Piano, Guitar, Electric Guitar, Violin)
- **Adjustable sensitivity** for detection and tracking confidence

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

### Navigating the Application

#### 🏠 Lobby Screen
When you start the application, you'll see the lobby screen with the following options:
- **Start Playing** - Begin performing with the finger piano
- **Settings** - Configure application settings
- **Quit** - Exit the application

**Navigation:**
- Use **UP/DOWN arrow keys** or **W/S** to navigate menu options
- Press **ENTER** or **SPACE** to select an option
- Press **1** to quickly start playing, **2** for settings
- Press **Q** or **ESC** to quit

#### ⚙️ Settings Screen
Configure the application to your preferences:
- **Camera ID** - Select camera device (0: built-in, 1-2: USB cameras)
- **Test Camera** - Preview the selected camera feed
- **Instrument** - Choose between Piano, Guitar, Electric Guitar, or Violin
- **Detection Confidence** - Adjust hand detection sensitivity (0.5-0.9)
- **Tracking Confidence** - Adjust hand tracking sensitivity (0.3-0.7)
- **Trigger Threshold** - Adjust finger press sensitivity (0.03-0.07)

**Navigation:**
- Use **UP/DOWN arrow keys** or **W/S** to navigate options
- Use **LEFT/RIGHT arrow keys** or **A/D** to change values
- Press **ENTER** or **SPACE** to activate options (e.g., test camera)
- Select **Save & Return** to save changes and return to lobby
- Select **Cancel** or press **Q/ESC** to return without saving

#### 🎮 Play Screen
The performance interface shows:
- **Camera feed** with hand tracking visualization
- **Top-left corner** - Finger status (which fingers are active)
- **Top-right corner** - Current instrument selection

**Navigation:**
- Press **Q** or **ESC** to return to the lobby

### How to Play

1. Start the application and select **Start Playing** from the lobby
2. Position your hand in front of the webcam
3. Keep your palm facing the camera
4. Move your fingers downward (like pressing piano keys) to play chords
5. Each finger corresponds to a different chord:
   - **Thumb** → C Major (C, E, G)
   - **Index finger** → G Major (G, B, D)
   - **Middle finger** → A Minor (A, C, E)
   - **Ring finger** → F Major (F, A, C)
   - **Pinky** → D Major (D, F#, A)

6. Press **Q** or **ESC** to return to the lobby

### Tips

- Ensure good lighting for better hand detection
- Keep your hand clearly visible to the camera
- Move fingers deliberately for best note triggering
- Experiment with different hand positions and movements
- Adjust detection and tracking confidence in settings if hand detection is unstable
- Adjust trigger threshold in settings to change how much finger movement is needed to trigger notes
- Test your camera in the settings screen before playing to ensure it's working properly

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
4. **Chord Triggering**: When a finger moves down past a threshold, plays the corresponding chord
5. **Sound Synthesis**: Generates piano-like chord sounds by combining multiple sine waves with ADSR envelope
6. **Configuration Management**: Saves user preferences to a JSON file for persistence across sessions
7. **Multi-Screen GUI**: Provides intuitive navigation between lobby, settings, and play screens

## Configuration

Settings are automatically saved to `finger_piano_config.json` in the application directory. The configuration file includes:

- **camera_id**: Camera device ID (default: 0)
- **instrument**: Selected instrument (default: 'piano')
- **min_detection_confidence**: Hand detection confidence threshold (default: 0.7)
- **min_tracking_confidence**: Hand tracking confidence threshold (default: 0.5)
- **trigger_threshold**: Finger movement threshold for note triggering (default: 0.05)

You can manually edit this file if needed, or use the Settings screen in the application.

## License

MIT License - see [LICENSE](LICENSE) file for details

## Author

Hong Dongwan

## Acknowledgments

- MediaPipe by Google for hand tracking
- Pygame community for audio support