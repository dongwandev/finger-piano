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
- **Gesture-based piano playing** - bend/curl fingers to play chords
- **Improved finger bending detection** - works reliably for all fingers including thumb
- **Visual feedback** showing which fingers are active with flexion percentage display
- **Synthesized piano sounds** for a rich musical experience
- **Customizable chord presets** - Choose from 4 different chord configurations
  - Default preset with 5 chords (all fingers)
  - 3 alternative presets with 3-4 chords (some fingers unassigned)
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
- **Chord Preset** - Select chord configuration for each finger:
  - **Default** - C Major, G Major, A Minor, F Major, D Major (all fingers assigned)
  - **Preset 1** - G Major, D Major, E Minor, C Major (pinky unassigned)
  - **Preset 2** - A Minor, C Major, G Major, D Major (pinky unassigned)
  - **Preset 3** - E Major, A Minor, A Sus4 (ring and pinky unassigned)
  - *Note: Unassigned fingers produce no sound when bent*
- **Detection Confidence** - Adjust hand detection sensitivity (0.5-0.9)
- **Tracking Confidence** - Adjust hand tracking sensitivity (0.3-0.7)
- **Sensitivity** - Adjust finger bending sensitivity (0.10-0.30)
  - **Lower values = More sensitive** (easier to trigger chords)
  - **Higher values = Less sensitive** (requires more finger bending)

**Navigation:**
- Use **UP/DOWN arrow keys** or **W/S** to navigate options
- Use **LEFT/RIGHT arrow keys** or **A/D** to change values
- Press **ENTER** or **SPACE** to activate options (e.g., test camera)
- Select **Save & Return** to save changes and return to lobby
- Select **Cancel** or press **Q/ESC** to return without saving

#### 🎮 Play Screen
The performance interface shows:
- **Camera feed** with hand tracking visualization
- **Top-left corner** - Finger status showing:
  - Which fingers are active (green when playing, gray when inactive)
  - The chord assigned to each finger
  - Real-time flexion percentage (0-100%) indicating how much each finger is bent
- **Top-right corner** - Current instrument selection

**Navigation:**
- Press **Q** or **ESC** to return to the lobby

### How to Play

1. Start the application and select **Start Playing** from the lobby
2. Position your hand in front of the webcam
3. Keep your palm facing the camera
4. **Bend/curl your fingers** (like pressing piano keys) to play chords
5. Each finger corresponds to a different chord (default preset):
   - **Thumb** → C Major (C, E, G)
   - **Index finger** → G Major (G, B, D)
   - **Middle finger** → A Minor (A, C, E)
   - **Ring finger** → F Major (F, A, C)
   - **Pinky** → D Major (D, F#, A)
   - *Note: Chord assignments can be changed in Settings using different presets*

6. Press **Q** or **ESC** to return to the lobby

### Tips

- Ensure good lighting for better hand detection
- Keep your hand clearly visible to the camera
- **Curl/bend your fingers deliberately** for best chord triggering
- Experiment with different hand positions and movements
- The thumb now works reliably with the bending detection algorithm
- Adjust detection and tracking confidence in settings if hand detection is unstable
- **Adjust sensitivity in settings** to control chord triggering:
  - If chords trigger too easily (too sensitive): **increase** the sensitivity value
  - If chords don't trigger easily enough: **decrease** the sensitivity value
- **Try different chord presets** in settings to explore various chord progressions
- Test your camera in the settings screen before playing to ensure it's working properly

## Technical Details

### Technology Stack

- **OpenCV**: Camera input and image processing
- **MediaPipe**: Hand landmark detection and tracking
- **Pygame**: Audio playback system
- **NumPy**: Signal processing for sound synthesis

### How It Works

1. **Hand Detection**: MediaPipe detects hand landmarks in real-time from webcam feed
2. **Finger Tracking**: Tracks finger joint positions (fingertips and MCP knuckles)
3. **Bending Detection**: Measures 3D Euclidean distance between fingertip and MCP joint for each finger
4. **Flexion Calculation**: Calculates flexion percentage (0-100%) based on current vs maximum extension
5. **Chord Triggering**: When a finger bends (distance decreases by threshold %), plays the corresponding chord
6. **Visual Feedback**: Displays real-time finger status with chord assignments and flexion percentages
7. **Sound Synthesis**: Generates piano-like chord sounds by combining multiple sine waves with ADSR envelope
8. **Configuration Management**: Saves user preferences to a JSON file for persistence across sessions
9. **Multi-Screen GUI**: Provides intuitive navigation between lobby, settings, and play screens

### Algorithm Details

The **finger bending detection** algorithm:
- Calculates the 3D distance between each fingertip and its corresponding MCP (knuckle) joint
- When a finger curls/bends, this distance decreases
- Triggers a chord when the relative distance change exceeds the threshold (default: 15%)
- Works for all fingers including the thumb (which moves primarily in x-axis)
- More reliable than the previous y-axis movement detection

The **flexion percentage calculation**:
- Tracks the maximum extension (fully extended position) for each finger
- Calculates flexion as: `(max_extension - current_extension) / max_extension × 100%`
- 0% = fully extended (straight finger)
- 100% = fully flexed (completely bent/curled finger)
- Updates in real-time and displays alongside each finger's status

## Configuration

Settings are automatically saved to `finger_piano_config.json` in the application directory. The configuration file includes:

- **camera_id**: Camera device ID (default: 0)
- **instrument**: Selected instrument (default: 'piano')
- **chord_preset**: Selected chord preset (default: 'default', options: 'default', 'preset1', 'preset2', 'preset3')
- **min_detection_confidence**: Hand detection confidence threshold (default: 0.7)
- **min_tracking_confidence**: Hand tracking confidence threshold (default: 0.5)
- **trigger_threshold**: Finger bending sensitivity for chord triggering (default: 0.15, range: 0.10-0.30)
  - Lower values = more sensitive (chords trigger more easily)
  - Higher values = less sensitive (requires more finger bending to trigger chords)

You can manually edit this file if needed, or use the Settings screen in the application.

## License

MIT License - see [LICENSE](LICENSE) file for details

## Author

Hong Dongwan

## Acknowledgments

- MediaPipe by Google for hand tracking
- Pygame community for audio support