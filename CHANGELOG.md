# Changelog

## [Unreleased] - Multi-Screen GUI

### Added
- **Multi-screen GUI system** with three main screens:
  - 🏠 Lobby Screen: Main menu interface
  - ⚙️ Settings Screen: Configuration interface
  - 🎮 Play Screen: Performance interface
  
- **Configuration Management**:
  - JSON-based persistent configuration storage
  - Camera ID selection (support for multiple cameras)
  - Instrument selection (Piano, Guitar, Electric Guitar, Violin)
  - Adjustable detection confidence (0.5-0.9)
  - Adjustable tracking confidence (0.3-0.7)
  - Adjustable trigger threshold (0.03-0.07)
  
- **Camera Testing**: Preview camera feed before playing

- **Enhanced Visual Feedback**:
  - Finger status display in play screen
  - Current instrument indicator
  - Professional GUI design with clear navigation

### Changed
- Updated `finger_piano.py` to use configuration-based initialization
- Enhanced README with detailed usage instructions for new GUI
- Updated tests to support new configuration system

### Technical Details
- New modules: `config.py`, `screens.py`
- All tests passing (11 tests)
- No security vulnerabilities detected
- Code review completed with no issues
