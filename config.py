#!/usr/bin/env python3
"""
Configuration management for Finger Piano application.
Handles saving and loading user settings.
"""

import json
import os
from typing import Dict, Any


class Config:
    """Manages application configuration and settings."""
    
    DEFAULT_CONFIG = {
        'camera_id': 0,
        'instrument': 'piano',
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.5,
        'trigger_threshold': 0.15,
    }
    
    INSTRUMENTS = ['piano', 'guitar', 'electric_guitar', 'violin']
    
    def __init__(self, config_file: str = 'finger_piano_config.json'):
        """Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.settings = self.load()
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Dictionary containing configuration settings
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**self.DEFAULT_CONFIG, **config}
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()
    
    def save(self) -> bool:
        """Save configuration to file.
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except IOError as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.settings[key] = value
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.settings = self.DEFAULT_CONFIG.copy()
