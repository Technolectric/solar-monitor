import os
import time
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from threading import Thread, Lock
from flask import Flask, render_template_string, request, jsonify
from collections import deque
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import logging

# Configure logging to see errors in Railway console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

print("🚀 Starting application initialization...", flush=True)

# ----------------------------
# Flask App & Config
# ----------------------------
app = Flask(__name__)

# API Configuration
API_URL = "https://openapi.growatt.com/v1/device/storage/storage_last_data"
TOKEN = os.getenv("API_TOKEN")
SERIAL_NUMBERS = os.getenv("SERIAL_NUMBERS", "").split(",")
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", 5))
DATA_FILE = "load_patterns.json"
HISTORY_FILE = "daily_history.json"
ML_MODEL_FILE = "appliance_models.pkl"

# Initialize Data Files immediately so they exist for Gunicorn
for file in [DATA_FILE, HISTORY_FILE, ML_MODEL_FILE]:
    if not Path(file).exists():
        Path(file).touch()
        print(f"Created empty file: {file}", flush=True)

# Inverter Mapping
INVERTER_CONFIG = {
    "RKG3B0400T": {"label": "Inverter 1", "type": "primary"},
    "KAM4N5W0AG": {"label": "Inverter 2", "type": "primary"},
    "JNK1CDR0KQ": {"label": "Inverter 3 (Backup)", "type": "backup"}
}

# System Physics Constants
PRIMARY_BATTERY_CAPACITY_WH = 30000 
BACKUP_BATTERY_DEGRADED_WH = 21000   
BACKUP_DEGRADATION = 0.70
SOLAR_EFFICIENCY_FACTOR = 0.85
TOTAL_SOLAR_CAPACITY_KW = 10
LATITUDE = -1.85238
LONGITUDE = 36.77683
EAT = timezone(timedelta(hours=3))

# Email Config
RESEND_API_KEY = os.getenv('RESEND_API_KEY')
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')

# Weather API Config
WEATHERAPI_KEY = os.getenv('WEATHERAPI_KEY')

# ----------------------------
# 1. Machine Learning Appliance Detection
# ----------------------------
class ApplianceDetector:
    def __init__(self):
        self.model_file = ML_MODEL_FILE
        self.load_history = deque(maxlen=1000)  # Store recent load patterns
        self.appliance_classifier = None
        self.scaler = StandardScaler()
        self.model_lock = Lock()
        self.load_model()
        
        # Comprehensive appliance categories for classification (100 appliances)
        self.APPLIANCE_CLASSES = [
            # Basic States
            'idle',
            'multiple_loads',
            
            # Kitchen Appliances (25)
            'microwave', 'electric_kettle', 'toaster', 'toaster_oven', 'coffee_maker',
            'rice_cooker', 'slow_cooker', 'pressure_cooker', 'blender', 'food_processor',
            'stand_mixer', 'hand_mixer', 'bread_maker', 'juicer', 'electric_grill',
            'sandwich_maker', 'waffle_maker', 'deep_fryer', 'electric_wok', 'hot_plate',
            'induction_cooker', 'garbage_disposal', 'electric_can_opener', 'food_warmer', 'egg_cooker',
            
            # Cooling/Refrigeration (8)
            'refrigerator', 'freezer', 'mini_fridge', 'wine_cooler', 'ice_maker',
            'chest_freezer', 'commercial_fridge', 'beverage_cooler',
            
            # Laundry & Cleaning (10)
            'washing_machine', 'dryer', 'iron', 'steamer', 'vacuum_cleaner',
            'carpet_cleaner', 'steam_mop', 'spin_dryer', 'garment_steamer', 'handheld_vacuum',
            
            # Heating & Comfort (10)
            'electric_fan', 'ceiling_fan', 'tower_fan', 'table_fan', 'box_fan',
            'exhaust_fan', 'humidifier', 'dehumidifier', 'electric_heater', 'space_heater',
            
            # Personal Care (10)
            'hair_dryer', 'hair_straightener', 'curling_iron', 'electric_shaver', 'toothbrush_charger',
            'hair_clippers', 'facial_steamer', 'hot_comb', 'electric_nail_file', 'foot_spa',
            
            # Electronics & Entertainment (15)
            'desktop_computer', 'laptop', 'gaming_console', 'tv_32', 'tv_55',
            'tv_plasma', 'sound_system', 'stereo', 'bluetooth_speaker', 'projector',
            'dvd_player', 'cable_box', 'satellite_decoder', 'gaming_pc', 'monitor',
            
            # Lighting (8)
            'led_bulbs', 'led_strips', 'fluorescent_tubes', 'cfl_bulbs', 'halogen_bulbs',
            'incandescent_bulbs', 'desk_lamp', 'floor_lamp',
            
            # Office & Work (8)
            'printer', 'scanner', 'fax_machine', 'paper_shredder', 'laminator',
            'label_maker', 'pencil_sharpener', 'photocopier',
            
            # Utility & Infrastructure (6)
            'water_pump', 'pool_pump', 'borehole_pump', 'sump_pump', 'router', 'security_system'
        ]
        
        # Training data for initial model (load, duration, spike_rate, variance) -> class
        self.training_data = []
        self.training_labels = []
        
        # Feature window for pattern analysis
        self.feature_window = 10  # Analyze last 10 data points
        
    def load_model(self):
        """Load trained ML models if they exist"""
        if Path(self.model_file).exists():
            try:
                with self.model_lock:
                    # Check if file is empty
                    if os.path.getsize(self.model_file) > 0:
                        models = joblib.load(self.model_file)
                        self.appliance_classifier = models.get('appliance_classifier')
                        self.scaler = models.get('scaler', StandardScaler())
                        self.training_data = models.get('training_data', [])
                        self.training_labels = models.get('training_labels', [])
                        print("✅ Loaded ML appliance models from disk", flush=True)
                    else:
                        print("⚠️ ML model file exists but is empty. initializing default.", flush=True)
                        self.init_default_models()
            except Exception as e:
                print(f"⚠️ Failed to load ML models (reinitializing): {e}", flush=True)
                self.init_default_models()
        else:
            self.init_default_models()
    
    def init_default_models(self):
        """Initialize default models with seed training data"""
        self.appliance_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        
        # Seed with comprehensive patterns for common appliances
        # Features: [mean_load, std, max, min, median, rate_change, change_std, max_spike, min_spike, spike_count, coef_var, time_factor]
        seed_data = [
            # Idle (0-150W)
            [50, 20, 100, 10, 45, 5, 10, 20, -15, 0, 0.4, 0.7],
            [30, 15, 80, 5, 28, 3, 8, 15, -10, 0, 0.5, 0.7],
            [75, 25, 120, 20, 70, 8, 12, 30, -20, 0, 0.33, 0.5],
            
            # Multiple loads (complex, high variance)
            [3500, 600, 4200, 2800, 3400, 200, 400, 800, -600, 3, 0.17, 0.8],
            [4000, 700, 5000, 3000, 3900, 250, 450, 900, -700, 4, 0.18, 0.75],
            [3800, 650, 4500, 2900, 3700, 220, 420, 850, -650, 3, 0.17, 0.8],
            
            # Kitchen - Microwave (800-1500W, short bursts)
            [1200, 150, 1400, 1000, 1180, 100, 80, 300, -250, 2, 0.13, 1.0],
            [1100, 130, 1300, 900, 1090, 90, 70, 280, -230, 2, 0.12, 1.0],
            
            # Kitchen - Electric Kettle (1500-2200W, high spike, short)
            [2100, 200, 2200, 1900, 2080, 150, 120, 400, -350, 3, 0.10, 0.7],
            [2000, 180, 2150, 1850, 1980, 140, 110, 380, -330, 2, 0.09, 0.7],
            [1900, 170, 2100, 1750, 1880, 130, 100, 360, -310, 3, 0.09, 1.0],
            
            # Kitchen - Toaster (800-1500W, short)
            [1100, 120, 1250, 950, 1080, 70, 60, 200, -180, 1, 0.11, 1.0],
            [1000, 110, 1150, 850, 980, 60, 50, 180, -160, 1, 0.11, 1.0],
            
            # Kitchen - Rice Cooker (400-1000W, sustained then low)
            [700, 200, 950, 400, 650, 50, 120, 300, -350, 1, 0.29, 0.7],
            [650, 180, 900, 350, 600, 45, 110, 280, -320, 1, 0.28, 0.7],
            
            # Kitchen - Blender (300-1000W, very short bursts)
            [600, 250, 900, 300, 550, 200, 180, 500, -450, 4, 0.42, 0.8],
            [550, 230, 850, 280, 520, 180, 160, 480, -420, 4, 0.42, 0.8],
            
            # Kitchen - Coffee Maker (800-1400W, short sustained)
            [1100, 80, 1200, 1000, 1090, 30, 40, 120, -100, 1, 0.07, 1.0],
            [1050, 75, 1150, 950, 1040, 28, 38, 110, -95, 1, 0.07, 1.0],
            
            # Refrigerator (100-300W, cycling compressor)
            [180, 60, 250, 100, 170, 20, 40, 100, -120, 2, 0.33, 0.7],
            [200, 65, 270, 110, 190, 22, 42, 110, -130, 2, 0.33, 0.7],
            [160, 55, 230, 90, 155, 18, 38, 95, -110, 2, 0.34, 0.7],
            
            # Freezer (150-400W, cycling)
            [250, 80, 350, 150, 240, 30, 50, 140, -150, 2, 0.32, 0.7],
            [280, 85, 380, 160, 270, 32, 52, 150, -160, 2, 0.30, 0.7],
            
            # Washing Machine (500-2000W, cycling phases)
            [1200, 400, 1800, 600, 1150, 150, 250, 600, -550, 3, 0.33, 0.8],
            [1100, 380, 1700, 550, 1050, 140, 240, 580, -530, 3, 0.35, 0.8],
            [1300, 420, 1900, 650, 1250, 160, 260, 620, -570, 3, 0.32, 0.7],
            
            # Iron (1000-1800W, sustained with variations)
            [1400, 180, 1650, 1100, 1380, 60, 100, 280, -250, 1, 0.13, 0.8],
            [1300, 170, 1550, 1000, 1280, 55, 95, 260, -230, 1, 0.13, 0.8],
            
            # Vacuum Cleaner (1000-1400W, intermittent)
            [1200, 120, 1350, 1050, 1180, 80, 70, 200, -180, 2, 0.10, 0.8],
            [1150, 115, 1300, 1000, 1130, 75, 68, 190, -170, 2, 0.10, 0.8],
            
            # Electric Fan (50-100W, sustained)
            [70, 10, 85, 55, 68, 3, 5, 12, -10, 0, 0.14, 0.8],
            [65, 9, 80, 50, 63, 2, 4, 10, -9, 0, 0.14, 0.8],
            [80, 12, 95, 65, 78, 4, 6, 14, -12, 0, 0.15, 1.0],
            
            # Ceiling Fan (50-120W, sustained)
            [85, 15, 110, 60, 82, 5, 8, 18, -15, 0, 0.18, 0.8],
            [90, 16, 115, 65, 88, 6, 9, 20, -17, 0, 0.18, 0.8],
            
            # Hair Dryer (1500-2000W, short bursts)
            [1750, 180, 1950, 1500, 1730, 120, 100, 320, -280, 2, 0.10, 1.0],
            [1700, 170, 1900, 1450, 1680, 115, 95, 310, -270, 2, 0.10, 1.0],
            
            # Desktop Computer (200-500W, sustained)
            [300, 40, 380, 220, 295, 15, 25, 60, -50, 1, 0.13, 0.8],
            [350, 45, 430, 270, 345, 18, 28, 65, -55, 1, 0.13, 0.8],
            
            # Laptop (50-100W, sustained)
            [70, 15, 95, 45, 68, 8, 10, 25, -22, 0, 0.21, 0.8],
            [65, 14, 90, 40, 63, 7, 9, 23, -20, 0, 0.22, 0.8],
            
            # TV 32" LED (50-100W, sustained)
            [75, 12, 95, 55, 73, 5, 8, 18, -15, 0, 0.16, 0.8],
            [70, 11, 90, 50, 68, 4, 7, 16, -14, 0, 0.16, 1.0],
            
            # TV 55" LED (100-200W, sustained)
            [140, 20, 175, 105, 138, 8, 12, 30, -28, 0, 0.14, 0.8],
            [150, 22, 185, 115, 148, 9, 13, 32, -30, 0, 0.15, 1.0],
            
            # LED Bulbs (10-20W, constant)
            [15, 2, 18, 12, 15, 1, 1, 3, -2, 0, 0.13, 0.8],
            [12, 1, 14, 10, 12, 0, 1, 2, -2, 0, 0.08, 1.0],
            
            # Fluorescent Tubes (30-60W, constant)
            [45, 5, 52, 38, 44, 2, 3, 7, -6, 0, 0.11, 0.8],
            [50, 6, 58, 42, 49, 2, 3, 8, -7, 0, 0.12, 1.0],
            
            # Printer (300-500W, short intermittent)
            [400, 120, 520, 280, 390, 80, 70, 180, -160, 2, 0.30, 0.8],
            [380, 115, 500, 260, 370, 75, 68, 170, -150, 2, 0.30, 0.8],
            
            # Water Pump (750-1500W, intermittent cycling)
            [1100, 200, 1400, 800, 1080, 100, 120, 350, -320, 2, 0.18, 0.7],
            [1000, 180, 1300, 700, 980, 90, 110, 330, -300, 2, 0.18, 0.7],
            [1200, 220, 1500, 900, 1180, 110, 130, 370, -340, 2, 0.18, 0.8],
            
            # Pool Pump (1000-1350W, sustained, very stable)
            [1200, 50, 1300, 1100, 1190, 10, 30, 80, -70, 0, 0.04, 0.7],
            [1150, 45, 1250, 1050, 1140, 8, 28, 75, -65, 0, 0.04, 0.7],
            [1250, 55, 1350, 1150, 1240, 12, 32, 85, -75, 0, 0.04, 0.8],
            
            # Router/Modem (10-30W, constant)
            [18, 3, 22, 14, 18, 1, 2, 4, -3, 0, 0.17, 0.7],
            [20, 3, 24, 16, 20, 1, 2, 4, -4, 0, 0.15, 0.7],
            
            # Security System (20-50W, constant)
            [35, 5, 42, 28, 34, 2, 3, 7, -6, 0, 0.14, 0.7],
            [32, 4, 38, 26, 31, 1, 2, 6, -5, 0, 0.13, 0.7],
        ]
        
        seed_labels = [
            # Idle
            'idle', 'idle', 'idle',
            # Multiple
            'multiple_loads', 'multiple_loads', 'multiple_loads',
            # Kitchen
            'microwave', 'microwave',
            'electric_kettle', 'electric_kettle', 'electric_kettle',
            'toaster', 'toaster',
            'rice_cooker', 'rice_cooker',
            'blender', 'blender',
            'coffee_maker', 'coffee_maker',
            # Cooling
            'refrigerator', 'refrigerator', 'refrigerator',
            'freezer', 'freezer',
            # Laundry
            'washing_machine', 'washing_machine', 'washing_machine',
            'iron', 'iron',
            'vacuum_cleaner', 'vacuum_cleaner',
            # Comfort
            'electric_fan', 'electric_fan', 'electric_fan',
            'ceiling_fan', 'ceiling_fan',
            # Personal Care
            'hair_dryer', 'hair_dryer',
            # Electronics
            'desktop_computer', 'desktop_computer',
            'laptop', 'laptop',
            'tv_32', 'tv_32',
            'tv_55', 'tv_55',
            # Lighting
            'led_bulbs', 'led_bulbs',
            'fluorescent_tubes', 'fluorescent_tubes',
            # Office
            'printer', 'printer',
            # Utility
            'water_pump', 'water_pump', 'water_pump',
            'pool_pump', 'pool_pump', 'pool_pump',
            'router', 'router',
            'security_system', 'security_system',
        ]
        
        self.training_data = seed_data
        self.training_labels = seed_labels
        
        # Train initial model
        X = np.array(seed_data)
        y = np.array(seed_labels)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.appliance_classifier.fit(X_scaled, y)
        
        print("✅ Initialized ML appliance models with seed data", flush=True)
    
    def save_model(self):
        """Save trained models to disk"""
        try:
            with self.model_lock:
                models = {
                    'appliance_classifier': self.appliance_classifier,
                    'scaler': self.scaler,
                    'training_data': self.training_data,
                    'training_labels': self.training_labels
                }
                joblib.dump(models, self.model_file)
            print("💾 Saved ML appliance models to disk", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to save ML models: {e}", flush=True)
    
    def extract_features(self, load_data, time_data=None):
        """
        Extract features from load data for appliance classification.
        Features capture the signature of different appliances.
        """
        features = []
        
        if len(load_data) == 0:
            return [0] * 12
        
        # Basic statistical features
        features.append(np.mean(load_data))      # Mean load
        features.append(np.std(load_data))       # Variability
        features.append(np.max(load_data))       # Peak load
        features.append(np.min(load_data))       # Minimum load
        features.append(np.median(load_data))    # Median
        
        # Rate of change features (detect spikes vs gradual)
        if len(load_data) > 1:
            changes = np.diff(load_data)
            features.append(np.mean(changes))     # Average rate of change
            features.append(np.std(changes))      # Change volatility
            features.append(np.max(changes))      # Max spike up
            features.append(np.min(changes))      # Max spike down
            
            # Spike detection (sudden changes > 500W)
            large_changes = np.abs(changes) > 500
            features.append(np.sum(large_changes))  # Spike count
            
            # Pattern stability (low = stable like pool pump, high = cycling like AC)
            if np.mean(load_data) > 0:
                features.append(np.std(load_data) / np.mean(load_data))  # Coefficient of variation
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Time-based feature (if available)
        if time_data:
            hour = time_data.hour
            features.append(1.0 if 6 <= hour <= 22 else 0.5)  # Day vs night factor
        else:
            features.append(0.7)  # Default
        
        # Pad or trim to 12 features
        while len(features) < 12:
            features.append(0)
        
        return features[:12]
    
    def detect_appliances(self, current_load, previous_load=0):
        """
        ML-based appliance detection using pattern recognition.
        Returns: List of detected appliance names with confidence scores
        """
        try:
            now = datetime.now(EAT)
            
            # Add current load to history
            self.load_history.append({
                'timestamp': now,
                'load': current_load
            })
            
            # Need enough data for pattern analysis
            if len(self.load_history) < self.feature_window:
                return self._simple_fallback_detection(current_load)
            
            # Get recent load pattern
            recent_loads = [item['load'] for item in list(self.load_history)[-self.feature_window:]]
            
            # Extract features from pattern
            features = self.extract_features(recent_loads, now)
            
            # Use ML classifier to predict appliance type
            try:
                features_scaled = self.scaler.transform([features])
                predicted_class = self.appliance_classifier.predict(features_scaled)[0]
                confidence = np.max(self.appliance_classifier.predict_proba(features_scaled))
                
                # Format output
                detected = []
                
                # Only report if confidence is reasonable
                if confidence > 0.4:
                    appliance_name = predicted_class.replace('_', ' ').title()
                    detected.append(f"{appliance_name}")
                    
                    # Check for multiple appliances (high load + high variance)
                    if current_load > 3000 and np.std(recent_loads) > 500:
                        if predicted_class != 'multiple_loads':
                            detected.append("+ Other Loads")
                else:
                    # Low confidence - use simple detection
                    detected = self._simple_fallback_detection(current_load)
                
                return detected
                
            except Exception as e:
                print(f"⚠️ ML classification error: {e}", flush=True)
                return self._simple_fallback_detection(current_load)
            
        except Exception as e:
            print(f"⚠️ Appliance detection error: {e}", flush=True)
            return ["System Error"]
    
    def _simple_fallback_detection(self, current_load):
        """Simple rule-based fallback when ML is not ready"""
        if current_load < 50:
            return ["Idle"]
        elif 50 <= current_load < 150:
            return ["Router/Lights/Chargers"]
        elif 150 <= current_load < 400:
            return ["Refrigerator/Freezer/Lights"]
        elif 400 <= current_load < 800:
            return ["TV/Computer/Fans"]
        elif 800 <= current_load < 1100:
            return ["Microwave/Vacuum/Iron"]
        elif 1100 <= current_load < 1400:
            return ["Pool Pump/Washing Machine"]
        elif 1400 <= current_load < 1800:
            return ["Iron/Hair Dryer/Kettle"]
        elif 1800 <= current_load < 2500:
            return ["Kettle/Toaster/Cooking"]
        elif 2500 <= current_load < 3500:
            return ["Heavy Cooking/Multiple Loads"]
        else:
            return ["Multiple Appliances"]
    
    def train_from_feedback(self, feedback_data):
        """
        Improve models based on user feedback.
        feedback_data: {
            'timestamp': str, 
            'actual_appliance': str,  # e.g., 'kettle', 'pool_pump'
            'load_pattern': list,     # Recent load readings
            'current_load': float
        }
        """
        try:
            if not feedback_data.get('actual_appliance') or not feedback_data.get('load_pattern'):
                return
            
            # Extract features from the corrected pattern
            load_pattern = feedback_data['load_pattern']
            timestamp = datetime.fromisoformat(feedback_data['timestamp']) if isinstance(feedback_data['timestamp'], str) else feedback_data['timestamp']
            features = self.extract_features(load_pattern, timestamp)
            
            # Add to training data
            self.training_data.append(features)
            self.training_labels.append(feedback_data['actual_appliance'])
            
            # Keep training set manageable (last 500 examples)
            if len(self.training_data) > 500:
                self.training_data = self.training_data[-500:]
                self.training_labels = self.training_labels[-500:]
            
            # Retrain model if we have enough diverse data
            if len(self.training_data) >= 20:
                try:
                    X = np.array(self.training_data)
                    y = np.array(self.training_labels)
                    
                    # Refit scaler and classifier
                    self.scaler.fit(X)
                    X_scaled = self.scaler.transform(X)
                    self.appliance_classifier.fit(X_scaled, y)
                    
                    # Save updated model
                    self.save_model()
                    
                    print(f"✅ ML model retrained with {len(self.training_data)} examples", flush=True)
                except Exception as e:
                    print(f"⚠️ Model retraining failed: {e}", flush=True)
            
            # Also save feedback to file for analysis
            feedback_file = "ml_feedback.json"
            feedback_history = []
            
            if Path(feedback_file).exists():
                try:
                    with open(feedback_file, 'r') as f:
                        if os.path.getsize(feedback_file) > 0:
                            feedback_history = json.load(f)
                except:
                    pass
            
            feedback_history.append({
                'timestamp': feedback_data['timestamp'] if isinstance(feedback_data['timestamp'], str) else feedback_data['timestamp'].isoformat(),
                'actual_appliance': feedback_data['actual_appliance'],
                'current_load': feedback_data.get('current_load', 0)
            })
            
            # Keep only recent feedback
            if len(feedback_history) > 200:
                feedback_history = feedback_history[-200:]
            
            with open(feedback_file, 'w') as f:
                json.dump(feedback_history, f)
            
            print(f"📝 Saved ML feedback: {feedback_data['actual_appliance']}", flush=True)
            
        except Exception as e:
            print(f"⚠️ Feedback training error: {e}", flush=True)

# Initialize ML detector
ml_detector = ApplianceDetector()

# ----------------------------
# 2. Logic Engine: Persistence & Detection
# ----------------------------
class PersistentLoadManager:
    def __init__(self, filename):
        self.filename = filename
        self.patterns = self.load_data()

    def load_data(self):
        if Path(self.filename).exists():
            try:
                with open(self.filename, 'r') as f:
                    # Check for empty file
                    if os.path.getsize(self.filename) == 0:
                        return {"weekday": {str(h): [] for h in range(24)}, "weekend": {str(h): [] for h in range(24)}}
                    return json.load(f)
            except Exception as e:
                print(f"Error loading load patterns: {e}", flush=True)
        return {"weekday": {str(h): [] for h in range(24)}, "weekend": {str(h): [] for h in range(24)}}

    def save_data(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.patterns, f)
        except: pass

    def update(self, load_watts):
        now = datetime.now(EAT)
        day_type = "weekend" if now.weekday() >= 5 else "weekday"
        hour = str(now.hour)
        self.patterns[day_type][hour].append(load_watts)
        if len(self.patterns[day_type][hour]) > 100:
            self.patterns[day_type][hour] = self.patterns[day_type][hour][-100:]

    def get_forecast(self, hours_ahead=24):
        forecast = []
        now = datetime.now(EAT)
        for i in range(hours_ahead):
            ft = now + timedelta(hours=i)
            day_type = "weekend" if ft.weekday() >= 5 else "weekday"
            hour = str(ft.hour)
            history = self.patterns[day_type][hour]
            est = sum(history) / len(history) if history else (2500 if 18<=ft.hour<=21 else 1000)
            forecast.append({'time': ft, 'estimated_load': est})
        return forecast

class DailyHistoryManager:
    def __init__(self, filename):
        self.filename = filename
        self.history = self.load_history()
        self.hourly_data = []  

    def load_history(self):
        if Path(self.filename).exists():
            try:
                with open(self.filename, 'r') as f:
                    if os.path.getsize(self.filename) == 0:
                        return {}
                    return json.load(f)
            except: pass
        return {}

    def save_history(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.history, f)
        except: pass

    def add_hourly_datapoint(self, timestamp, load_w, battery_discharge_w, solar_w):
        self.hourly_data.append({
            'timestamp': timestamp.isoformat(),
            'load': load_w,
            'battery_discharge': battery_discharge_w,
            'solar': solar_w
        })
        if len(self.hourly_data) > 288:
            self.hourly_data = self.hourly_data[-288:]

    def get_last_24h_data(self):
        return self.hourly_data

    def update_daily(self, date_str, total_consumption_wh, total_solar_wh, actual_irradiance_wh):
        """
        Updated to use actual irradiance instead of theoretical maximum
        actual_irradiance_wh: Sum of hourly irradiance * panel capacity for the day
        """
        if date_str not in self.history:
            self.history[date_str] = {
                'consumption': 0,
                'solar': 0,
                'potential': 0
            }
        self.history[date_str]['consumption'] = total_consumption_wh
        self.history[date_str]['solar'] = total_solar_wh
        self.history[date_str]['potential'] = actual_irradiance_wh

    def get_last_30_days(self):
        now = datetime.now(EAT)
        result = []
        for i in range(29, -1, -1):
            date = now - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            data = self.history.get(date_str, {'consumption': 0, 'solar': 0, 'potential': 0})

            efficiency = 0
            if data['potential'] > 0:
                efficiency = min(100, (data['solar'] / data['potential']) * 100)

            result.append({
                'date': date_str,
                'day': date.day,
                'month': date.strftime('%b'),
                'weekday': date.strftime('%a'),
                'consumption_kwh': round(data['consumption'] / 1000, 1),
                'solar_kwh': round(data['solar'] / 1000, 1),
                'efficiency': round(efficiency, 0)
            })
        return result

load_manager = PersistentLoadManager(DATA_FILE)
history_manager = DailyHistoryManager(HISTORY_FILE)

def identify_active_appliances(current, previous, gen_active, backup_volts, primary_pct):
    """
    Enhanced detection using ML appliance classifier
    """
    detected = []
    
    # CRITICAL: Manual generator detection
    if gen_active:
        if primary_pct > 42: 
            detected.append("Generator Load")
        else: 
            detected.append("System Charging")
        return detected  # Return early if generator is on
    
    # Use ML detector for appliance classification
    appliances = ml_detector.detect_appliances(current, previous)
    
    if appliances:
        detected = appliances
    else:
        # Fallback
        if current < 100: 
            detected.append("Idle")
        else:
            detected.append(f"Load: {int(current)}W")
    
    return detected


# ----------------------------
# 3. Physics & Scheduler Engine
# ----------------------------

APPLIANCE_PROFILES = [
    {"id": "pool", "name": "Pool Pump", "watts": 1200, "hours": 4, "icon": "🏊", "priority": "low"},
    {"id": "wash", "name": "Washer", "watts": 800, "hours": 1.5, "icon": "🧺", "priority": "medium"},
    {"id": "oven", "name": "Oven", "watts": 2500, "hours": 1.5, "icon": "🍳", "priority": "high"}
]

def get_energy_status(p_pct, b_volts):
    """
    Centralized physics engine to calculate system state.
    Used by both current status (donut) and simulation (forecast).
    """
    p_total_wh = PRIMARY_BATTERY_CAPACITY_WH
    b_total_wh = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION

    # Calculate current Wh
    curr_p_wh = (p_pct / 100.0) * p_total_wh
    b_pct = max(0, min(100, (b_volts - 51.0) / 2.0 * 100))
    curr_b_wh = (b_pct / 100.0) * b_total_wh

    # Calculate Capacities (Tiers)
    primary_tier1_capacity = p_total_wh * 0.60
    backup_capacity = b_total_wh * 0.80
    emergency_capacity = p_total_wh * 0.20
    total_system_capacity = primary_tier1_capacity + backup_capacity + emergency_capacity

    # Calculate Available Energy (Tiers)
    primary_tier1_available = max(0, curr_p_wh - (p_total_wh * 0.40))
    backup_available = max(0, curr_b_wh - (b_total_wh * 0.20))
    emergency_available = max(0, min(curr_p_wh, p_total_wh * 0.40) - (p_total_wh * 0.20))

    total_available = primary_tier1_available + backup_available + emergency_available
    total_pct = (total_available / total_system_capacity * 100) if total_system_capacity > 0 else 0

    # Determine Active Tier
    if primary_tier1_available > 0: active_tier = 'primary'
    elif backup_available > 0: active_tier = 'backup'
    elif emergency_available > 0: active_tier = 'reserve'
    else: active_tier = 'empty'

    return {
        'total_pct': total_pct,
        'total_available_wh': total_available,
        'breakdown_wh': [primary_tier1_available, backup_available, emergency_available],
        'curr_p_wh': curr_p_wh,
        'curr_b_wh': curr_b_wh,
        'b_pct': b_pct,
        'active_tier': active_tier
    }

def generate_smart_schedule(status, solar_forecast_kw=0, load_forecast_kw=0, now_hour=None, 
                           heavy_loads_safe=False, gen_on=False, b_active=False):
    """Smart appliance scheduler - properly coordinated with recommendations."""
    battery_kwh_available = status['total_available_wh'] / 1000
    battery_soc_pct = status['total_pct']
    primary_pct = status.get('primary_battery_pct', 0)
    
    now = datetime.now(EAT)
    current_hour = now_hour if now_hour is not None else now.hour
    is_night = current_hour < 7 or current_hour >= 18
    is_past_4pm = current_hour >= 16
    
    advice = []
    
    for app in APPLIANCE_PROFILES:
        app_kw = app["watts"] / 1000
        app_kwh_required = (app["watts"] * app["hours"]) / 1000
        
        # Default to unsafe
        decision = {"msg": "Wait", "status": "unsafe", "color": "var(--warn)", "reason": ""}
        
        # ABSOLUTE BLOCKERS (same as recommendations)
        if gen_on:
            decision.update({
                "msg": "Generator On", 
                "status": "unsafe",
                "color": "var(--crit)",
                "reason": "Generator running - no loads"
            })
        elif b_active:
            decision.update({
                "msg": "Backup Active", 
                "status": "unsafe",
                "color": "var(--crit)",
                "reason": "Backup battery in use"
            })
        elif battery_soc_pct < 40:
            decision.update({
                "msg": "Battery Too Low", 
                "status": "unsafe",
                "color": "var(--crit)",
                "reason": f"System at {battery_soc_pct:.0f}%"
            })
        
        # Heavy Loads (>1500W) - STRICT RULES
        elif app["watts"] > 1500:
            if is_night or is_past_4pm:
                decision.update({
                    "msg": "Not Safe", 
                    "status": "unsafe",
                    "color": "var(--crit)",
                    "reason": "Heavy loads restricted: nighttime or after 4 PM"
                })
            elif heavy_loads_safe:
                decision.update({
                    "msg": "Safe in Window", 
                    "status": "safe", 
                    "color": "var(--success)", 
                    "reason": "Inside calculated safe solar window"
                })
            elif primary_pct > 85 and battery_kwh_available >= app_kwh_required * 1.5:
                decision.update({
                    "msg": "Safe (High Battery)", 
                    "status": "safe", 
                    "color": "var(--success)", 
                    "reason": f"Primary {primary_pct:.0f}% > 85%"
                })
            else:
                decision.update({
                    "msg": "Wait for Window", 
                    "status": "unsafe",
                    "color": "var(--warn)",
                    "reason": "Need solar window or >85% battery"
                })
        
        # Moderate Loads (800-1500W)
        elif app["watts"] >= 800:
            if is_night and battery_soc_pct < 60:
                decision.update({
                    "msg": "Avoid at Night", 
                    "status": "unsafe",
                    "color": "var(--warn)",
                    "reason": "Night + battery < 60%"
                })
            elif heavy_loads_safe or primary_pct > 75:
                decision.update({
                    "msg": "Conditionally Safe", 
                    "status": "safe", 
                    "color": "var(--success)", 
                    "reason": "Good conditions"
                })
            else:
                decision.update({
                    "msg": "Wait", 
                    "status": "unsafe",
                    "color": "var(--warn)",
                    "reason": "Conditions not optimal"
                })
        
        # Light Loads (<800W)
        else:
            if is_night and battery_soc_pct < 50:
                decision.update({
                    "msg": "Conserve", 
                    "status": "unsafe",
                    "color": "var(--warn)",
                    "reason": "Night + low battery"
                })
            elif battery_kwh_available >= app_kwh_required * 1.2:
                decision.update({
                    "msg": "Safe to Run", 
                    "status": "safe", 
                    "color": "var(--success)", 
                    "reason": "Battery sufficient"
                })
            else:
                decision.update({
                    "msg": "Wait", 
                    "status": "unsafe",
                    "color": "var(--warn)",
                    "reason": "Insufficient battery"
                })
        
        advice.append({
            **app, 
            "required_kwh": round(app_kwh_required, 2), 
            "decision": decision["msg"], 
            "status": decision["status"], 
            "color": decision["color"], 
            "reason": decision["reason"]
        })
    
    return advice

def calculate_battery_breakdown(p_pct, b_volts):
    """Calculates breakdown for circular chart using centralized logic."""
    status = get_energy_status(p_pct, b_volts)

    return {
        'chart_data': [round(x / 1000, 1) for x in status['breakdown_wh']],
        'tier_labels': ['Primary', 'Backup', 'Reserve'],
        'total_pct': round(status['total_pct'], 1),
        'total_kwh': round(status['total_available_wh'] / 1000, 1),
        'primary_pct': p_pct,
        'backup_voltage': round(b_volts, 1),
        'backup_pct': round(status['b_pct'], 1),
        'status_obj': {**status, 'primary_battery_pct': p_pct}  # Add primary_pct to status
    }

def calculate_battery_cascade(solar, load, p_pct, b_volts):
    """
    Simulates battery levels. 
    CRITICAL: Anchors the first data point to current breakdown state.
    """
    if not solar or not load: return {'labels': [], 'data': [], 'tiers': []}

    # 1. Initialize logic with current state
    start_status = get_energy_status(p_pct, b_volts)

    curr_p_wh = start_status['curr_p_wh']
    curr_b_wh = start_status['curr_b_wh']

    # Constants
    p_total_wh = PRIMARY_BATTERY_CAPACITY_WH
    b_total_wh = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION

    # 2. Set Start Points (Time 0)
    sim_data = [start_status['total_pct']]
    sim_labels = ["Now"] # Corresponds to current state
    tier_info = [start_status['active_tier']]

    # 3. Simulate future steps
    count = min(len(solar), len(load))

    for i in range(count):
        net = solar[i]['estimated_generation'] - load[i]['estimated_load']

        # Apply physics
        if net > 0:
            space_in_primary = p_total_wh - curr_p_wh
            if net <= space_in_primary:
                curr_p_wh += net
            else:
                curr_p_wh = p_total_wh
                overflow = net - space_in_primary
                curr_b_wh = min(b_total_wh, curr_b_wh + overflow)
        else:
            drain = abs(net)
            # Tier 1 Drain
            primary_min = p_total_wh * 0.40
            available_tier1 = max(0, curr_p_wh - primary_min)

            if available_tier1 >= drain:
                curr_p_wh -= drain
                drain = 0
            else:
                curr_p_wh = primary_min
                drain -= available_tier1

            # Tier 2 Drain
            if drain > 0:
                backup_min = b_total_wh * 0.20
                available_backup = max(0, curr_b_wh - backup_min)

                if available_backup >= drain:
                    curr_b_wh -= drain
                    drain = 0
                else:
                    curr_b_wh = backup_min
                    drain -= available_backup

            # Tier 3 Drain
            if drain > 0:
                emergency_min = p_total_wh * 0.20
                available_emergency = max(0, curr_p_wh - emergency_min)

                if available_emergency >= drain:
                    curr_p_wh -= drain
                else:
                    curr_p_wh = emergency_min

        # Calculate resulting state percentage
        primary_tier1_avail = max(0, curr_p_wh - (p_total_wh * 0.40))
        backup_avail = max(0, curr_b_wh - (b_total_wh * 0.20))
        emergency_avail = max(0, min(curr_p_wh, p_total_wh * 0.40) - (p_total_wh * 0.20))

        total_capacity = (p_total_wh * 0.60) + (b_total_wh * 0.80) + (p_total_wh * 0.20)
        total_available = primary_tier1_avail + backup_avail + emergency_avail

        percentage = (total_available / total_capacity) * 100 if total_capacity > 0 else 0

        # Determine active tier
        if primary_tier1_avail > 0: active_tier = 'primary'
        elif backup_avail > 0: active_tier = 'backup'
        elif emergency_avail > 0: active_tier = 'reserve'
        else: active_tier = 'empty'

        # Add point
        sim_data.append(percentage)
        sim_labels.append(solar[i]['time'].strftime('%H:%M'))
        tier_info.append(active_tier)

    return {'labels': sim_labels, 'data': sim_data, 'tiers': tier_info}

# ----------------------------
# 4. Helpers
# ----------------------------
headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"} if TOKEN else {}
last_alert_time, alert_history = {}, []
daily_accumulator = {'consumption_wh': 0, 'solar_wh': 0, 'last_date': None}
last_communication = {}

# Pool pump monitoring
pool_pump_start_time = None
pool_pump_last_alert = None

latest_data = {
    "timestamp": "Initializing...", "total_output_power": 0, "total_solar_input_W": 0,
    "primary_battery_min": 0, "backup_battery_voltage": 0, "backup_active": False,
    "generator_running": False, "inverters": [], "detected_appliances": [], 
    "solar_forecast": [], "load_forecast": [], 
    "battery_sim": {"labels": [], "data": [], "tiers": []},
    "energy_breakdown": {"chart_data": [1, 0, 1], "total_pct": 0, "total_kwh": 0},
    "scheduler": [],
    "heatmap_data": [],
    "hourly_24h": [],
    "ml_status": "Initializing"
}

def get_weather_forecast():
    """
    Multi-source weather forecast with automatic fallback.
    Tries Open-Meteo first, then WeatherAPI if it fails.
    Returns: {'times': [...], 'rad': [...]} or None
    """
    # Try Open-Meteo first (free, no API key needed)
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=shortwave_radiation&timezone=Africa/Nairobi&forecast_days=2"
        r = requests.get(url, timeout=5).json()
        print("✅ Weather data from Open-Meteo", flush=True)
        return {'times': r['hourly']['time'], 'rad': r['hourly']['shortwave_radiation'], 'source': 'Open-Meteo'}
    except Exception as e:
        print(f"⚠️ Open-Meteo failed: {e}", flush=True)
    
    # Fallback to WeatherAPI.com
    if WEATHERAPI_KEY:
        try:
            # WeatherAPI provides solar radiation in W/m²
            url = f"https://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={LATITUDE},{LONGITUDE}&days=2&hour=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
            r = requests.get(url, timeout=5).json()
            
            # Parse WeatherAPI response into same format as Open-Meteo
            times = []
            radiation = []
            
            for day in r['forecast']['forecastday']:
                for hour in day['hour']:
                    times.append(hour['time'])
                    # WeatherAPI gives solar_radiation directly in W/m²
                    radiation.append(hour.get('solar_radiation', 0))
            
            print("✅ Weather data from WeatherAPI (fallback)", flush=True)
            return {'times': times, 'rad': radiation, 'source': 'WeatherAPI'}
        except Exception as e:
            print(f"⚠️ WeatherAPI fallback failed: {e}", flush=True)
    else:
        print("⚠️ WEATHERAPI_KEY not configured, skipping fallback", flush=True)
    
    print("❌ All weather sources failed", flush=True)
    return None

def generate_solar_forecast(weather_data):
    """
    Generate 24-hour solar forecast from weather data.
    Handles both Open-Meteo and WeatherAPI formats.
    """
    forecast = []
    now = datetime.now(EAT)
    
    if not weather_data:
        print("⚠️ No weather data available - solar forecast will be empty", flush=True)
        return forecast
    
    source = weather_data.get('source', 'Unknown')
    w_map = {t: r for t, r in zip(weather_data['times'], weather_data['rad'])}
    
    for i in range(24):
        ft = now + timedelta(hours=i)
        
        # Try different time formats for compatibility
        key = ft.strftime('%Y-%m-%dT%H:00')
        rad = w_map.get(key, 0)
        
        # If not found, try alternative format (WeatherAPI uses different format)
        if rad == 0 and source == 'WeatherAPI':
            key_alt = ft.strftime('%Y-%m-%d %H:00')
            rad = w_map.get(key_alt, 0)
        
        # Convert radiation to estimated generation
        # radiation is in W/m², convert to total system watts
        est = (rad / 1000.0) * (TOTAL_SOLAR_CAPACITY_KW * 1000) * SOLAR_EFFICIENCY_FACTOR
        forecast.append({'time': ft, 'estimated_generation': est})
    
    print(f"✅ Generated solar forecast from {source}: {len(forecast)} hours", flush=True)
    return forecast

def calculate_daily_irradiance_potential(weather_data, target_date):
    """
    Calculate the actual solar potential for a specific day based on irradiance data.
    Returns the sum of hourly potential generation in Wh.
    Handles both Open-Meteo and WeatherAPI formats.
    """
    if not weather_data:
        print("⚠️ No weather data for irradiance calculation", flush=True)
        return 0
    
    total_potential_wh = 0
    source = weather_data.get('source', 'Unknown')
    w_map = {t: r for t, r in zip(weather_data['times'], weather_data['rad'])}
    
    # Iterate through 24 hours of the target date
    for hour in range(24):
        dt = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
        key = dt.strftime('%Y-%m-%dT%H:00')
        rad = w_map.get(key, 0)
        
        # Try alternative format for WeatherAPI
        if rad == 0 and source == 'WeatherAPI':
            key_alt = dt.strftime('%Y-%m-%d %H:00')
            rad = w_map.get(key_alt, 0)
        
        # Calculate potential generation for this hour
        hourly_potential = (rad / 1000.0) * (TOTAL_SOLAR_CAPACITY_KW * 1000) * SOLAR_EFFICIENCY_FACTOR
        total_potential_wh += hourly_potential
    
    return total_potential_wh

def send_email(subject, html, alert_type="general", send_via_email=True):
    """
    IMPROVED: Now includes send_via_email parameter to prevent email spam
    during manual generator operation or backup mode.
    """
    global last_alert_time, alert_history
    
    # Cooldown logic - different times for different alert severities
    cooldown_minutes = 120  # Default 2 hours
    if "critical" in alert_type.lower(): 
        cooldown_minutes = 60  # 1 hour for critical
    elif "high_load" in alert_type.lower(): 
        cooldown_minutes = 30  # 30 min for high load
    
    # Check cooldown
    if alert_type in last_alert_time:
        time_since = datetime.now(EAT) - last_alert_time[alert_type]
        if time_since < timedelta(minutes=cooldown_minutes):
            return  # Skip this alert
    
    # Send email only if requested
    if send_via_email and RESEND_API_KEY:
        try:
            requests.post(
                "https://api.resend.com/emails", 
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"}, 
                json={
                    "from": SENDER_EMAIL, 
                    "to": [RECIPIENT_EMAIL], 
                    "subject": subject, 
                    "html": html
                }
            )
        except: 
            pass
    
    # Always log the alert
    now = datetime.now(EAT)
    last_alert_time[alert_type] = now
    alert_history.insert(0, {"timestamp": now, "type": alert_type, "subject": subject})
    alert_history = alert_history[:20]

def check_alerts(inv_data, solar, total_solar, bat_discharge, gen_run):
    """
    Comprehensive alert checking from Render version.
    Checks inverter health, battery status, and discharge levels.
    """
    inv1 = next((i for i in inv_data if i['SN'] == 'RKG3B0400T'), None)
    inv2 = next((i for i in inv_data if i['SN'] == 'KAM4N5W0AG'), None)
    inv3 = next((i for i in inv_data if i['SN'] == 'JNK1CDR0KQ'), None)
    if not all([inv1, inv2, inv3]): return
    
    p_cap = min(inv1['Capacity'], inv2['Capacity'])
    b_active = inv3['OutputPower'] > 50
    b_volt = inv3['vBat']
    
    # Inverter health checks
    for inv in inv_data:
        if inv.get('communication_lost'): 
            send_email(f"⚠️ Comm Lost: {inv['Label']}", "Check inverter", "communication_lost")
        if inv.get('has_fault'): 
            send_email(f"🚨 FAULT: {inv['Label']}", "Fault code", "fault_alarm")
        if inv.get('high_temperature'): 
            send_email(f"🌡️ High Temp: {inv['Label']}", f"Temp: {inv['temperature']}", "high_temperature")
    
    # Critical: Generator or backup voltage low
    if gen_run or b_volt < 51.2:
        send_email("🚨 CRITICAL: Generator Running", "Backup critical", "critical")
        return
    
    # High alert: Backup active with low primary
    if b_active and p_cap < 40:
        send_email("⚠️ HIGH ALERT: Backup Active", "Reduce Load", "backup_active")
        return
    
    # Warning: Primary low
    if 40 < p_cap < 50:
        send_email("⚠️ Primary Low", "Reduce Load", "warning", send_via_email=b_active)
    
    # Discharge level alerts - only send email if backup is active
    if bat_discharge >= 4500: 
        send_email("🚨 URGENT: High Discharge", "Critical", "very_high_load", send_via_email=b_active)
    elif 2500 <= bat_discharge < 4500: 
        send_email("⚠️ High Discharge", "Warning", "high_load", send_via_email=b_active)
    elif 1500 <= bat_discharge < 2000 and p_cap < 50: 
        send_email("ℹ️ Moderate Discharge", "Info", "moderate_load", send_via_email=b_active)

# ----------------------------
# 5. Polling Loop
# ----------------------------
polling_active = False
polling_thread = None

def poll_growatt():
    global latest_data, polling_active, daily_accumulator, pool_pump_start_time, pool_pump_last_alert, last_communication
    if not TOKEN: return

    wx_data = get_weather_forecast()
    prev_watts = 0 
    last_save = datetime.now(EAT)
    last_ml_save = datetime.now(EAT)
    polling_active = True

    print("🚀 System Started: Enhanced Dashboard Mode with ML Detection", flush=True)

    while polling_active:
        try:
            now = datetime.now(EAT)
            tot_out, tot_sol, tot_bat = 0, 0, 0
            inv_data, p_caps = [], []
            b_data, gen_on = None, False

            for sn in SERIAL_NUMBERS:
                cfg = INVERTER_CONFIG.get(sn, {"label": sn, "type": "unknown"})
                success = False
                
                # FIX 1: Retry logic with 3 attempts
                for attempt in range(3):
                    try:
                        r = requests.post(API_URL, data={"storage_sn": sn}, headers=headers, timeout=10) # Reduced timeout
                        if r.status_code == 200:
                            try:
                                json_resp = r.json()
                            except ValueError:
                                raise Exception("Invalid JSON") # Trigger retry

                            if json_resp.get("error_code") == 0:
                                d = json_resp.get("data", {})
                                last_communication[sn] = now
                                
                                op = float(d.get("outPutPower") or 0)
                                cap = float(d.get("capacity") or 0)
                                vb = float(d.get("vBat") or 0)
                                pb = float(d.get("pBat") or 0)
                                sol = float(d.get("ppv") or 0) + float(d.get("ppv2") or 0)
                                temp = max(
                                    float(d.get("invTemperature") or 0),
                                    float(d.get("dcDcTemperature") or 0),
                                    float(d.get("temperature") or 0)
                                )
                                flt = int(d.get("errorCode") or 0) != 0

                                tot_out += op
                                tot_sol += sol
                                if pb > 0: tot_bat += pb

                                info = {
                                    "SN": sn, 
                                    "Label": cfg['label'], 
                                    "OutputPower": op, 
                                    "Capacity": cap, 
                                    "vBat": vb, 
                                    "temp": temp,
                                    "temperature": temp,
                                    "high_temperature": temp >= 60,
                                    "has_fault": flt,
                                    "communication_lost": False
                                }
                                inv_data.append(info)

                                if cfg['type'] == 'primary': p_caps.append(cap)
                                elif cfg['type'] == 'backup':
                                    b_data = info
                                    if float(d.get("vac") or 0) > 100 or float(d.get("pAcInPut") or 0) > 50: gen_on = True
                                
                                success = True
                                break # Success, exit retry loop
                    except Exception as e:
                        print(f"⚠️ Polling attempt {attempt+1} failed for {sn}: {e}", flush=True)
                        time.sleep(1) # Wait before retry

                if not success:
                    # FIX 1: Always add inverter to data even if failed so it doesn't disappear
                    inv_data.append({
                        "SN": sn, 
                        "Label": cfg.get('label', sn), 
                        "Type": cfg.get('type'),
                        "OutputPower": 0,
                        "Capacity": 0,
                        "vBat": 0,
                        "temp": 0,
                        "temperature": 0,
                        "high_temperature": False,
                        "has_fault": False,
                        "communication_lost": True
                    })

            p_min = min(p_caps) if p_caps else 0
            b_volts = b_data['vBat'] if b_data else 0
            b_act = b_data['OutputPower'] > 50 if b_data else False

            # Daily History Logic
            current_date = now.strftime('%Y-%m-%d')
            if daily_accumulator['last_date'] != current_date:
                if daily_accumulator['last_date']:
                    # Get yesterday's actual irradiance potential
                    yesterday = now - timedelta(days=1)
                    actual_irradiance_wh = calculate_daily_irradiance_potential(wx_data, yesterday)
                    
                    history_manager.update_daily(
                        daily_accumulator['last_date'],
                        daily_accumulator['consumption_wh'],
                        daily_accumulator['solar_wh'],
                        actual_irradiance_wh  # Use actual irradiance instead of hardcoded value
                    )
                    history_manager.save_history()
                daily_accumulator = {'consumption_wh': 0, 'solar_wh': 0, 'last_date': current_date}

            interval_hours = POLL_INTERVAL_MINUTES / 60.0
            daily_accumulator['consumption_wh'] += tot_out * interval_hours
            daily_accumulator['solar_wh'] += tot_sol * interval_hours

            # ML ENHANCED: Appliance Detection
            detected = identify_active_appliances(tot_out, prev_watts, gen_on, b_volts, p_min)
            is_manual_gen = any("Generator" in x for x in detected)
            
            # Only update load manager if NOT manually running generator
            if not is_manual_gen: 
                load_manager.update(tot_out)
            
            history_manager.add_hourly_datapoint(now, tot_out, tot_bat, tot_sol)

            # COMPREHENSIVE ALERT SYSTEM
            check_alerts(inv_data, None, tot_sol, tot_bat, gen_on)

            # Pool pump monitoring - check for sustained high discharge after 4pm
            if now.hour >= 16:
                if tot_bat > 1100:
                    if pool_pump_start_time is None:
                        pool_pump_start_time = now
                    
                    duration = now - pool_pump_start_time
                    if duration > timedelta(hours=3) and now.hour >= 18:
                        if pool_pump_last_alert is None or (now - pool_pump_last_alert) > timedelta(hours=1):
                            duration_hours = int(duration.total_seconds() // 3600)
                            send_email(
                                "⚠️ HIGH LOAD ALERT: Pool Pumps?", 
                                f"Battery discharge has been over 1.1kW for {duration_hours} hours. Did you leave the pool pumps on?", 
                                "high_load_continuous",
                                send_via_email=True
                            )
                            pool_pump_last_alert = now
                else:
                    pool_pump_start_time = None
            else:
                pool_pump_start_time = None

            # Save ML models periodically
            if (now - last_ml_save) > timedelta(hours=6):
                ml_detector.save_model()
                last_ml_save = now

            if (now - last_save) > timedelta(hours=1):
                load_manager.save_data()
                last_save = now

            # Forecasting & Simulation
            l_cast = load_manager.get_forecast(24)
            s_cast = generate_solar_forecast(wx_data)

            # Calculate Breakdown & Simulation ensuring synchronization
            breakdown = calculate_battery_breakdown(p_min, b_volts)
            sim_res = calculate_battery_cascade(s_cast, l_cast, p_min, b_volts)

            # Calculate safe window for scheduler (same logic as home() route)
            heavy_loads_safe = False
            if s_cast and l_cast:
                best_start, best_end, current_run = None, None, 0
                temp_start = None
                limit = min(len(s_cast), len(l_cast))
                
                for i in range(limit):
                    s_item = s_cast[i]
                    l_item = l_cast[i]
                    t = s_item['time']
                    
                    if t.hour >= 16:
                        if current_run > 0:
                            if best_start is None or current_run > ((best_end - best_start).total_seconds() // 3600 if best_end else 0):
                                best_start = temp_start
                                best_end = t
                            current_run = 0
                        continue
                    
                    gen = s_item['estimated_generation']
                    base_load = l_item.get('estimated_load', 600)
                    net_surplus = gen - base_load
                    
                    if net_surplus > 2500:
                        if current_run == 0: 
                            temp_start = t
                        current_run += 1
                    else:
                        if current_run > 0:
                            current_duration = (t - temp_start).total_seconds()
                            previous_duration = (best_end - best_start).total_seconds() if best_start and best_end else 0
                            
                            if best_start is None or current_duration > previous_duration:
                                best_start = temp_start
                                best_end = t
                            current_run = 0
                
                if current_run > 0:
                    if best_start is None or current_run > ((best_end - best_start).total_seconds() // 3600 if best_end else 0):
                        best_start = temp_start
                        best_end = s_cast[limit-1]['time'] + timedelta(hours=1)
                
                if best_start and best_end:
                    if best_start <= now <= best_end:
                        heavy_loads_safe = True

            # Generate schedule with proper context
            schedule = generate_smart_schedule(
                status=breakdown['status_obj'],
                solar_forecast_kw=s_cast, 
                load_forecast_kw=l_cast,
                now_hour=now.hour,
                heavy_loads_safe=heavy_loads_safe,
                gen_on=gen_on,
                b_active=b_act
            )

            # Remove status_obj before sending to frontend
            del breakdown['status_obj']

            heatmap = history_manager.get_last_30_days()
            hourly_24h = history_manager.get_last_24h_data()

            prev_watts = tot_out
            latest_data = {
                "timestamp": now.strftime("%H:%M:%S"),
                "total_output_power": tot_out,
                "total_solar_input_W": tot_sol,
                "total_battery_discharge_W": tot_bat,
                "primary_battery_min": p_min,
                "backup_battery_voltage": b_volts,
                "backup_active": b_act,
                "generator_running": gen_on,
                "detected_appliances": detected,
                "load_forecast": l_cast[:12],
                "solar_forecast": s_cast[:12],
                "battery_sim": sim_res,
                "energy_breakdown": breakdown,
                "scheduler": schedule,
                "inverters": inv_data,
                "heatmap_data": heatmap,
                "hourly_24h": hourly_24h,
                "ml_status": "Active",
                "heavy_loads_safe": heavy_loads_safe
            }
            
            # Log ML insights
            appliance_str = ", ".join(detected[:3]) if detected else "Idle"
            print(f"Update: Load={tot_out}W, Battery={breakdown['total_pct']}% | ML Detected: {appliance_str}", flush=True)

        except Exception as e: 
            print(f"Error: {e}", flush=True)
            latest_data['ml_status'] = f"Error: {str(e)[:50]}"
        
        if polling_active:
            for _ in range(POLL_INTERVAL_MINUTES * 60):
                if not polling_active: break
                time.sleep(1)

# ----------------------------
# 6. UI & Routes
# ----------------------------
@app.route('/health')
def health(): 
    return jsonify({"status": "healthy", "polling_thread_alive": polling_active})

@app.route('/start-polling')
def start_polling():
    global polling_active, polling_thread
    if not polling_active:
        polling_active = True
        polling_thread = Thread(target=poll_growatt, daemon=True)
        polling_thread.start()
    return jsonify({"status": "started"})

@app.route('/api/data')
def api_data(): return jsonify(latest_data)

@app.route('/api/ml-feedback', methods=['POST'])
def ml_feedback():
    """Endpoint for ML feedback from dashboard"""
    try:
        feedback = request.json
        if feedback:
            ml_detector.train_from_feedback(feedback)
            return jsonify({"status": "success", "message": "Feedback received"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error", "message": "No data received"})

@app.route('/api/ml-retrain', methods=['POST'])
def ml_retrain():
    """Manually trigger ML model retraining"""
    try:
        ml_detector.save_model()
        return jsonify({"status": "success", "message": "Models saved successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/")
def home():
    d = latest_data
    def _n(k): return float(d.get(k, 0) or 0)

    load = _n("total_output_power")
    solar = _n("total_solar_input_W")
    bat_dis = _n("total_battery_discharge_W")
    p_pct = _n("primary_battery_min")
    b_volt = _n("backup_battery_voltage")
    gen_on = d.get("generator_running", False)
    detected = d.get("detected_appliances", [])
    heavy_loads_safe = d.get("heavy_loads_safe", False)

    breakdown = d.get("energy_breakdown") or {
        "chart_data": [1,0,1], 
        "tier_labels": ['Primary', 'Backup', 'Reserve'],
        "total_pct": 0, 
        "total_kwh": 0,
        "primary_pct": 0,
        "backup_voltage": 0,
        "backup_pct": 0
    }
    sim = d.get("battery_sim") or {"labels": [], "data": [], "tiers": []}
    s_fc = d.get("solar_forecast") or []
    l_fc = d.get("load_forecast") or []
    schedule = d.get("scheduler") or []
    heatmap = d.get("heatmap_data") or []
    hourly_24h = d.get("hourly_24h") or []

    st_txt, st_col = "NORMAL", "var(--info)"
    if gen_on: st_txt, st_col = "GENERATOR ON", "var(--crit)"
    elif p_pct < 40: st_txt, st_col = "BACKUP ACTIVE", "var(--warn)"
    elif solar > load + 500: st_txt, st_col = "CHARGING", "var(--success)"

    is_charging = solar > (load + 100)
    is_discharging = bat_dis > 100 or load > solar
    
    # Calculate daytime status
    now = datetime.now(EAT)
    is_night = (now.hour < 7 or now.hour >= 18)

    alerts = alert_history[:8]
    tier_labels = breakdown.get('tier_labels', ['Primary', 'Backup', 'Reserve'])
    primary_pct = breakdown.get('primary_pct', 0)
    backup_voltage = breakdown.get('backup_voltage', 0)
    backup_pct = breakdown.get('backup_pct', 0)
    
    # Calculate surplus and weather conditions
    surplus_power = solar - load
    b_active = d.get("backup_active", False)
    
    # --- 1. Calculate Safe Heavy Load Window (Schedule) ---
    schedule_items = []
    
    if s_fc and l_fc:
        best_start, best_end, current_run = None, None, 0
        temp_start = None
        
        # Iterate by index to ensure alignment between Solar and Load forecasts
        limit = min(len(s_fc), len(l_fc))
        
        for i in range(limit):
            s_item = s_fc[i]
            l_item = l_fc[i]
            t = s_item['time']
            
            # STRICT SAFETY RULE: Cutoff heavy loads at 4 PM
            if t.hour >= 16:
                if current_run > 0: # Close any open window
                    if best_start is None or current_run > ((best_end - best_start).total_seconds() // 3600 if best_end else 0):
                        best_start = temp_start
                        best_end = t
                    current_run = 0
                continue # Skip remaining hours

            gen = s_item['estimated_generation']
            base_load = l_item.get('estimated_load', 600)
            
            # NET SURPLUS CHECK: 
            # Generation - Expected Household Load must be > 2500W to run a 3kW appliance safely
            net_surplus = gen - base_load
            
            if net_surplus > 2500:
                if current_run == 0: 
                    temp_start = t
                current_run += 1
            else:
                if current_run > 0:
                    current_duration = (t - temp_start).total_seconds()
                    previous_duration = (best_end - best_start).total_seconds() if best_start and best_end else 0
                    
                    if best_start is None or current_duration > previous_duration:
                        best_start = temp_start
                        best_end = t
                    current_run = 0
        
        # Handle case where window extends to the end of the array
        if current_run > 0:
             if best_start is None or current_run > ((best_end - best_start).total_seconds() // 3600 if best_end else 0):
                best_start = temp_start
                # approximate end as 1 hour after last start
                best_end = s_fc[limit-1]['time'] + timedelta(hours=1)
        
        if best_start and best_end:
            schedule_items.append({
                'icon': '🚿',
                'title': 'Best Time for Heavy Loads',
                'time': f"{best_start.strftime('%I:%M %p').lstrip('0')} - {best_end.strftime('%I:%M %p').lstrip('0')}",
                'class': 'good'
            })
        else:
            schedule_items.append({
                'icon': '☁️',
                'title': 'No Safe Solar Window',
                'time': 'Net solar insufficient for 3kW+ loads',
                'class': 'warning'
            })
        
        # Cloud Warning
        next_3_gen = sum([x['estimated_generation'] for x in s_fc[:3]]) / 3 if len(s_fc) >= 3 else 0
        if next_3_gen < 500 and 8 <= now.hour <= 15:
            schedule_items.append({
                'icon': '☁️',
                'title': 'Cloud Warning',
                'time': 'Low solar next 3 hours',
                'class': 'warning'
            })
    
    # --- 2. Generate Recommendations (Synced with Schedule) ---
    recommendation_items = []
    
    # Check if schedule blocked heavy loads
    schedule_blocks_heavy = any('No Safe Solar' in item.get('title', '') for item in schedule_items)
    
    if gen_on:
        recommendation_items.append({
            'icon': '🚨',
            'title': 'NO HEAVY LOADS',
            'description': 'Generator running - turn off all non-essential appliances',
            'class': 'critical'
        })
    elif b_active:
        recommendation_items.append({
            'icon': '⚠️',
            'title': 'MINIMIZE LOADS',
            'description': 'Backup battery active - essential loads only',
            'class': 'warning'
        })
    elif is_night:
        recommendation_items.append({
            'icon': '🌙',
            'title': 'NO HEAVY LOADS',
            'description': 'Night time - preserve battery life',
            'class': 'warning'
        })
    # Condition 1: High Battery Exception (Override Schedule if battery is very full)
    elif p_pct > 85 and not schedule_blocks_heavy:
        recommendation_items.append({
            'icon': '✅',
            'title': 'SAFE TO USE HEAVY LOADS',
            'description': f'Primary battery {p_pct:.0f}% (>85%) allows heavy usage',
            'class': 'good'
        })
    # Condition 2: Inside Safe Solar Window (Synced with Schedule)
    elif heavy_loads_safe:
        recommendation_items.append({
            'icon': '✅',
            'title': 'SAFE TO USE HEAVY LOADS',
            'description': f'Inside optimal solar window | Surplus available',
            'class': 'good'
        })
    # Condition 3: Good Solar but Low Battery/Outside Best Window
    elif not schedule_blocks_heavy and surplus_power > 1000:
        recommendation_items.append({
            'icon': '✅',
            'title': 'MODERATE LOADS OK',
            'description': f'Solar is good, but wait for peak window for heavy loads',
            'class': 'good'
        })
    elif breakdown['total_pct'] < 50 and solar < load:
        recommendation_items.append({
            'icon': '⚠️',
            'title': 'CONSERVE POWER',
            'description': f'Battery low ({breakdown["total_pct"]:.0f}%) and not charging well',
            'class': 'warning'
        })
    else:
        recommendation_items.append({
            'icon': '⚠️',
            'title': 'LIMIT HEAVY LOADS',
            'description': f'Insufficient net surplus or battery < 85%',
            'class': 'warning'
        })

    # HTML Template
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solar Monitor Pro - Dual House</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
    <style>
        :root { 
            --bg: #0a0e27; --bg-secondary: #151b3d; --card: rgba(21, 27, 61, 0.7); 
            --border: rgba(99, 102, 241, 0.2); --border-hover: rgba(99, 102, 241, 0.4);
            --text: #e2e8f5; --text-muted: #94a3b8; --text-dim: #64748b;
            --success: #10b981; --warn: #f59e0b; --crit: #ef4444; --info: #3b82f6;
            --accent: #6366f1; --accent-glow: rgba(99, 102, 241, 0.3);
            --primary-color: #10b981; --backup-color: #3b82f6; --reserve-color: #f59e0b;
            --house1-color: #10b981; --house2-color: #3b82f6;
        }
        
        * { box-sizing: border-box; }
        
        body { 
            background: var(--bg);
            background-image: 
                radial-gradient(ellipse at top, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(245, 158, 11, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at top left, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
            color: var(--text); 
            font-family: 'Manrope', sans-serif; 
            margin: 0; 
            padding: 20px; 
            min-height: 100vh;
        }
        
        .container { max-width: 1800px; margin: 0 auto; }
        
        .header {
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 30px;
            padding: 20px 30px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 20px;
            backdrop-filter: blur(20px);
        }
        
        .header h1 { 
            margin: 0; 
            font-size: 1.8rem; 
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent) 0%, var(--house1-color) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        
        .status-badge {
            padding: 8px 20px;
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            background: var(--accent-glow);
            border: 2px solid;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        .time-display {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text);
        }
        
        .grid { 
            display: grid; 
            grid-template-columns: repeat(12, 1fr); 
            gap: 20px;
        }
        
        .col-12 { grid-column: span 12; } 
        .col-6 { grid-column: span 12; } 
        .col-4 { grid-column: span 12; } 
        .col-3 { grid-column: span 6; }
        .col-8 { grid-column: span 12; }
        
        @media(min-width:768px){ 
            .col-6 { grid-column: span 6; } 
            .col-4 { grid-column: span 4; } 
            .col-3 { grid-column: span 4; }
            .col-8 { grid-column: span 8; }
        }
        
        .card { 
            background: var(--card); 
            border: 1px solid var(--border); 
            border-radius: 20px; 
            padding: 25px; 
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3); 
            backdrop-filter: blur(20px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--house1-color) 100%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .card:hover {
            border-color: var(--border-hover);
            transform: translateY(-2px);
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4);
        }
        
        .card:hover::before {
            opacity: 1;
        }
        
        .card-title { 
            font-size: 0.75rem; 
            text-transform: uppercase; 
            color: var(--text-muted); 
            margin: 0 0 12px 0; 
            letter-spacing: 2px; 
            font-weight: 600;
        }
        
        .metric-val { 
            font-family: 'JetBrains Mono', monospace; 
            font-size: 2.2rem; 
            font-weight: 700; 
            line-height: 1.2;
            background: linear-gradient(135deg, var(--text) 0%, var(--text-muted) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-unit { 
            font-size: 0.85rem; 
            color: var(--text-dim); 
            font-weight: 600;
        }
        
        .tag { 
            padding: 6px 14px; 
            border-radius: 50px; 
            font-size: 0.75rem; 
            font-weight: 700; 
            background: rgba(99, 102, 241, 0.15); 
            border: 1px solid var(--border); 
            display: inline-flex; 
            align-items: center; 
            gap: 6px; 
            margin: 4px 4px 4px 0;
            transition: all 0.2s;
        }
        
        .tag.house1 { background: rgba(16, 185, 129, 0.15); border-color: var(--house1-color); }
        .tag.house2 { background: rgba(59, 130, 246, 0.15); border-color: var(--house2-color); }
        
        .tag:hover {
            background: rgba(99, 102, 241, 0.25);
            transform: translateY(-1px);
        }
        
        /* Enhanced Power Flow */
        .flow-diagram { 
            position: relative; 
            height: 320px; 
            width: 100%; 
            display: flex; 
            justify-content: center; 
            align-items: center;
            background: radial-gradient(ellipse at center, rgba(99, 102, 241, 0.05) 0%, transparent 70%);
        }
        
        .node { 
            position: absolute; 
            width: 90px; 
            height: 90px; 
            border-radius: 50%; 
            background: var(--bg-secondary); 
            border: 3px solid var(--border); 
            z-index: 2; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            box-shadow: 0 5px 25px rgba(0, 0, 0, 0.4);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .node:hover {
            transform: scale(1.1);
            box-shadow: 0 8px 35px rgba(0, 0, 0, 0.5);
        }
        
        .node-icon { font-size: 1.8rem; margin-bottom: 4px; }
        .node-val { 
            font-family: 'JetBrains Mono'; 
            font-size: 0.7rem; 
            font-weight: bold; 
            color: var(--text-muted);
        }
        
        .n-solar { top: 15px; left: 50%; transform: translateX(-50%); border-color: var(--warn); }
        .n-inv   { top: 50%; left: 50%; transform: translate(-50%, -50%); width: 110px; height: 110px; border-color: var(--info); }
        .n-home  { top: 50%; right: 12%; transform: translateY(-50%); border-color: var(--success); }
        .n-bat   { bottom: 15px; left: 50%; transform: translateX(-50%); border-color: var(--success); }
        .n-gen   { top: 50%; left: 12%; transform: translateY(-50%); border-color: var(--crit); }
        
        .line { position: absolute; background: var(--border); z-index: 1; overflow: hidden; border-radius: 2px; }
        .line-v { width: 5px; height: 85px; left: 50%; transform: translateX(-50%); }
        .l-solar { top: 105px; } 
        .l-bat { bottom: 105px; }
        .line-h { height: 5px; width: 28%; top: 50%; transform: translateY(-50%); }
        .l-gen { left: 18%; } 
        .l-home { right: 18%; }
        
        .dot { 
            position: absolute; 
            background: var(--accent); 
            border-radius: 50%; 
            width: 8px; 
            height: 8px; 
            box-shadow: 0 0 15px var(--accent-glow); 
            opacity: 0; 
        }
        
        .flow-down .dot { left: -1.5px; animation: flowY 1.8s linear infinite; opacity: 1; }
        .flow-up .dot { left: -1.5px; animation: flowY-rev 1.8s linear infinite; opacity: 1; }
        .flow-right .dot { top: -1.5px; animation: flowX 1.8s linear infinite; opacity: 1; }
        
        @keyframes flowY { 0%{top:0%} 100%{top:100%} } 
        @keyframes flowY-rev { 0%{top:100%} 100%{top:0%} } 
        @keyframes flowX { 0%{left:0%} 100%{left:100%} }
        
        .pulse-g { animation: pulse-green 2s infinite; } 
        .pulse-r { animation: pulse-red 2s infinite; } 
        .pulse-y { animation: pulse-yellow 2s infinite; }
        
        @keyframes pulse-green { 
            0%{box-shadow:0 0 0 0 rgba(16, 185, 129, 0.7)} 
            70%{box-shadow:0 0 0 20px rgba(16, 185, 129, 0)} 
            100%{box-shadow:0 0 0 0 rgba(16, 185, 129, 0)} 
        }
        @keyframes pulse-red { 
            0%{box-shadow:0 0 0 0 rgba(239, 68, 68, 0.7)} 
            70%{box-shadow:0 0 0 20px rgba(239, 68, 68, 0)} 
            100%{box-shadow:0 0 0 0 rgba(239, 68, 68, 0)} 
        }
        @keyframes pulse-yellow { 
            0%{box-shadow:0 0 0 0 rgba(245, 158, 11, 0.7)} 
            70%{box-shadow:0 0 0 20px rgba(245, 158, 11, 0)} 
            100%{box-shadow:0 0 0 0 rgba(245, 158, 11, 0)} 
        }

        /* Enhanced Scheduler */
        .sched-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); 
            gap: 15px; 
        }
        
        .sched-tile { 
            background: rgba(99, 102, 241, 0.05); 
            border: 2px solid var(--border); 
            border-radius: 18px; 
            padding: 18px;
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            text-align: center;
            cursor: pointer; 
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .sched-tile::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .sched-tile:hover::before {
            left: 100%;
        }
        
        .sched-tile:hover { 
            background: rgba(99, 102, 241, 0.15); 
            transform: translateY(-3px); 
            border-color: var(--accent);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
        }
        
        .sched-tile.active { 
            border-color: var(--accent); 
            background: rgba(99, 102, 241, 0.2); 
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.4);
        }
        
        .tile-icon { font-size: 2.5rem; margin-bottom: 10px; }
        .tile-name { font-weight: 700; font-size: 1rem; margin-bottom: 6px; color: var(--text); }
        .tile-status { 
            font-size: 0.7rem; 
            font-weight: 700; 
            padding: 4px 10px; 
            border-radius: 12px; 
            background: rgba(0,0,0,0.4);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Recommendations */
        .rec-item {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1rem;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 0.75rem;
            border-left: 4px solid;
        }
        
        .rec-item.critical { border-left-color: var(--crit); }
        .rec-item.warning { border-left-color: var(--warn); }
        .rec-item.good { border-left-color: var(--success); }
        .rec-item.info { border-left-color: var(--info); }
        
        .rec-icon { font-size: 1.5rem; }
        .rec-title { font-weight: 600; margin-bottom: 0.25rem; }
        .rec-desc { font-size: 0.85rem; color: var(--text-muted); }
        
        /* Heatmap Calendar */
        .heatmap-container {
            padding: 10px 0;
        }
        
        .heatmap-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(40px, 1fr));
            gap: 8px;
            margin-top: 15px;
        }
        
        .heatmap-cell {
            aspect-ratio: 1;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 700;
            transition: all 0.3s;
            cursor: pointer;
            border: 1px solid var(--border);
            position: relative;
        }
        
        .heatmap-cell:hover {
            transform: scale(1.15);
            z-index: 10;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.4);
        }
        
        .heatmap-day { font-size: 0.9rem; margin-bottom: 2px; }
        .heatmap-eff { font-size: 0.65rem; opacity: 0.8; }
        
        /* Efficiency color scale */
        .eff-0 { background: rgba(100, 116, 139, 0.3); }
        .eff-1 { background: rgba(239, 68, 68, 0.3); border-color: rgba(239, 68, 68, 0.5); }
        .eff-2 { background: rgba(245, 158, 11, 0.3); border-color: rgba(245, 158, 11, 0.5); }
        .eff-3 { background: rgba(250, 204, 21, 0.3); border-color: rgba(250, 204, 21, 0.5); }
        .eff-4 { background: rgba(132, 204, 22, 0.3); border-color: rgba(132, 204, 22, 0.5); }
        .eff-5 { background: rgba(16, 185, 129, 0.4); border-color: rgba(16, 185, 129, 0.6); }
        
        .heatmap-legend {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .legend-box {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid var(--border);
        }
        
        .alert-row { 
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            border-bottom: 1px solid var(--border); 
            padding: 12px 0; 
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        
        .alert-row:hover {
            background: rgba(99, 102, 241, 0.05);
            padding-left: 10px;
        }
        
        .alert-row:last-child { border-bottom: none; }
        
        .alert-time {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-dim);
        }
        
        /* Chart improvements */
        canvas {
            filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.2));
        }
        
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🏠 SOLAR MONITOR PRO - DUAL HOUSE</h1>
                <span class="status-badge" style="border-color: {{ st_col }}; color: {{ st_col }}">{{ st_txt }}</span>
            </div>
            <div class="time-display">{{ d['timestamp'] }}</div>
        </div>

        <div class="grid">
            <!-- VISUAL POWER FLOW -->
            <div class="col-12 card">
                <div class="card-title">Real-Time Energy Flow</div>
                <div class="flow-diagram">
                    <div class="line line-v l-solar {{ 'flow-down' if solar > 50 else '' }}"><div class="dot"></div></div>
                    <div class="line line-v l-bat {{ 'flow-down' if is_charging else ('flow-up' if is_discharging else '') }}"><div class="dot"></div></div>
                    <div class="line line-h l-home {{ 'flow-right' if load > 100 else '' }}"><div class="dot"></div></div>
                    <div class="line line-h l-gen {{ 'flow-right' if gen_on else '' }}"><div class="dot"></div></div>
                    
                    <div class="node n-solar {{ 'pulse-y' if solar > 50 else '' }}">
                        <div class="node-icon">☀️</div>
                        <div class="node-val">{{ '%0.f'|format(solar) }}W</div>
                    </div>
                    <div class="node n-gen {{ 'pulse-r' if gen_on else '' }}">
                        <div class="node-icon">⚙️</div>
                        <div class="node-val">{{ 'ON' if gen_on else 'OFF' }}</div>
                    </div>
                    <div class="node n-inv">
                        <div class="node-icon">⚡</div>
                        <div class="node-val">INV</div>
                    </div>
                    <div class="node n-home {{ 'pulse-g' if load > 2000 else '' }}">
                        <div class="node-icon">🏠</div>
                        <div class="node-val">{{ '%0.f'|format(load) }}W</div>
                    </div>
                    <div class="node n-bat {{ 'pulse-g' if is_charging else ('pulse-r' if is_discharging else '') }}">
                        <div class="node-icon">🔋</div>
                        <div class="node-val">{{ breakdown['total_pct'] }}%</div>
                    </div>
                </div>
            </div>

            <!-- KEY METRICS -->
            <div class="col-4 card">
                <div class="card-title">Solar Generation</div>
                <div class="metric-val" style="color:var(--warn)">{{ '%0.f'|format(solar) }}<span style="font-size:1.2rem">W</span></div>
                <div class="metric-unit">Current Input</div>
            </div>
            <div class="col-4 card">
                <div class="card-title">Home Consumption</div>
                <div class="metric-val" style="color:var(--info)">{{ '%0.f'|format(load) }}<span style="font-size:1.2rem">W</span></div>
                <div class="metric-unit">Active Load</div>
            </div>
            <div class="col-4 card">
                <div class="card-title">Battery Status</div>
                <div class="metric-val" style="color:var(--success)">{{ breakdown['total_pct'] }}<span style="font-size:1.2rem">%</span></div>
                <div class="metric-unit">{{ breakdown['total_kwh'] }} kWh Usable</div>
            </div>

            <!-- 30-DAY HEATMAP -->
            <div class="col-12 card">
                <div class="card-title">30-Day Solar Efficiency Calendar</div>
                <div class="heatmap-container">
                    <div class="heatmap-grid">
                        {% for day in heatmap %}
                        {% set eff_class = 'eff-0' %}
                        {% if day.efficiency >= 80 %}{% set eff_class = 'eff-5' %}
                        {% elif day.efficiency >= 60 %}{% set eff_class = 'eff-4' %}
                        {% elif day.efficiency >= 40 %}{% set eff_class = 'eff-3' %}
                        {% elif day.efficiency >= 20 %}{% set eff_class = 'eff-2' %}
                        {% elif day.efficiency > 0 %}{% set eff_class = 'eff-1' %}
                        {% endif %}
                        <div class="heatmap-cell {{ eff_class }}" title="{{ day.date }}: {{ day.solar_kwh }}kWh solar, {{ day.consumption_kwh }}kWh used, {{ day.efficiency }}% efficiency">
                            <div class="heatmap-day">{{ day.day }}</div>
                            <div class="heatmap-eff">{{ day.efficiency|int }}%</div>
                        </div>
                        {% endfor %}
                    </div>
                    <div class="heatmap-legend">
                        <div class="legend-item"><div class="legend-box eff-0"></div> No Data</div>
                        <div class="legend-item"><div class="legend-box eff-1"></div> 0-20%</div>
                        <div class="legend-item"><div class="legend-box eff-2"></div> 20-40%</div>
                        <div class="legend-item"><div class="legend-box eff-3"></div> 40-60%</div>
                        <div class="legend-item"><div class="legend-box eff-4"></div> 60-80%</div>
                        <div class="legend-item"><div class="legend-box eff-5"></div> 80-100%</div>
                    </div>
                </div>
            </div>

            <!-- SCENARIO PLANNER -->
            <div class="col-12 card">
                <div class="card-title">Smart Appliance Scheduler</div>
                <div class="sched-grid">
                    {% for s in schedule %}
                    <div class="sched-tile" onclick="toggleSim('{{ s.id }}', {{ s.watts }})" id="btn-{{ s.id }}">
                        <div class="tile-icon">{{ s.icon }}</div>
                        <div class="tile-name">{{ s.name }}</div>
                        <div class="tile-status" style="color: {{ s.color }}">{{ s.decision }}</div>
                    </div>
                    {% endfor %}
                </div>
                <div style="margin-top:15px; text-align:right; font-size:0.9rem; color: var(--text-muted)">
                    Simulated Additional Load: <span id="sim-val" style="color: var(--accent); font-weight: 700">0W</span>
                </div>
            </div>

            <!-- RECOMMENDATIONS -->
            <div class="col-6 card">
                <div class="card-title">📝 System Recommendations</div>
                {% for rec in recommendation_items %}
                <div class="rec-item {{ rec.class }}">
                    <div class="rec-icon">{{ rec.icon }}</div>
                    <div>
                        <div class="rec-title">{{ rec.title }}</div>
                        <div class="rec-desc">{{ rec.description }}</div>
                    </div>
                </div>
                {% endfor %}
            </div>

            <!-- SCHEDULE -->
            <div class="col-6 card">
                <div class="card-title">📅 Today's Schedule</div>
                {% for item in schedule_items %}
                <div class="rec-item {{ item.class }}" style="border-left: 3px solid {{ 'var(--primary)' if 'good' in item.class else 'var(--warning)' }}">
                    <div class="rec-icon">{{ item.icon }}</div>
                    <div>
                        <div class="rec-title">{{ item.title }}</div>
                        <div class="rec-desc">{{ item.time }}</div>
                    </div>
                </div>
                {% endfor %}
            </div>

            <!-- BATTERY PROJECTION -->
            <div class="col-8 card">
                <div class="card-title">24-Hour Battery Projection (Tier-Aware)</div>
                <div style="height:280px"><canvas id="simChart"></canvas></div>
            </div>

            <!-- STORAGE BREAKDOWN -->
            <div class="col-4 card">
                <div class="card-title">Storage Breakdown</div>
                <div style="height:200px"><canvas id="pieChart"></canvas></div>
                <div style="text-align:center; margin-top:15px; font-size:1.1rem; color: var(--success); font-weight: 700">
                    {{ breakdown['total_pct'] }}% Available
                </div>
                <div style="text-align:center; color: var(--text-muted); font-size:0.85rem">
                    {{ breakdown['total_kwh'] }} kWh Usable</div>
                <div style="margin-top:10px; padding-top:10px; border-top: 1px solid var(--border); font-size:0.75rem; color: var(--text-dim)">
                    <div>Primary: {{ primary_pct }}%</div>
                    <div>Backup: {{ backup_voltage }}V ({{ backup_pct }}%)</div>
                </div>
            </div>

            <!-- 24-HOUR LOAD VS DISCHARGE -->
            <div class="col-12 card">
                <div class="card-title">Last 24 Hours: Load vs Battery Discharge vs Solar</div>
                <div style="height:300px"><canvas id="hourlyChart"></canvas></div>
                <div style="margin-top:10px; text-align:right; font-size:0.75rem; color: var(--text-muted)">
                    Use mouse wheel or pinch to zoom | Drag to pan | Double-click to reset
                </div>
            </div>

            <!-- ACTIVITY & ALERTS -->
            <div class="col-12 card">
                <div class="card-title">House Activity Detection</div>
                <div style="margin-bottom:20px">
                    {% if detected %}
                        {% for a in detected %}
                            {% if 'House 1' in a %}
                                <span class="tag house1">🏠 {{ a }}</span>
                            {% elif 'House 2' in a %}
                                <span class="tag house2">🏠 {{ a }}</span>
                            {% elif 'Water' in a or 'Generator' in a %}
                                <span class="tag" style="background: rgba(239, 68, 68, 0.15); border-color: var(--crit);">⚠️ {{ a }}</span>
                            {% else %}
                                <span class="tag">{{ a }}</span>
                            {% endif %}
                        {% endfor %}
                    {% else %}
                        <span class="tag" style="opacity:0.5">System Idle</span>
                    {% endif %}
                </div>
                
                <div class="card-title" style="margin-top:20px">Recent Alerts</div>
                {% if alerts %}
                    {% for a in alerts %}
                    <div class="alert-row">
                        <div style="color:{{ 'var(--crit)' if 'crit' in a.type else 'var(--text)' }}; font-weight: 600;">
                            {{ a.subject }}
                        </div>
                        <div class="alert-time">{{ a.timestamp.strftime('%b %d, %H:%M') }}</div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div style="color: var(--text-dim); font-style: italic; padding: 10px 0;">No recent alerts</div>
                {% endif %}
            </div>
        </div>
    </div>

    <script>
        // Data from Backend
        const labels = {{ sim['labels']|tojson }};
        const baseData = {{ sim['data']|tojson }};
        const tierData = {{ sim['tiers']|tojson }};
        const sForecast = {{ s_fc|tojson }};
        const lForecast = {{ l_fc|tojson }};
        const pieData = {{ breakdown['chart_data']|tojson }};
        const tierLabels = {{ tier_labels|tojson }};
        const hourly24h = {{ hourly_24h|tojson }};
        
        // Sim State
        let activeSims = {};
        let simTierData = [];
        
        // Chart.js Global Config
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.2)';
        Chart.defaults.font.family = "'Manrope', sans-serif";
        
        // --- 1. Tier-Aware Battery Projection Chart ---
        const ctx = document.getElementById('simChart');
        
        // Create color array based on tier
        const borderColors = tierData.map(tier => {
            if (tier === 'primary') return '#10b981';  // Green
            if (tier === 'backup') return '#3b82f6';   // Blue
            if (tier === 'reserve') return '#f59e0b';  // Orange
            return '#64748b';  // Grey for empty
        });
        
        const backgroundColors = tierData.map(tier => {
            if (tier === 'primary') return 'rgba(16, 185, 129, 0.1)';
            if (tier === 'backup') return 'rgba(59, 130, 246, 0.1)';
            if (tier === 'reserve') return 'rgba(245, 158, 11, 0.1)';
            return 'rgba(100, 116, 139, 0.1)';
        });
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Battery Level',
                        data: baseData,
                        segment: {
                            borderColor: ctx => {
                                const idx = ctx.p0DataIndex;
                                return borderColors[idx] || '#10b981';
                            },
                            backgroundColor: ctx => {
                                const idx = ctx.p0DataIndex;
                                return backgroundColors[idx] || 'rgba(16, 185, 129, 0.1)';
                            }
                        },
                        borderWidth: 3,
                        tension: 0.4,
                        pointRadius: 0,
                        fill: true
                    },
                    {
                        label: 'With Additional Load',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        borderDash: [5, 5],
                        borderWidth: 3,
                        tension: 0.4,
                        pointRadius: 0,
                        fill: true,
                        hidden: true
                    }
                ]
            },
            options: {
                responsive: true, 
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: {
                        labels: {
                            usePointStyle: true,
                            padding: 15,
                            font: { size: 12, weight: '600' },
                            filter: (item, chart) => item.text !== 'Battery Level' // Hide default legend
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#e2e8f0',
                        bodyColor: '#94a3b8',
                        borderColor: 'rgba(99, 102, 241, 0.3)',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                let tier;
                                let scenarioLabel = "";

                                // Determine if this is the "Normal" line or "Simulated" line
                                if (context.datasetIndex === 1 && simTierData.length > 0) {
                                    tier = simTierData[context.dataIndex];
                                    scenarioLabel = "With Load"; // Label for the orange line
                                } else {
                                    tier = tierData[context.dataIndex];
                                    scenarioLabel = "Normal";    // Label for the green line
                                }

                                tier = tier || 'unknown';
                                const tierName = tier.charAt(0).toUpperCase() + tier.slice(1);
                                
                                // Return clear format: "Scenario: Value% (Tier)"
                                return `${scenarioLabel}: ${context.parsed.y.toFixed(1)}% (${tierName})`;
                            }
                        }
                    }
                },
                scales: { 
                    y: { 
                        min: 0, 
                        max: 100, 
                        grid: { color: 'rgba(99, 102, 241, 0.1)' },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: { 
                        grid: { display: false },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });

        // --- 2. Storage Pie Chart (No Percentages) ---
        new Chart(document.getElementById('pieChart'), {
            type: 'doughnut',
            data: {
                labels: tierLabels,
                datasets: [{
                    data: pieData,
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.9)',   // Primary (green)
                        'rgba(59, 130, 246, 0.8)',    // Backup (blue)
                        'rgba(245, 158, 11, 0.8)'     // Reserve (orange)
                    ],
                    borderWidth: 0,
                    borderRadius: 4,
                    spacing: 2
                }]
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false, 
                cutout: '70%', 
                plugins: { 
                    legend: { 
                        position: 'bottom',
                        labels: {
                            padding: 12,
                            usePointStyle: true,
                            font: { size: 10, weight: '600' }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#e2e8f0',
                        bodyColor: '#94a3b8',
                        borderColor: 'rgba(99, 102, 241, 0.3)',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + ' kWh';
                            }
                        }
                    }
                } 
            }
        });

        // --- 3. 24-Hour Load vs Discharge Chart with Zoom ---
        const hourlyLabels = hourly24h.map(d => {
            const dt = new Date(d.timestamp);
            return dt.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});
        });
        const loadData = hourly24h.map(d => d.load);
        const dischargeData = hourly24h.map(d => d.battery_discharge);
        const solarData = hourly24h.map(d => d.solar);

        const hourlyCtx = document.getElementById('hourlyChart');
        const hourlyChart = new Chart(hourlyCtx, {
            type: 'line',
            data: {
                labels: hourlyLabels,
                datasets: [
                    {
                        label: 'Load (W)',
                        data: loadData,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 1,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Battery Discharge (W)',
                        data: dischargeData,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 1,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Solar (W)',
                        data: solarData,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 1,
                        pointHoverRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: {
                        labels: {
                            usePointStyle: true,
                            padding: 15,
                            font: { size: 12, weight: '600' }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#e2e8f0',
                        bodyColor: '#94a3b8',
                        borderColor: 'rgba(99, 102, 241, 0.3)',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(0) + 'W';
                            }
                        }
                    },
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: true,
                                speed: 0.1
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x',
                        },
                        pan: {
                            enabled: true,
                            mode: 'x',
                        },
                        limits: {
                            x: {min: 'original', max: 'original'}
                        }
                    }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(99, 102, 241, 0.1)' },
                        ticks: {
                            callback: function(value) {
                                return value + 'W';
                            }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45,
                            maxTicksLimit: 20
                        }
                    }
                }
            }
        });

        // Double-click to reset zoom
        hourlyCtx.ondblclick = function() {
            hourlyChart.resetZoom();
        };

        // --- 4. Interaction Logic ---
        function toggleSim(id, watts) {
            const btn = document.getElementById('btn-' + id);
            
            if (activeSims[id]) {
                delete activeSims[id];
                btn.classList.remove('active');
            } else {
                activeSims[id] = watts;
                btn.classList.add('active');
            }
            recalcSimulation();
        }

        function recalcSimulation() {
            const totalSimWatts = Object.values(activeSims).reduce((a, b) => a + b, 0);
            document.getElementById('sim-val').innerText = totalSimWatts + "W";
            
            if (totalSimWatts === 0) {
                chart.data.datasets[1].hidden = true;
                chart.update();
                return;
            }

            // Full tiered discharge simulation matching Python
            const P_TOTAL = 30000;
            const B_TOTAL = 21000 * 0.7;  // 14,700
            const TOTAL_CAPACITY = (P_TOTAL * 0.60) + (B_TOTAL * 0.80) + (P_TOTAL * 0.20); 
            
            // Reconstruct initial state from simple variables
            let curr_p_wh = ({{ p_pct }} / 100.0) * P_TOTAL;
            let curr_b_wh = ({{ backup_pct }} / 100.0) * B_TOTAL;
            
            // First point is simply the current percentage (Index 0)
            let simCurve = [ baseData[0] ];
            let newSimTiers = [ tierData[0] ]; // Start with current tier
            
            for (let i = 0; i < 24; i++) {
                // Get forecasted values
                const solar_gen = sForecast[i]?.estimated_generation || 0;
                const base_load = lForecast[i]?.estimated_load || {{ load }};
                
                // Calculate net energy flow including simulated load
                const total_load = base_load + totalSimWatts;
                const net_flow = solar_gen - total_load;
                
                // CORRECTED: Handle both charging and discharging
                if (net_flow > 0) {  // Surplus - charge batteries
                    let charge_amount = net_flow;
                    
                    // Charge primary first (up to 100%)
                    const primary_space = P_TOTAL - curr_p_wh;
                    if (primary_space > 0) {
                        const charge_to_primary = Math.min(charge_amount, primary_space);
                        curr_p_wh += charge_to_primary;
                        charge_amount -= charge_to_primary;
                    }
                    
                    // Then charge backup if there's still surplus
                    if (charge_amount > 0) {
                        const backup_space = B_TOTAL - curr_b_wh;
                        if (backup_space > 0) {
                            const charge_to_backup = Math.min(charge_amount, backup_space);
                            curr_b_wh += charge_to_backup;
                        }
                    }
                }
                else {  // Deficit - discharge batteries (existing logic)
                    let drain = Math.abs(net_flow);
                    
                    // Tier 1: Primary down to 40%
                    let primary_min = P_TOTAL * 0.40;
                    let available_tier1 = Math.max(0, curr_p_wh - primary_min);
                    
                    if (available_tier1 >= drain) {
                        curr_p_wh -= drain;
                        drain = 0;
                    } else {
                        curr_p_wh = primary_min;
                        drain -= available_tier1;
                    }
                    
                    // Tier 2: Backup if drain remains
                    if (drain > 0) {
                        let backup_min = B_TOTAL * 0.20;
                        let available_backup = Math.max(0, curr_b_wh - backup_min);
                        
                        if (available_backup >= drain) {
                            curr_b_wh -= drain;
                            drain = 0;
                        } else {
                            curr_b_wh = backup_min;
                            drain -= available_backup;
                        }
                    }
                    
                    // Tier 3: Emergency primary (40% -> 20%)
                    if (drain > 0) {
                        let emergency_min = P_TOTAL * 0.20;
                        let available_emergency = Math.max(0, curr_p_wh - emergency_min);
                        
                        if (available_emergency >= drain) {
                            curr_p_wh -= drain;
                        } else {
                            curr_p_wh = emergency_min;
                        }
                    }
                }
                
                // Calculate display values
                let primary_tier1_avail = Math.max(0, curr_p_wh - (P_TOTAL * 0.40));
                let backup_avail = Math.max(0, curr_b_wh - (B_TOTAL * 0.20));
                let emergency_avail = Math.max(0, Math.min(curr_p_wh, P_TOTAL * 0.40) - (P_TOTAL * 0.20));
                
                let total_available = primary_tier1_avail + backup_avail + emergency_avail;
                let percentage = (total_available / TOTAL_CAPACITY) * 100;

                // Determine Active Tier
                let active_tier = 'empty';
                if (primary_tier1_avail > 0) active_tier = 'primary';
                else if (backup_avail > 0) active_tier = 'backup';
                else if (emergency_avail > 0) active_tier = 'reserve';
                
                simCurve.push(percentage);
                newSimTiers.push(active_tier);
            }
            
            simTierData = newSimTiers;
            chart.data.datasets[1].data = simCurve;
            chart.data.datasets[1].hidden = false;
            chart.update();
        }
        
        // Health check and auto-refresh
        fetch('/health').then(r => r.json()).then(d => { 
            if(!d.polling_thread_alive) fetch('/start-polling'); 
        });
        
        setTimeout(() => location.reload(), 120000); // Refresh every 2 minutes
    </script>
</body>
</html>
    """
    return render_template_string(html, 
        d=d, solar=solar, load=load, p_pct=p_pct, b_volt=b_volt, 
        gen_on=gen_on, detected=detected, st_txt=st_txt, st_col=st_col,
        is_charging=is_charging, is_discharging=is_discharging,
        s_fc=s_fc, l_fc=l_fc, sim=sim, breakdown=breakdown, schedule=schedule,
        heatmap=heatmap, alerts=alerts, hourly_24h=hourly_24h,
        tier_labels=tier_labels, primary_pct=primary_pct, 
        backup_voltage=backup_voltage, backup_pct=backup_pct,
        recommendation_items=recommendation_items, schedule_items=schedule_items,
        heavy_loads_safe=heavy_loads_safe
    )

if __name__ == '__main__':
    # Create required directories and files
    for file in [DATA_FILE, HISTORY_FILE, ML_MODEL_FILE]:
        if not Path(file).exists():
            Path(file).touch()
    
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
