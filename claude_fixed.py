import os
import time
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from threading import Thread, Lock
from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
from collections import deque
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import logging
import math

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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "solar-multisite-2026")

# API Configuration
API_URL = "https://openapi.growatt.com/v1/device/storage/storage_last_data"
API_HISTORY_URL = "https://openapi.growatt.com/v1/device/storage/storage_data"
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", 5))
DATA_FILE = "load_patterns.json"
HISTORY_FILE = "daily_history.json"
ML_MODEL_FILE = "appliance_models.pkl"

for file in [DATA_FILE, HISTORY_FILE, ML_MODEL_FILE]:
    if not Path(file).exists():
        Path(file).touch()
        print(f"Created empty file: {file}", flush=True)

# Multi-Site Configuration
SITES = {
    "kajiado": {
        "password": os.getenv("PASSWORD_KAJIADO"),
        "api_token": os.getenv("GROWATT_API_KEY_KAJIADO"),
        "serial_numbers": ["RKG3B0400T", "KAM4N5W0AG", "JNK1CDR0KQ"],
        "label": "Kajiado Home Solar",
        "recipient_email": os.getenv("RECIPIENT_EMAIL_KAJIADO", os.getenv("RECIPIENT_EMAIL")),
        "inverter_config": {
            "RKG3B0400T": {"label": "Inverter 1", "type": "primary"},
            "KAM4N5W0AG": {"label": "Inverter 2", "type": "primary"},
            "JNK1CDR0KQ": {"label": "Inverter 3 (Backup)", "type": "backup"}
        },
        "primary_battery_wh": 30000,
        "backup_battery_wh": 21000,
        "backup_degradation": 0.70,
        "solar_capacity_kw": 10,
        "latitude": -1.85238,
        "longitude": 36.77683,
        "appliance_type": "home"
    },
    "nairobi": {
        "password": os.getenv("PASSWORD_NAIROBI"),
        "api_token": os.getenv("GROWATT_API_KEY_NAIROBI"),
        "serial_numbers": [os.getenv("SERIAL_NUMBERS_NAIROBI")] if os.getenv("SERIAL_NUMBERS_NAIROBI") else [],
        "label": "Nairobi Office Solar",
        "recipient_email": os.getenv("RECIPIENT_EMAIL_NAIROBI", os.getenv("RECIPIENT_EMAIL")),
        "inverter_config": {},
        "primary_battery_wh": 6000,
        "backup_battery_wh": 0,
        "backup_degradation": 0,
        "solar_capacity_kw": 5,
        "latitude": -1.2921,
        "longitude": 36.8219,
        "appliance_type": "office"
    }
}

# Auto-config Nairobi inverter
_nbr_sn = SITES["nairobi"]["serial_numbers"][0] if SITES["nairobi"]["serial_numbers"] else None
if _nbr_sn:
    SITES["nairobi"]["inverter_config"][_nbr_sn] = {"label": "Primary Inverter", "type": "primary"}

SOLAR_EFFICIENCY_FACTOR = 0.85
EAT = timezone(timedelta(hours=3))

def get_current_site():
    site_id = session.get('site_id')
    if site_id and site_id in SITES:
        return site_id, SITES[site_id]
    return None, None

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
    def __init__(self, appliance_type="home"):
        self.appliance_type = appliance_type
        self.model_file = f"{ML_MODEL_FILE}_{appliance_type}"
        self.load_history = deque(maxlen=1000)  # Store recent load patterns
        self.appliance_classifier = None
        self.scaler = StandardScaler()
        self.model_lock = Lock()
        
        # Define appliance classes based on type
        if appliance_type == "office":
            self.APPLIANCE_CLASSES = [
                'idle', 'multiple_loads',
                'desktop_computer', 'laptop', 'workstation', 'server', 'printer',
                'laser_printer', 'inkjet_printer', 'photocopier', 'scanner',
                'air_conditioner_window', 'air_conditioner_split', 'ceiling_fan',
                'coffee_maker', 'espresso_machine', 'microwave', 'mini_fridge',
                'monitor', 'projector', 'router', 'switch', 'ups'
            ]
        else:
            self.APPLIANCE_CLASSES = [
                'idle', 'multiple_loads', 'microwave', 'electric_kettle', 'toaster', 
                'refrigerator', 'freezer', 'washing_machine', 'dryer', 'iron', 
                'electric_fan', 'ceiling_fan', 'hair_dryer', 'tv_55', 'water_pump', 
                'pool_pump', 'borehole_pump', 'security_system'
            ]
        
        self.training_data = []
        self.training_labels = []
        self.feature_window = 10
        self.load_model()
        
    def load_model(self):
        if Path(self.model_file).exists():
            try:
                with self.model_lock:
                    if os.path.getsize(self.model_file) > 0:
                        models = joblib.load(self.model_file)
                        self.appliance_classifier = models.get('appliance_classifier')
                        self.scaler = models.get('scaler', StandardScaler())
                        self.training_data = models.get('training_data', [])
                        self.training_labels = models.get('training_labels', [])
                        print(f"✅ Loaded ML appliance models for {self.appliance_type}", flush=True)
                    else:
                        print("⚠️ ML model file exists but is empty. initializing default.", flush=True)
                        self.init_default_models()
            except Exception as e:
                print(f"⚠️ Failed to load ML models (reinitializing): {e}", flush=True)
                self.init_default_models()
        else:
            self.init_default_models()
    
    def init_default_models(self):
        self.appliance_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        seed_data = [
            [50, 20, 100, 10, 45, 5, 10, 20, -15, 0, 0.4, 0.7],
            [3500, 600, 4200, 2800, 3400, 200, 400, 800, -600, 3, 0.17, 0.8],
        ]
        seed_labels = ['idle', 'multiple_loads']
        
        if self.appliance_type == 'office':
            office_seeds = [
                [200, 40, 380, 120, 195, 15, 25, 60, -50, 1, 0.13, 0.8],
                [400, 120, 800, 280, 390, 80, 70, 180, -160, 2, 0.30, 0.8],
                [1800, 50, 1850, 1750, 1800, 10, 20, 50, -50, 0, 0.03, 0.8]
            ]
            office_labels = ['desktop_computer', 'printer', 'air_conditioner_split']
            seed_data.extend(office_seeds)
            seed_labels.extend(office_labels)

        self.training_data = seed_data
        self.training_labels = seed_labels
        
        X = np.array(seed_data)
        y = np.array(seed_labels)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.appliance_classifier.fit(X_scaled, y)
        print(f"✅ Initialized ML appliance models with seed data for {self.appliance_type}", flush=True)
    
    def save_model(self):
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
        features = []
        if len(load_data) == 0: return [0] * 12
        features.append(np.mean(load_data))
        features.append(np.std(load_data))
        features.append(np.max(load_data))
        features.append(np.min(load_data))
        features.append(np.median(load_data))
        if len(load_data) > 1:
            changes = np.diff(load_data)
            features.append(np.mean(changes))
            features.append(np.std(changes))
            features.append(np.max(changes))
            features.append(np.min(changes))
            large_changes = np.abs(changes) > 500
            features.append(np.sum(large_changes))
            if np.mean(load_data) > 0:
                features.append(np.std(load_data) / np.mean(load_data))
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        if time_data:
            hour = time_data.hour
            features.append(1.0 if 6 <= hour <= 22 else 0.5)
        else:
            features.append(0.7)
        while len(features) < 12:
            features.append(0)
        return features[:12]
    
    def detect_appliances(self, current_load, previous_load=0):
        try:
            now = datetime.now(EAT)
            self.load_history.append({'timestamp': now, 'load': current_load})
            if len(self.load_history) < self.feature_window:
                return self._simple_fallback_detection(current_load)
            
            recent_loads = [item['load'] for item in list(self.load_history)[-self.feature_window:]]
            features = self.extract_features(recent_loads, now)
            
            try:
                features_scaled = self.scaler.transform([features])
                predicted_class = self.appliance_classifier.predict(features_scaled)[0]
                confidence = np.max(self.appliance_classifier.predict_proba(features_scaled))
                
                detected = []
                if confidence > 0.4:
                    appliance_name = predicted_class.replace('_', ' ').title()
                    detected.append(f"{appliance_name}")
                    if current_load > 3000 and np.std(recent_loads) > 500:
                        if predicted_class != 'multiple_loads':
                            detected.append("+ Other Loads")
                else:
                    detected = self._simple_fallback_detection(current_load)
                return detected
            except Exception as e:
                print(f"⚠️ ML classification error: {e}", flush=True)
                return self._simple_fallback_detection(current_load)
        except Exception as e:
            print(f"⚠️ Appliance detection error: {e}", flush=True)
            return ["System Error"]
    
    def _simple_fallback_detection(self, current_load):
        if current_load < 50: return ["Idle"]
        elif 50 <= current_load < 150: return ["Lights/Router"]
        elif 150 <= current_load < 400: return ["Computer/Screens"] if self.appliance_type == 'office' else ["Fridge/Lights"]
        elif 400 <= current_load < 800: return ["Servers/Printers"] if self.appliance_type == 'office' else ["TV/Computer"]
        elif 800 <= current_load < 1500: return ["AC/Microwave"] if self.appliance_type == 'office' else ["Microwave/Kettle"]
        else: return ["High Load/Multiple"]
    
    def train_from_feedback(self, feedback_data):
        try:
            if not feedback_data.get('actual_appliance') or not feedback_data.get('load_pattern'): return
            load_pattern = feedback_data['load_pattern']
            timestamp = datetime.fromisoformat(feedback_data['timestamp']) if isinstance(feedback_data['timestamp'], str) else feedback_data['timestamp']
            features = self.extract_features(load_pattern, timestamp)
            self.training_data.append(features)
            self.training_labels.append(feedback_data['actual_appliance'])
            if len(self.training_data) > 500:
                self.training_data = self.training_data[-500:]
                self.training_labels = self.training_labels[-500:]
            if len(self.training_data) >= 20:
                X = np.array(self.training_data)
                y = np.array(self.training_labels)
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.appliance_classifier.fit(X_scaled, y)
                self.save_model()
                print(f"✅ ML model retrained with {len(self.training_data)} examples", flush=True)
        except Exception as e:
            print(f"⚠️ Feedback training error: {e}", flush=True)

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

    def add_hourly_datapoint(self, timestamp, load_w, battery_discharge_w, solar_w, grid_gen_w=0):
        self.hourly_data.append({
            'timestamp': timestamp.isoformat(),
            'load': load_w,
            'battery_discharge': battery_discharge_w,
            'solar': solar_w,
            'grid_gen': grid_gen_w
        })
        if len(self.hourly_data) > 288:
            self.hourly_data = self.hourly_data[-288:]

    def get_last_24h_data(self):
        return self.hourly_data

    def update_daily(self, date_str, total_consumption_wh, total_solar_wh, actual_irradiance_wh):
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

def identify_active_appliances(current, previous, gen_active, backup_volts, primary_pct, ml_detector_instance, site_id='kajiado'):
    detected = []
    
    # Only block for Generator if not Nairobi (Nairobi has Utility which is normal)
    if gen_active and site_id != 'nairobi':
        if primary_pct > 42:
            detected.append("Generator Load")
        else: 
            detected.append("System Charging")
        return detected
    
    appliances = ml_detector_instance.detect_appliances(current, previous)
    
    if appliances:
        detected = appliances
    else:
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

def get_energy_status(p_pct, b_volts, site_config):
    p_total_wh = site_config["primary_battery_wh"]
    b_total_wh = site_config["backup_battery_wh"] * site_config["backup_degradation"]

    curr_p_wh = (p_pct / 100.0) * p_total_wh
    
    b_pct = 0
    curr_b_wh = 0
    if b_total_wh > 0:
        b_pct = max(0, min(100, (b_volts - 51.0) / 2.0 * 100))
        curr_b_wh = (b_pct / 100.0) * b_total_wh

    primary_tier1_capacity = p_total_wh * 0.60
    backup_capacity = b_total_wh * 0.80
    emergency_capacity = p_total_wh * 0.20
    total_system_capacity = primary_tier1_capacity + backup_capacity + emergency_capacity

    primary_tier1_available = max(0, curr_p_wh - (p_total_wh * 0.40))
    backup_available = 0
    if b_total_wh > 0:
        backup_available = max(0, curr_b_wh - (b_total_wh * 0.20))
    emergency_available = max(0, min(curr_p_wh, p_total_wh * 0.40) - (p_total_wh * 0.20))

    total_available = primary_tier1_available + backup_available + emergency_available
    total_pct = (total_available / total_system_capacity * 100) if total_system_capacity > 0 else 0

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
                           heavy_loads_safe=False, gen_on=False, b_active=False, site_id='kajiado'):
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
        
        decision = {"msg": "Wait", "status": "unsafe", "color": "var(--warn)", "reason": ""}
        
        if gen_on:
            if site_id == 'nairobi':
                pass 
            else:
                decision.update({
                    "msg": "Generator On", 
                    "status": "unsafe",
                    "color": "var(--crit)",
                    "reason": "Generator running - no loads"
                })
        elif site_id == 'nairobi':
            decision.update({
                "msg": "Grid Failure", 
                "status": "unsafe",
                "color": "var(--crit)",
                "reason": "No utility power - battery only"
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

def calculate_battery_breakdown(p_pct, b_volts, site_config):
    status = get_energy_status(p_pct, b_volts, site_config)
    
    if site_config.get("backup_battery_wh", 0) > 0:
        chart_data = [round(x / 1000, 1) for x in status['breakdown_wh']]
        tier_labels = ['Primary', 'Backup', 'Reserve']
        tier_colors = ['rgba(16, 185, 129, 0.9)', 'rgba(59, 130, 246, 0.8)', 'rgba(245, 158, 11, 0.8)']
    else:
        wh_list = status['breakdown_wh']
        chart_data = [round(wh_list[0] / 1000, 1), round(wh_list[2] / 1000, 1)]
        tier_labels = ['Primary', 'Reserve']
        tier_colors = ['rgba(16, 185, 129, 0.9)', 'rgba(245, 158, 11, 0.8)']

    return {
        'chart_data': chart_data,
        'tier_labels': tier_labels,
        'tier_colors': tier_colors,
        'total_pct': round(status['total_pct'], 1),
        'total_kwh': round(status['total_available_wh'] / 1000, 1),
        'primary_pct': p_pct,
        'backup_voltage': round(b_volts, 1),
        'backup_pct': round(status['b_pct'], 1),
        'status_obj': {**status, 'primary_battery_pct': p_pct}
    }

def calculate_battery_cascade(solar, load, p_pct, b_volts, site_config):
    if not solar or not load: return {'labels': [], 'data': [], 'tiers': []}

    start_status = get_energy_status(p_pct, b_volts, site_config)

    curr_p_wh = start_status['curr_p_wh']
    curr_b_wh = start_status['curr_b_wh']

    p_total_wh = site_config["primary_battery_wh"]
    b_total_wh = site_config["backup_battery_wh"] * site_config["backup_degradation"]

    sim_data = [start_status['total_pct']]
    sim_labels = ["Now"]
    tier_info = [start_status['active_tier']]

    count = min(len(solar), len(load))

    for i in range(count):
        net = solar[i]['estimated_generation'] - load[i]['estimated_load']

        if net > 0:
            space_in_primary = p_total_wh - curr_p_wh
            if net <= space_in_primary:
                curr_p_wh += net
            else:
                curr_p_wh = p_total_wh
                if b_total_wh > 0:
                    overflow = net - space_in_primary
                    curr_b_wh = min(b_total_wh, curr_b_wh + overflow)
        else:
            drain = abs(net)
            primary_min = p_total_wh * 0.40
            available_tier1 = max(0, curr_p_wh - primary_min)

            if available_tier1 >= drain:
                curr_p_wh -= drain
                drain = 0
            else:
                curr_p_wh = primary_min
                drain -= available_tier1

            if drain > 0 and b_total_wh > 0:
                backup_min = b_total_wh * 0.20
                available_backup = max(0, curr_b_wh - backup_min)

                if available_backup >= drain:
                    curr_b_wh -= drain
                    drain = 0
                else:
                    curr_b_wh = backup_min
                    drain -= available_backup

            if drain > 0:
                emergency_min = p_total_wh * 0.20
                available_emergency = max(0, curr_p_wh - emergency_min)

                if available_emergency >= drain:
                    curr_p_wh -= drain
                else:
                    curr_p_wh = emergency_min

        primary_tier1_avail = max(0, curr_p_wh - (p_total_wh * 0.40))
        backup_avail = 0
        if b_total_wh > 0:
            backup_avail = max(0, curr_b_wh - (b_total_wh * 0.20))
        emergency_avail = max(0, min(curr_p_wh, p_total_wh * 0.40) - (p_total_wh * 0.20))

        total_capacity = (p_total_wh * 0.60) + (b_total_wh * 0.80) + (p_total_wh * 0.20)
        total_available = primary_tier1_avail + backup_avail + emergency_avail

        percentage = (total_available / total_capacity) * 100 if total_capacity > 0 else 0

        if primary_tier1_avail > 0: active_tier = 'primary'
        elif backup_avail > 0: active_tier = 'backup'
        elif emergency_avail > 0: active_tier = 'reserve'
        else: active_tier = 'empty'

        sim_data.append(percentage)
        sim_labels.append(solar[i]['time'].strftime('%H:%M'))
        tier_info.append(active_tier)

    return {'labels': sim_labels, 'data': sim_data, 'tiers': tier_info}

# ----------------------------
# 4. Helpers
# ----------------------------
headers_template = {"Content-Type": "application/x-www-form-urlencoded"}
last_alert_time = {}  # Will store {site_id: {alert_type: timestamp}}
alert_history = {}    # Will store {site_id: [alert_list]}
site_latest_data = {}

def get_historical_weather(lat, lon, date):
    """Fetch historical solar irradiance data for a specific date using Open-Meteo Archive API"""
    try:
        date_str = date.strftime('%Y-%m-%d')
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&hourly=shortwave_radiation&timezone=Africa/Nairobi"
        r = requests.get(url, timeout=10).json()
        print(f"✅ Historical weather data from Open-Meteo Archive for {date_str}", flush=True)
        return {'times': r['hourly']['time'], 'rad': r['hourly']['shortwave_radiation'], 'source': 'Open-Meteo-Archive'}
    except Exception as e:
        print(f"⚠️ Open-Meteo Archive failed for {date.strftime('%Y-%m-%d')}: {e}", flush=True)
        return None

def get_weather_forecast(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=shortwave_radiation&timezone=Africa/Nairobi&forecast_days=2"
        r = requests.get(url, timeout=5).json()
        print("✅ Weather data from Open-Meteo", flush=True)
        return {'times': r['hourly']['time'], 'rad': r['hourly']['shortwave_radiation'], 'source': 'Open-Meteo'}
    except Exception as e:
        print(f"⚠️ Open-Meteo failed: {e}", flush=True)
    
    if WEATHERAPI_KEY:
        try:
            url = f"https://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={lat},{lon}&days=2&hour=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
            r = requests.get(url, timeout=5).json()
            times = []
            radiation = []
            for day in r['forecast']['forecastday']:
                for hour in day['hour']:
                    times.append(hour['time'])
                    radiation.append(hour.get('solar_radiation', 0))
            print("✅ Weather data from WeatherAPI (fallback)", flush=True)
            return {'times': times, 'rad': radiation, 'source': 'WeatherAPI'}
        except Exception as e:
            print(f"⚠️ WeatherAPI fallback failed: {e}", flush=True)
    else:
        print("⚠️ WEATHERAPI_KEY not configured, skipping fallback", flush=True)
    
    print("❌ All weather sources failed", flush=True)
    return None

def generate_solar_forecast(weather_data, site_config):
    forecast = []
    now = datetime.now(EAT)
    if not weather_data:
        print("⚠️ No weather data available - solar forecast will be empty", flush=True)
        return forecast
    source = weather_data.get('source', 'Unknown')
    w_map = {t: r for t, r in zip(weather_data['times'], weather_data['rad'])}
    for i in range(24):
        ft = now + timedelta(hours=i)
        key = ft.strftime('%Y-%m-%dT%H:00')
        rad = w_map.get(key, 0)
        if rad == 0 and source == 'WeatherAPI':
            key_alt = ft.strftime('%Y-%m-%d %H:00')
            rad = w_map.get(key_alt, 0)
        est = (rad / 1000.0) * (site_config["solar_capacity_kw"] * 1000) * SOLAR_EFFICIENCY_FACTOR
        forecast.append({'time': ft, 'estimated_generation': est})
    print(f"✅ Generated solar forecast from {source}: {len(forecast)} hours", flush=True)
    return forecast

def calculate_daily_irradiance_potential(weather_data, target_date, site_config):
    if not weather_data:
        print("⚠️ No weather data for irradiance calculation", flush=True)
        return 0
    total_potential_wh = 0
    source = weather_data.get('source', 'Unknown')
    w_map = {t: r for t, r in zip(weather_data['times'], weather_data['rad'])}
    for hour in range(24):
        dt = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
        key = dt.strftime('%Y-%m-%dT%H:00')
        rad = w_map.get(key, 0)
        if rad == 0 and source == 'WeatherAPI':
            key_alt = dt.strftime('%Y-%m-%d %H:00')
            rad = w_map.get(key_alt, 0)
        hourly_potential = (rad / 1000.0) * (site_config["solar_capacity_kw"] * 1000) * SOLAR_EFFICIENCY_FACTOR
        total_potential_wh += hourly_potential
    return total_potential_wh

def send_email(subject, html, alert_type="general", send_via_email=True, site_id="kajiado"):
    global last_alert_time, alert_history
    
    # Initialize site-specific structures if needed
    if site_id not in last_alert_time:
        last_alert_time[site_id] = {}
    if site_id not in alert_history:
        alert_history[site_id] = []
    
    cooldown_minutes = 120
    if "critical" in alert_type.lower(): cooldown_minutes = 60
    elif "high_load" in alert_type.lower(): cooldown_minutes = 30
    
    # Check cooldown for this site and alert type
    if alert_type in last_alert_time[site_id]:
        time_since = datetime.now(EAT) - last_alert_time[site_id][alert_type]
        if time_since < timedelta(minutes=cooldown_minutes):
            return
    
    # Get site-specific recipient email
    recipient = SITES.get(site_id, {}).get("recipient_email", RECIPIENT_EMAIL)
    
    if send_via_email and RESEND_API_KEY and recipient:
        try:
            # Add site label to subject
            site_label = SITES.get(site_id, {}).get("label", site_id)
            full_subject = f"[{site_label}] {subject}"
            
            requests.post(
                "https://api.resend.com/emails", 
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"}, 
                json={"from": SENDER_EMAIL, "to": [recipient], "subject": full_subject, "html": html}
            )
        except: pass
    
    now = datetime.now(EAT)
    last_alert_time[site_id][alert_type] = now
    alert_history[site_id].insert(0, {"timestamp": now, "type": alert_type, "subject": subject, "site_id": site_id})
    alert_history[site_id] = alert_history[site_id][:20]

def check_alerts(inv_data, solar, total_solar, bat_discharge, gen_run, site_id='kajiado'):
    inv1 = next((i for i in inv_data if i['SN'] == 'RKG3B0400T'), None)
    inv2 = next((i for i in inv_data if i['SN'] == 'KAM4N5W0AG'), None)
    inv3 = next((i for i in inv_data if i['SN'] == 'JNK1CDR0KQ'), None)
    
    if not any([inv1, inv2, inv3]):
        p_cap = 0
        if inv_data: p_cap = min([i['Capacity'] for i in inv_data])
        b_active = False
        b_volt = 53
    else:
        p_cap = min(inv1['Capacity'], inv2['Capacity']) if inv1 and inv2 else 0
        b_active = inv3['OutputPower'] > 50 if inv3 else False
        b_volt = inv3['vBat'] if inv3 else 0
    
    for inv in inv_data:
        if inv.get('communication_lost'): 
            send_email(f"⚠️ Comm Lost: {inv['Label']}", "Check inverter", "communication_lost", site_id=site_id)
        if inv.get('has_fault'): 
            send_email(f"🚨 FAULT: {inv['Label']}", "Fault code", "fault_alarm", site_id=site_id)
        if inv.get('high_temperature'): 
            send_email(f"🌡️ High Temp: {inv['Label']}", f"Temp: {inv['temperature']}", "high_temperature", site_id=site_id)
    
    if site_id == 'nairobi':
        if not gen_run:
            send_email("🚨 CRITICAL: Grid Failure", "No utility power - running on battery", "critical", site_id=site_id)
            return
    else:
        if gen_run or (b_volt < 51.2 and b_volt > 10):
            send_email("🚨 CRITICAL: Generator Running", "Backup critical", "critical", site_id=site_id)
            return
    
    if b_active and p_cap < 40:
        send_email("⚠️ HIGH ALERT: Backup Active", "Reduce Load", "backup_active", site_id=site_id)
        return
    
    if 40 < p_cap < 50:
        send_email("⚠️ Primary Low", "Reduce Load", "warning", send_via_email=b_active, site_id=site_id)
    
    if bat_discharge >= 4500: 
        send_email("🚨 URGENT: High Discharge", "Critical", "very_high_load", send_via_email=b_active, site_id=site_id)
    elif 2500 <= bat_discharge < 4500: 
        send_email("⚠️ High Discharge", "Warning", "high_load", send_via_email=b_active, site_id=site_id)
    elif 1500 <= bat_discharge < 2000 and p_cap < 50: 
        send_email("ℹ️ Moderate Discharge", "Info", "moderate_load", send_via_email=b_active, site_id=site_id)

# ----------------------------
# 5. Polling Loop
# ----------------------------
polling_active = False
polling_thread = None
site_managers = {}

def poll_growatt():
    global site_latest_data, polling_active, last_communication, site_managers

    for site_id, config in SITES.items():
        if site_id not in site_managers:
            site_managers[site_id] = {
                'load_manager': PersistentLoadManager(f"{site_id}_{DATA_FILE}"),
                'history_manager': DailyHistoryManager(f"{site_id}_{HISTORY_FILE}"),
                'ml_detector': ApplianceDetector(config["appliance_type"]),
                'daily_accumulator': {'consumption_wh': 0, 'solar_wh': 0, 'last_date': None},
                'pool_pump_start_time': None,
                'pool_pump_last_alert': None,
                'last_save': datetime.now(EAT),
                'last_ml_save': datetime.now(EAT),
                'prev_watts': 0
            }

    polling_active = True
    print("🚀 System Started: Multi-Site Mode", flush=True)

    while polling_active:
        try:
            for site_id, config in SITES.items():
                try:
                    managers = site_managers[site_id]
                    token = config["api_token"]
                    if not token: continue

                    wx_data = get_weather_forecast(config["latitude"], config["longitude"])
                    
                    now = datetime.now(EAT)
                    tot_out, tot_sol, tot_bat, tot_grid = 0, 0, 0, 0
                    inv_data, p_caps = [], []
                    b_data, gen_on = None, False
                    
                    # NEW: Initialize Grid Stats container
                    grid_stats = {
                        "eToUserToday": "0", "eToUserTotal": "0", "eToGridToday": "0", "eToGridTotal": "0",
                        "eacChargeToday": "0", "eacChargeTotal": "0", 
                        "eBatDisChargeToday": "0", "eBatDisChargeTotal": "0",
                        "eacDisChargeToday": "0", "eacDisChargeTotal": "0"
                    }

                    for sn in config["serial_numbers"]:
                        inv_cfg = config["inverter_config"].get(sn, {"label": sn, "type": "unknown"})
                        success = False
                        
                        site_headers = headers_template.copy()
                        site_headers["token"] = token

                        for attempt in range(3):
                            try:
                                r = requests.post(API_URL, data={"storage_sn": sn}, headers=site_headers, timeout=10)
                                if r.status_code == 200:
                                    try:
                                        json_resp = r.json()
                                    except ValueError:
                                        raise Exception("Invalid JSON")

                                    if json_resp.get("error_code") == 0:
                                        d = json_resp.get("data", {})
                                        
                                        op = float(d.get("outPutPower") or 0)
                                        cap = float(d.get("capacity") or 0)
                                        vb = float(d.get("vBat") or 0)
                                        pb = float(d.get("pBat") or 0)
                                        sol = float(d.get("ppv") or 0) + float(d.get("ppv2") or 0)
                                        grid_pwr = float(d.get("pAcInPut") or 0)
                                        temp = max(
                                            float(d.get("invTemperature") or 0),
                                            float(d.get("dcDcTemperature") or 0),
                                            float(d.get("temperature") or 0)
                                        )
                                        flt = int(d.get("errorCode") or 0) != 0

                                        tot_out += op
                                        tot_sol += sol
                                        tot_grid += grid_pwr
                                        if pb > 0: tot_bat += pb

                                        info = {
                                            "SN": sn, 
                                            "Label": inv_cfg['label'], 
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

                                        if inv_cfg['type'] == 'primary': 
                                            p_caps.append(cap)
                                            if site_id == 'nairobi' and float(d.get("vGrid") or 0) >= 180: 
                                                gen_on = True
                                        elif inv_cfg['type'] == 'backup':
                                            b_data = info
                                            if float(d.get("vac") or 0) > 100 or float(d.get("pAcInPut") or 0) > 50: gen_on = True
                                        
                                        # Extract Specific Nairobi Fields
                                        if site_id == 'nairobi':
                                            for k in grid_stats.keys():
                                                if d.get(k): grid_stats[k] = d.get(k)
                                        
                                        success = True
                                        break
                            except Exception as e:
                                print(f"⚠️ Polling attempt {attempt+1} failed for {sn}: {e}", flush=True)
                                time.sleep(1)

                        if not success:
                            inv_data.append({
                                "SN": sn, 
                                "Label": inv_cfg.get('label', sn), 
                                "Type": inv_cfg.get('type'),
                                "OutputPower": 0, "Capacity": 0, "vBat": 0, "temp": 0, "temperature": 0,
                                "high_temperature": False, "has_fault": False, "communication_lost": True
                            })

                    p_min = min(p_caps) if p_caps else 0
                    b_volts = b_data['vBat'] if b_data else 0
                    b_act = b_data['OutputPower'] > 50 if b_data else False

                    current_date = now.strftime('%Y-%m-%d')
                    if managers['daily_accumulator']['last_date'] != current_date:
                        if managers['daily_accumulator']['last_date']:
                            yesterday = now - timedelta(days=1)
                            # Fetch actual historical weather data for yesterday
                            historical_wx = get_historical_weather(config["latitude"], config["longitude"], yesterday)
                            actual_irradiance_wh = calculate_daily_irradiance_potential(historical_wx, yesterday, config)
                            
                            managers['history_manager'].update_daily(
                                managers['daily_accumulator']['last_date'],
                                managers['daily_accumulator']['consumption_wh'],
                                managers['daily_accumulator']['solar_wh'],
                                actual_irradiance_wh
                            )
                            managers['history_manager'].save_history()
                        managers['daily_accumulator'] = {'consumption_wh': 0, 'solar_wh': 0, 'last_date': current_date}

                    interval_hours = POLL_INTERVAL_MINUTES / 60.0
                    managers['daily_accumulator']['consumption_wh'] += tot_out * interval_hours
                    managers['daily_accumulator']['solar_wh'] += tot_sol * interval_hours

                    detected = identify_active_appliances(tot_out, managers['prev_watts'], gen_on, b_volts, p_min, managers['ml_detector'], site_id)
                    is_manual_gen = any("Generator" in x for x in detected)
                    
                    if not is_manual_gen: 
                        managers['load_manager'].update(tot_out)
                    
                    managers['history_manager'].add_hourly_datapoint(now, tot_out, tot_bat, tot_sol, tot_grid)

                    check_alerts(inv_data, None, tot_sol, tot_bat, gen_on, site_id)

                    if now.hour >= 16 and site_id == "kajiado":
                        if tot_bat > 1100:
                            if managers['pool_pump_start_time'] is None:
                                managers['pool_pump_start_time'] = now
                            duration = now - managers['pool_pump_start_time']
                            if duration > timedelta(hours=3) and now.hour >= 18:
                                if managers['pool_pump_last_alert'] is None or (now - managers['pool_pump_last_alert']) > timedelta(hours=1):
                                    duration_hours = int(duration.total_seconds() // 3600)
                                    send_email("⚠️ HIGH LOAD ALERT", f"Battery discharge > 1.1kW for {duration_hours}h.", "high_load_continuous", site_id=site_id)
                                    managers['pool_pump_last_alert'] = now
                        else:
                            managers['pool_pump_start_time'] = None

                    if (now - managers['last_ml_save']) > timedelta(hours=6):
                        managers['ml_detector'].save_model()
                        managers['last_ml_save'] = now

                    if (now - managers['last_save']) > timedelta(hours=1):
                        managers['load_manager'].save_data()
                        managers['last_save'] = now

                    l_cast = managers['load_manager'].get_forecast(24)
                    s_cast = generate_solar_forecast(wx_data, config)

                    breakdown = calculate_battery_breakdown(p_min, b_volts, config)
                    sim_res = calculate_battery_cascade(s_cast, l_cast, p_min, b_volts, config)

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
                                if current_run == 0: temp_start = t
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
                        
                        if best_start and best_end and best_start <= now <= best_end:
                            heavy_loads_safe = True

                    schedule = generate_smart_schedule(
                        status=breakdown['status_obj'],
                        solar_forecast_kw=s_cast, 
                        load_forecast_kw=l_cast,
                        now_hour=now.hour,
                        heavy_loads_safe=heavy_loads_safe,
                        gen_on=gen_on,
                        b_active=b_act,
                        site_id=site_id
                    )
                    del breakdown['status_obj']

                    heatmap = managers['history_manager'].get_last_30_days()
                    hourly_24h = managers['history_manager'].get_last_24h_data()

                    managers['prev_watts'] = tot_out
                    
                    with Lock():
                        site_latest_data[site_id] = {
                            "data": {
                                "timestamp": now.strftime("%H:%M:%S"),
                                "total_output_power": tot_out,
                                "total_solar_input_W": tot_sol,
                                "total_grid_input_W": tot_grid,  # Capture grid wattage
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
                                "heavy_loads_safe": heavy_loads_safe,
                                "grid_stats": grid_stats  # <-- ADDED GRID STATS HERE
                            },
                            "timestamp": now
                        }
                    
                    print(f"Update {site_id}: Load={tot_out}W, Bat={p_min}%", flush=True)

                except Exception as e:
                    print(f"Error processing site {site_id}: {e}", flush=True)

        except Exception as e: 
            print(f"Global Loop Error: {e}", flush=True)
        
        time.sleep(POLL_INTERVAL_MINUTES * 60)

# ----------------------------
# 6. UI & Routes
# ----------------------------
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Solar Monitor Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 400px;
            width: 90%;
        }
        h1 { color: #333; margin-bottom: 10px; font-size: 28px; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 14px; }
        input[type="password"] {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            margin-bottom: 20px;
        }
        input:focus { outline: none; border-color: #667eea; }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .icon { font-size: 48px; text-align: center; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">☀️</div>
        <h1>Solar Monitor</h1>
        <p class="subtitle">Enter your site password</p>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        <form method="POST">
            <input type="password" name="password" placeholder="Site Password" required autofocus>
            <button type="submit">Access Dashboard</button>
        </form>
    </div>
</body>
</html>
"""

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

@app.route('/api/history', methods=['POST'])
def get_history():
    try:
        data = request.json
        site_id = data.get('site_id')
        date_str = data.get('date') # YYYY-MM-DD
        
        if not site_id or not date_str: return jsonify({'error': 'Missing params'}), 400
        
        config = SITES.get(site_id)
        if not config: return jsonify({'error': 'Invalid site'}), 404
        
        token = config['api_token']
        sn = config['serial_numbers'][0] # Use first SN
        
        url = API_HISTORY_URL
        
        # 1. First Request to get total count
        params = {
            'storage_sn': sn,
            'start_date': date_str,
            'end_date': date_str,
            'page': 1,
            'perpage': 100 # Safe limit to get count
        }
        headers = {"token": token, "Content-Type": "application/x-www-form-urlencoded"}
        
        r = requests.post(url, data=params, headers=headers, timeout=10)
        resp = r.json()
        
        if resp.get('error_code') != 0:
            return jsonify({'error': f"API Error: {resp.get('error_msg')}"}), 500
        
        count = int(resp.get('data', {}).get('count', 0))
        if count == 0:
            return jsonify({'error': 'No data for date'}), 404
        
        # 2. Calculate last page
        last_page = math.ceil(count / 100)
        
        # 3. Request last page if needed
        if last_page > 1:
            params['page'] = last_page
            r = requests.post(url, data=params, headers=headers, timeout=10)
            resp = r.json()
            
        records = resp.get('data', {}).get('datas', [])
        if not records:
            return jsonify({'error': 'Data missing in last page'}), 404
        
        last_record = records[-1]
        
        # Extract fields
        keys = [
            "eToUserToday", "eToUserTotal", "eToGridToday", "eToGridTotal",
            "eacChargeToday", "eacChargeTotal", 
            "eBatDisChargeToday", "eBatDisChargeTotal",
            "eacDisChargeToday", "eacDisChargeTotal"
        ]
        result = {k: last_record.get(k, "0") for k in keys}
        
        # Explicit check for eacDisChargeTotal as it is critical for KPLC
        if not result["eacDisChargeTotal"] or result["eacDisChargeTotal"] == "0":
             result["eacDisChargeTotal"] = last_record.get("eacDisChargeTotal", "0")

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data')
def api_data():
    site_id, _ = get_current_site()
    if not site_id: return jsonify({"error": "Unauthorized"}), 401
    
    data = site_latest_data.get(site_id, {}).get('data', {})
    if not data:
        return jsonify({
            "timestamp": "Initializing...", "total_output_power": 0, "total_solar_input_W": 0,
            "primary_battery_min": 0, "backup_battery_voltage": 0, "backup_active": False,
            "generator_running": False, "inverters": [], "detected_appliances": [], 
            "solar_forecast": [], "load_forecast": [], 
            "battery_sim": {"labels": [], "data": [], "tiers": []},
            "energy_breakdown": {
                "chart_data": [1, 0, 1], 
                "total_pct": 0, 
                "total_kwh": 0,
                "tier_labels": ['Primary', 'Backup', 'Reserve'],
                "tier_colors": ['rgba(16, 185, 129, 0.9)', 'rgba(59, 130, 246, 0.8)', 'rgba(245, 158, 11, 0.8)']
            },
            "scheduler": [], "heatmap_data": [], "hourly_24h": [], "ml_status": "Initializing",
            "grid_stats": {"eToUserToday": "0", "eToUserTotal": "0", "eToGridToday": "0", "eToGridTotal": "0"}
        })
    return jsonify(data)

@app.route('/api/ml-feedback', methods=['POST'])
def ml_feedback():
    site_id, _ = get_current_site()
    if not site_id: return jsonify({"error": "Unauthorized"}), 401
    try:
        feedback = request.json
        if feedback and site_id in site_managers:
            site_managers[site_id]['ml_detector'].train_from_feedback(feedback)
            return jsonify({"status": "success", "message": "Feedback received"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error", "message": "No data received"})

@app.route('/api/ml-retrain', methods=['POST'])
def ml_retrain():
    site_id, _ = get_current_site()
    if not site_id: return jsonify({"error": "Unauthorized"}), 401
    try:
        if site_id in site_managers:
            site_managers[site_id]['ml_detector'].save_model()
            return jsonify({"status": "success", "message": "Models saved successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password', '')
        for site_id, config in SITES.items():
            if password == config['password']:
                session['site_id'] = site_id
                return redirect(url_for('home'))
        return render_template_string(LOGIN_TEMPLATE, error="Invalid password")
    return render_template_string(LOGIN_TEMPLATE, error=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/")
def home():
    site_id, site_config = get_current_site()
    if not site_id:
        return redirect(url_for('login'))
        
    d = site_latest_data.get(site_id, {}).get('data', {})
    
    if not d:
         d = {
            "timestamp": "Initializing...", "total_output_power": 0, "total_solar_input_W": 0,
            "primary_battery_min": 0, "backup_battery_voltage": 0, "backup_active": False,
            "generator_running": False, "inverters": [], "detected_appliances": [], 
            "solar_forecast": [], "load_forecast": [], 
            "battery_sim": {"labels": [], "data": [], "tiers": []},
            "energy_breakdown": {
                "chart_data": [1, 0, 1], 
                "total_pct": 0, 
                "total_kwh": 0,
                "tier_labels": ['Primary', 'Backup', 'Reserve'],
                "tier_colors": ['rgba(16, 185, 129, 0.9)', 'rgba(59, 130, 246, 0.8)', 'rgba(245, 158, 11, 0.8)']
            },
            "scheduler": [], "heatmap_data": [], "hourly_24h": [], "ml_status": "Initializing",
            "grid_stats": {
                "eToUserToday": "0", "eToUserTotal": "0", "eToGridToday": "0", "eToGridTotal": "0",
                "eacChargeToday": "0", "eacChargeTotal": "0", 
                "eBatDisChargeToday": "0", "eBatDisChargeTotal": "0",
                "eacDisChargeToday": "0", "eacDisChargeTotal": "0"
            }
        }

    def _n(k): return float(d.get(k, 0) or 0)

    load = _n("total_output_power")
    solar = _n("total_solar_input_W")
    bat_dis = _n("total_battery_discharge_W")
    p_pct = _n("primary_battery_min")
    b_volt = _n("backup_battery_voltage")
    grid_watts = _n("total_grid_input_W")
    gen_on = d.get("generator_running", False)
    detected = d.get("detected_appliances", [])
    heavy_loads_safe = d.get("heavy_loads_safe", False)
    
    # Extract Grid Stats
    grid_stats = d.get("grid_stats", {
        "eToUserToday": "0", "eToUserTotal": "0", "eToGridToday": "0", "eToGridTotal": "0",
        "eacChargeToday": "0", "eacChargeTotal": "0", 
        "eBatDisChargeToday": "0", "eBatDisChargeTotal": "0",
        "eacDisChargeToday": "0", "eacDisChargeTotal": "0"
    })
    
    # Determine actual grid import (threshold for noise)
    is_importing = grid_watts > 20

    breakdown = d.get("energy_breakdown") or {}
    
    breakdown.setdefault("chart_data", [1, 0, 1])
    breakdown.setdefault("tier_labels", ['Primary', 'Backup', 'Reserve'])
    breakdown.setdefault("tier_colors", ['rgba(16, 185, 129, 0.9)', 'rgba(59, 130, 246, 0.8)', 'rgba(245, 158, 11, 0.8)'])
    breakdown.setdefault("total_pct", 0)
    breakdown.setdefault("total_kwh", 0)
    breakdown.setdefault("primary_pct", 0)
    breakdown.setdefault("backup_voltage", 0)
    breakdown.setdefault("backup_pct", 0)
    
    sim = d.get("battery_sim") or {"labels": [], "data": [], "tiers": []}
    s_fc = d.get("solar_forecast") or []
    l_fc = d.get("load_forecast") or []
    schedule = d.get("scheduler") or []
    heatmap = d.get("heatmap_data") or []
    hourly_24h = d.get("hourly_24h") or []

    st_txt, st_col = "NORMAL", "var(--info)"
    
    # NAIROBI SPECIFIC LOGIC
    if site_id == 'nairobi':
        if not gen_on:
            st_txt, st_col = "GRID FAILURE - BATTERY MODE", "var(--crit)"
        elif is_importing:
            st_txt, st_col = "USING UTILITY POWER", "var(--backup-color)"
        else:
            st_txt, st_col = "GRID AVAILABLE - SOLAR MODE", "var(--success)"
    else:
        # KAJIADO / OTHER LOGIC
        if gen_on: 
            st_txt, st_col = "GENERATOR ON", "var(--crit)" 
        elif p_pct < 40: 
            st_txt, st_col = "BACKUP ACTIVE", "var(--warn)"
        elif solar > load + 500: 
            st_txt, st_col = "CHARGING", "var(--success)"

    is_charging = solar > (load + 100)
    is_discharging = bat_dis > 100 or load > solar
    
    now = datetime.now(EAT)
    is_night = (now.hour < 7 or now.hour >= 18)

    alerts = alert_history.get(site_id, [])[:8]
    tier_labels = breakdown.get('tier_labels', ['Primary', 'Backup', 'Reserve'])
    primary_pct = breakdown.get('primary_pct', 0)
    backup_voltage = breakdown.get('backup_voltage', 0)
    backup_pct = breakdown.get('backup_pct', 0)
    
    surplus_power = solar - load
    b_active = d.get("backup_active", False)
    
    schedule_items = []
    
    if s_fc and l_fc:
        best_start, best_end, current_run = None, None, 0
        temp_start = None
        limit = min(len(s_fc), len(l_fc))
        
        for i in range(limit):
            s_item = s_fc[i]
            l_item = l_fc[i]
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
                if current_run == 0: temp_start = t
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
        
        next_3_gen = sum([x['estimated_generation'] for x in s_fc[:3]]) / 3 if len(s_fc) >= 3 else 0
        if next_3_gen < 500 and 8 <= now.hour <= 15:
            schedule_items.append({
                'icon': '☁️',
                'title': 'Cloud Warning',
                'time': 'Low solar next 3 hours',
                'class': 'warning'
            })
    
    recommendation_items = []
    schedule_blocks_heavy = any('No Safe Solar' in item.get('title', '') for item in schedule_items)
    
    if gen_on:
        if site_id == 'nairobi':
            recommendation_items.append({'icon': '✅', 'title': 'UTILITY POWER ACTIVE', 'description': 'Grid power available - normal operation', 'class': 'good'})
        else:
            recommendation_items.append({'icon': '🚨', 'title': 'NO HEAVY LOADS', 'description': 'Generator running - turn off all non-essential appliances', 'class': 'critical'})
    elif site_id == 'nairobi':
        recommendation_items.append({'icon': '🚨', 'title': 'GRID FAILURE', 'description': 'No utility power - running on battery only', 'class': 'critical'})
    elif b_active:
        recommendation_items.append({'icon': '⚠️', 'title': 'MINIMIZE LOADS', 'description': 'Backup battery active - essential loads only', 'class': 'warning'})
    elif is_night:
        recommendation_items.append({'icon': '🌙', 'title': 'NO HEAVY LOADS', 'description': 'Night time - preserve battery life', 'class': 'warning'})
    elif p_pct > 85 and not schedule_blocks_heavy:
        recommendation_items.append({'icon': '✅', 'title': 'SAFE TO USE HEAVY LOADS', 'description': f'Primary battery {p_pct:.0f}% (>85%) allows heavy usage', 'class': 'good'})
    elif heavy_loads_safe:
        recommendation_items.append({'icon': '✅', 'title': 'SAFE TO USE HEAVY LOADS', 'description': f'Inside optimal solar window | Surplus available', 'class': 'good'})
    elif not schedule_blocks_heavy and surplus_power > 1000:
        recommendation_items.append({'icon': '✅', 'title': 'MODERATE LOADS OK', 'description': f'Solar is good, but wait for peak window for heavy loads', 'class': 'good'})
    elif breakdown['total_pct'] < 50 and solar < load:
        recommendation_items.append({'icon': '⚠️', 'title': 'CONSERVE POWER', 'description': f'Battery low ({breakdown["total_pct"]:.0f}%) and not charging well', 'class': 'warning'})
    else:
        recommendation_items.append({'icon': '⚠️', 'title': 'LIMIT HEAVY LOADS', 'description': f'Insufficient net surplus or battery < 85%', 'class': 'warning'})

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ site_config['label'] }}</title>
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
        
        canvas {
            filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.2));
        }
        
        /* New Table Style for KPLC */
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid var(--border); }
        th { color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; }
        td { font-size: 0.9rem; color: var(--text); }
        input[type="number"], input[type="date"] {
            background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text); padding: 8px; border-radius: 6px;
        }
        button { background: var(--accent); color: white; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; }
        button:hover { background: #4f46e5; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>{{ '🏢' if site_config['appliance_type'] == 'office' else '🏠' }} {{ site_config['label']|upper }}</h1>
                <span class="status-badge" style="border-color: {{ st_col }}; color: {{ st_col }}">{{ st_txt }}</span>
            </div>
            <div style="display:flex; align-items:center; gap:20px;">
                <div class="time-display">{{ d['timestamp'] }}</div>
                <a href="/logout" style="background:#e74c3c;color:white;padding:10px 20px;border-radius:8px;text-decoration:none;font-weight:600;font-size:0.9rem;">Logout</a>
            </div>
        </div>

        <div class="grid">
            <div class="col-12 card">
                <div class="card-title">Real-Time Energy Flow</div>
                <div class="flow-diagram">
                    <div class="line line-v l-solar {{ 'flow-down' if solar > 50 else '' }}"><div class="dot"></div></div>
                    <div class="line line-v l-bat {{ 'flow-down' if is_charging else ('flow-up' if is_discharging else '') }}"><div class="dot"></div></div>
                    <div class="line line-h l-home {{ 'flow-right' if load > 100 else '' }}"><div class="dot"></div></div>
                    <div class="line line-h l-gen {{ 'flow-right' if (site_id == 'nairobi' and is_importing) or (site_id != 'nairobi' and gen_on) else '' }}"><div class="dot"></div></div>
                    
                    <div class="node n-solar {{ 'pulse-y' if solar > 50 else '' }}">
                        <div class="node-icon">☀️</div>
                        <div class="node-val">{{ '%0.f'|format(solar) }}W</div>
                    </div>
                    <div class="node n-gen {{ 'pulse-r' if (site_id != 'nairobi' and gen_on) else '' }}" style="{{ 'border-color:var(--info);' if site_id == 'nairobi' and gen_on else '' }}">
                        <div class="node-icon">{{ '🏭' if site_id == 'nairobi' else '⚙️' }}</div>
                        <div class="node-val">
                            {% if site_id == 'nairobi' %}
                                {{ '%0.f'|format(grid_watts) }}W
                            {% else %}
                                {{ 'ON' if gen_on else 'OFF' }}
                            {% endif %}
                        </div>
                    </div>
                    <div style="position: absolute; top: 110px; left: 5px; font-size: 0.7rem; color: var(--text-muted);">
                        {{ 'KPLC Grid' if site_id == 'nairobi' else 'Generator' }}
                    </div>
                    <div class="node n-inv">
                        <div class="node-icon">⚡</div>
                        <div class="node-val">INV</div>
                    </div>
                    <div class="node n-home {{ 'pulse-g' if load > 2000 else '' }}">
                        <div class="node-icon">{{ '🏢' if site_config['appliance_type'] == 'office' else '🏠' }}</div>
                        <div class="node-val">{{ '%0.f'|format(load) }}W</div>
                    </div>
                    <div class="node n-bat {{ 'pulse-g' if is_charging else ('pulse-r' if is_discharging else '') }}">
                        <div class="node-icon">🔋</div>
                        <div class="node-val">{{ breakdown['total_pct'] }}%</div>
                    </div>
                </div>
            </div>

            <!-- NAIROBI SPECIFIC: KPLC BILL & GRID ENERGY STATS -->
            {% if site_id == 'nairobi' %}
            <div class="col-12 card">
                <h3>⚡ KPLC Bill Estimator</h3>
                <div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:20px; align-items:center;">
                    <div>
                        <label style="font-size:0.8rem; color:var(--text-muted)">Cost per kWh (KES)</label><br>
                        <input type="number" id="kplcCost" value="35" onchange="saveCost()" style="width:100px; margin-top:5px;">
                    </div>
                    <div>
                        <label style="font-size:0.8rem; color:var(--text-muted)">Current Month Est. (1st - Today)</label><br>
                        <div id="currMonthEst" style="font-weight:700; font-size:1.4rem; color:var(--accent)">Loading...</div>
                    </div>
                </div>
                
                <h4 style="margin-bottom:5px; color:var(--text-dim)">Historical Usage (Past Month)</h4>
                <div id="kplcHistoryTable" style="margin-bottom:20px;">Loading history...</div>

                <div style="padding-top:15px; border-top:1px solid var(--border)">
                    <h4 style="color:var(--text-dim)">Custom Date Calculator</h4>
                    <div style="display:flex; gap:10px; align-items:center; margin-top:10px;">
                        <input type="date" id="calcStart"> <span style="font-weight:bold">to</span> <input type="date" id="calcEnd">
                        <button onclick="calculateCustomKPLC()">Calculate Cost</button>
                    </div>
                    <div id="calcResult" style="margin-top:10px; font-weight:700; color:var(--success); font-size:1.1rem"></div>
                </div>
            </div>
            {% endif %}

            <div class="col-4 card">
                <div class="card-title">Solar Generation</div>
                <div class="metric-val" style="color:var(--warn)">{{ '%0.f'|format(solar) }}<span style="font-size:1.2rem">W</span></div>
                <div class="metric-unit">Current Input</div>
            </div>
            <div class="col-4 card">
                <div class="card-title">{{ 'Office Consumption' if site_config['appliance_type'] == 'office' else 'Home Consumption' }}</div>
                <div class="metric-val" style="color:var(--info)">{{ '%0.f'|format(load) }}<span style="font-size:1.2rem">W</span></div>
                <div class="metric-unit">Active Load</div>
            </div>
            <div class="col-4 card">
                <div class="card-title">Battery Status</div>
                <div class="metric-val" style="color:var(--success)">{{ breakdown['total_pct'] }}<span style="font-size:1.2rem">%</span></div>
                <div class="metric-unit">{{ breakdown['total_kwh'] }} kWh Usable</div>
            </div>

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

            <div class="col-12 card">
                <div class="card-title">Appliance Simulator</div>
                <div class="sched-grid">
                    {% if site_config['appliance_type'] == 'office' %}
                        {% set sim_items = [
                            ('Desktop Computer', 200, '🖥️'), ('Coffee Maker', 1200, '☕'), ('Laser Printer', 800, '🖨️')
                        ] %}
                    {% else %}
                        {% set sim_items = [
                            ('Washing Machine', 800, '🧺'), ('Electric Oven', 2500, '🍳'), ('Water Heater', 3000, '🚿')
                        ] %}
                    {% endif %}
                    
                    {% for name, watts, icon in sim_items %}
                    <div class="sched-tile" onclick="toggleSim('{{ name|replace(' ', '_') }}', {{ watts }})" id="btn-{{ name|replace(' ', '_') }}">
                        <div class="tile-icon">{{ icon }}</div>
                        <div class="tile-name">{{ name }}</div>
                        <div class="tile-status">{{ watts }}W</div>
                    </div>
                    {% endfor %}
                </div>
                <div style="margin-top:15px; text-align:right; font-size:0.9rem; color: var(--text-muted)">
                    Simulated Additional Load: <span id="sim-val" style="color: var(--accent); font-weight: 700">0W</span>
                </div>
            </div>

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

            <div class="col-8 card">
                <div class="card-title">24-Hour Battery Projection (Tier-Aware)</div>
                <div style="height:280px"><canvas id="simChart"></canvas></div>
            </div>

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
                    {% if site_config['backup_battery_wh'] > 0 %}
                    <div>Backup: {{ backup_voltage }}V ({{ backup_pct }}%)</div>
                    {% endif %}
                </div>
            </div>

            <div class="col-12 card">
                <div class="card-title">Last 24 Hours: Load vs Battery Discharge vs Solar</div>
                <div style="height:300px"><canvas id="hourlyChart"></canvas></div>
                <div style="margin-top:10px; text-align:right; font-size:0.75rem; color: var(--text-muted)">
                    Use mouse wheel or pinch to zoom | Drag to pan | Double-click to reset
                </div>
            </div>

            <div class="col-12 card">
                <div class="card-title">{{ 'Office Activity Detection' if site_config['appliance_type'] == 'office' else 'House Activity Detection' }}</div>
                <div style="margin-bottom:20px">
                    {% if detected %}
                        {% for a in detected %}
                            {% if 'House 1' in a %}
                                <span class="tag house1">{{ '🏢' if site_config['appliance_type'] == 'office' else '🏠' }} {{ a }}</span>
                            {% elif 'House 2' in a %}
                                <span class="tag house2">{{ '🏢' if site_config['appliance_type'] == 'office' else '🏠' }} {{ a }}</span>
                            {% elif 'Water' in a or 'Generator' in a or 'Utility' in a %}
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
        const SITE_ID = '{{ site_id }}';
        
        // --- Persistence for Cost ---
        if(localStorage.getItem('kplcCost')) {
            const savedCost = localStorage.getItem('kplcCost');
            const costInput = document.getElementById('kplcCost');
            if(costInput) costInput.value = savedCost;
        }
        
        function saveCost() { 
            const val = document.getElementById('kplcCost').value;
            localStorage.setItem('kplcCost', val); 
            loadKplcStats(); 
        }

        // --- Helper to fetch single date total (using eacDisChargeTotal for billable grid use) ---
        async function fetchTotalForDate(dateStr) {
            try {
                const r = await fetch('/api/history', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({site_id: SITE_ID, date: dateStr}) });
                const d = await r.json();
                // eacDisChargeTotal is "Grid total bypass load energy" - usually main grid consumption
                return d.error ? 0 : parseFloat(d.eacDisChargeTotal || 0);
            } catch { return 0; }
        }

     // --- Main Logic for Monthly Stats ---
async function loadKplcStats() {
    if(SITE_ID !== 'nairobi') return;
    const costPerUnit = parseFloat(document.getElementById('kplcCost').value) || 0;
    const now = new Date();
    const y = now.getFullYear(), m = now.getMonth(); // 0-indexed
    
    // Helper to get YYYY-MM-DD in local time
    const fmt = d => {
        const offset = d.getTimezoneOffset() * 60000;
        return new Date(d.getTime() - offset).toISOString().split('T')[0];
    }

    // We need: Today, 1st of This Month, 1st of Last Month, 1st of Month Before Last
    const dates = [
        fmt(now), // Today
        fmt(new Date(y, m, 1)), // 1st Current Month
        fmt(new Date(y, m-1, 1)), // 1st Last Month
        fmt(new Date(y, m-2, 1))  // 1st Month Before Last
    ];

    // Fetch in parallel
    document.getElementById('currMonthEst').innerText = "Loading...";
    document.getElementById('kplcHistoryTable').innerText = "Loading history...";
    
    const vals = await Promise.all(dates.map(d => fetchTotalForDate(d)));
    
    // Current Month Calculation (Today - Start of Current Month)
    let currUsage = 0;
    if(vals[0] > 0 && vals[1] > 0) currUsage = vals[0] - vals[1];
    document.getElementById('currMonthEst').innerHTML = `${currUsage.toFixed(1)} kWh <span style="font-size:0.9rem; color:var(--text-muted)">~ ${Math.round(currUsage * costPerUnit).toLocaleString()} KES</span>`;

    // History Table - Show actual months with correct labels
    let html = '<table><tr><th>Month</th><th>Usage (kWh)</th><th>Est. Cost (KES)</th></tr>';

    // Last Month (Actual - e.g., December if today is January)
    const lastMonthUsage = (vals[1] > 0 && vals[2] > 0) ? vals[1] - vals[2] : 0;
    const lastMonthDate = new Date(y, m-1, 1);
    html += `<tr><td>${lastMonthDate.toLocaleString('default', { month: 'long', year: 'numeric' })}</td><td>${lastMonthUsage.toFixed(1)}</td><td>${Math.round(lastMonthUsage * costPerUnit).toLocaleString()}</td></tr>`;

    // Month Before Last (Actual - e.g., November if today is January)
    const monthBeforeUsage = (vals[2] > 0 && vals[3] > 0) ? vals[2] - vals[3] : 0;
    const monthBeforeDate = new Date(y, m-2, 1);
    html += `<tr><td>${monthBeforeDate.toLocaleString('default', { month: 'long', year: 'numeric' })}</td><td>${monthBeforeUsage.toFixed(1)}</td><td>${Math.round(monthBeforeUsage * costPerUnit).toLocaleString()}</td></tr>`;

    html += '</table>';
    document.getElementById('kplcHistoryTable').innerHTML = html;
}
        async function calculateCustomKPLC() {
            const start = document.getElementById('calcStart').value;
            const end = document.getElementById('calcEnd').value;
            const cost = parseFloat(document.getElementById('kplcCost').value) || 0;
            const div = document.getElementById('calcResult');
            
            if(!start || !end) { div.innerHTML = "Select dates"; return; }
            div.innerHTML = "Fetching...";
            
            const vStart = await fetchTotalForDate(start);
            const vEnd = await fetchTotalForDate(end);
            
            if(vStart > 0 && vEnd > 0) {
                const diff = vEnd - vStart;
                div.innerHTML = `${diff.toFixed(1)} kWh  =  ${Math.round(diff * cost).toLocaleString()} KES`;
            } else {
                div.innerHTML = "Data unavailable for range";
            }
        }

        // --- Existing Chart Logic & Scripts ---
        const labels = {{ sim['labels']|tojson }};
        const baseData = {{ sim['data']|tojson }};
        const tierData = {{ sim['tiers']|tojson }};
        const sForecast = {{ s_fc|tojson }};
        const lForecast = {{ l_fc|tojson }};
        const pieData = {{ breakdown['chart_data']|tojson }};
        const tierLabels = {{ tier_labels|tojson }};
        const pieColors = {{ breakdown['tier_colors']|tojson }}; 
        const hourly24h = {{ hourly_24h|tojson }};
        
        let activeSims = {};
        let simTierData = [];
        
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.2)';
        Chart.defaults.font.family = "'Manrope', sans-serif";
        
        function fetchHistory() {
            const date = document.getElementById('histDate').value;
            if(!date) return alert("Select a date");
            
            const btn = document.querySelector('button[onclick="fetchHistory()"]');
            const originalText = btn.innerText;
            btn.innerText = "Loading...";
            
            fetch('/api/history', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({site_id: '{{ site_id }}', date: date})
            })
            .then(r => r.json())
            .then(d => {
                btn.innerText = originalText;
                if(d.error) {
                    alert(d.error);
                    return;
                }
                
                const resDiv = document.getElementById('histResult');
                resDiv.style.display = 'grid';
                // Using string concatenation to ensure compatibility
                resDiv.innerHTML = 
                    '<div style="background:rgba(59,130,246,0.1); padding:10px; border-radius:8px;"><div>Import Today</div><b>' + d.eToUserToday + ' kWh</b></div>' +
                    '<div style="background:rgba(59,130,246,0.1); padding:10px; border-radius:8px;"><div>Import Total</div><b>' + d.eToUserTotal + ' kWh</b></div>' +
                    '<div style="background:rgba(16,185,129,0.1); padding:10px; border-radius:8px;"><div>Export Today</div><b>' + d.eToGridToday + ' kWh</b></div>' +
                    '<div style="background:rgba(16,185,129,0.1); padding:10px; border-radius:8px;"><div>Export Total</div><b>' + d.eToGridTotal + ' kWh</b></div>' +
                    '<div style="background:rgba(99,102,241,0.1); padding:10px; border-radius:8px;"><div>AC Charge Today</div><b>' + d.eacChargeToday + ' kWh</b></div>' +
                    '<div style="background:rgba(99,102,241,0.1); padding:10px; border-radius:8px;"><div>AC Charge Total</div><b>' + d.eacChargeTotal + ' kWh</b></div>' +
                    '<div style="background:rgba(245,158,11,0.1); padding:10px; border-radius:8px;"><div>Bat Dischg Today</div><b>' + d.eBatDisChargeToday + ' kWh</b></div>' +
                    '<div style="background:rgba(245,158,11,0.1); padding:10px; border-radius:8px;"><div>Bat Dischg Total</div><b>' + d.eBatDisChargeTotal + ' kWh</b></div>' +
                    '<div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:8px;"><div>Grid Bypass Today</div><b>' + d.eacDisChargeToday + ' kWh</b></div>' +
                    '<div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:8px;"><div>Grid Bypass Total</div><b>' + d.eacDisChargeTotal + ' kWh</b></div>';
            })
            .catch(e => {
                btn.innerText = originalText;
                alert("Error fetching data");
            });
        }
        
        const ctx = document.getElementById('simChart');
        
        const borderColors = tierData.map(tier => {
            if (tier === 'primary') return '#10b981'; 
            if (tier === 'backup') return '#3b82f6';
            if (tier === 'reserve') return '#f59e0b';
            return '#64748b';
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
                            filter: (item, chart) => item.text !== 'Battery Level' 
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

                                if (context.datasetIndex === 1 && simTierData.length > 0) {
                                    tier = simTierData[context.dataIndex];
                                    scenarioLabel = "With Load";
                                } else {
                                    tier = tierData[context.dataIndex];
                                    scenarioLabel = "Normal";
                                }

                                tier = tier || 'unknown';
                                const tierName = tier.charAt(0).toUpperCase() + tier.slice(1);
                                
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

        new Chart(document.getElementById('pieChart'), {
            type: 'doughnut',
            data: {
                labels: tierLabels,
                datasets: [{
                    data: pieData,
                    backgroundColor: pieColors,
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

        const hourlyLabels = hourly24h.map(d => {
            const dt = new Date(d.timestamp);
            return dt.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});
        });
        const loadData = hourly24h.map(d => d.load);
        const dischargeData = hourly24h.map(d => d.battery_discharge);
        const solarData = hourly24h.map(d => d.solar);
        const gridGenData = hourly24h.map(d => d.grid_gen || 0);

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
                    },
                    {
                        label: 'Grid/Gen (W)',
                        data: gridGenData,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
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

        hourlyCtx.ondblclick = function() {
            hourlyChart.resetZoom();
        };

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

            const P_TOTAL = {{ site_config["primary_battery_wh"] }};
            const B_TOTAL = {{ site_config["backup_battery_wh"] }} * {{ site_config["backup_degradation"] }};
            const TOTAL_CAPACITY = (P_TOTAL * 0.60) + (B_TOTAL * 0.80) + (P_TOTAL * 0.20); 
            
            let curr_p_wh = ({{ p_pct }} / 100.0) * P_TOTAL;
            
            let curr_b_wh = 0;
            if (B_TOTAL > 0) {
                curr_b_wh = ({{ backup_pct }} / 100.0) * B_TOTAL;
            }
            
            let simCurve = [ baseData[0] ];
            let newSimTiers = [ tierData[0] ];
            
            for (let i = 0; i < 24; i++) {
                const solar_gen = sForecast[i]?.estimated_generation || 0;
                const base_load = lForecast[i]?.estimated_load || {{ load }};
                
                const total_load = base_load + totalSimWatts;
                const net_flow = solar_gen - total_load;
                
                if (net_flow > 0) { 
                    let charge_amount = net_flow;
                    
                    const primary_space = P_TOTAL - curr_p_wh;
                    if (primary_space > 0) {
                        const charge_to_primary = Math.min(charge_amount, primary_space);
                        curr_p_wh += charge_to_primary;
                        charge_amount -= charge_to_primary;
                    }
                    
                    if (charge_amount > 0 && B_TOTAL > 0) {
                        const backup_space = B_TOTAL - curr_b_wh;
                        if (backup_space > 0) {
                            const charge_to_backup = Math.min(charge_amount, backup_space);
                            curr_b_wh += charge_to_backup;
                        }
                    }
                }
                else { 
                    let drain = Math.abs(net_flow);
                    
                    let primary_min = P_TOTAL * 0.40;
                    let available_tier1 = Math.max(0, curr_p_wh - primary_min);
                    
                    if (available_tier1 >= drain) {
                        curr_p_wh -= drain;
                        drain = 0;
                    } else {
                        curr_p_wh = primary_min;
                        drain -= available_tier1;
                    }
                    
                    if (drain > 0 && B_TOTAL > 0) {
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
                
                let primary_tier1_avail = Math.max(0, curr_p_wh - (P_TOTAL * 0.40));
                let backup_avail = 0;
                if (B_TOTAL > 0) {
                    backup_avail = Math.max(0, curr_b_wh - (B_TOTAL * 0.20));
                }
                let emergency_avail = Math.max(0, Math.min(curr_p_wh, P_TOTAL * 0.40) - (P_TOTAL * 0.20));
                
                let total_available = primary_tier1_avail + backup_avail + emergency_avail;
                let percentage = (total_available / TOTAL_CAPACITY) * 100;

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
        
        fetch('/health').then(r => r.json()).then(d => { 
            if(!d.polling_thread_alive) fetch('/start-polling'); 
        });
        
        // Trigger load logic immediately if viewing Nairobi
        if(SITE_ID === 'nairobi') loadKplcStats();
        
        setTimeout(() => location.reload(), 360000); 
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
        heavy_loads_safe=heavy_loads_safe, site_config=site_config, site_id=site_id,
        grid_watts=grid_watts, is_importing=is_importing,
        grid_stats=grid_stats
    )

if __name__ == '__main__':
    for file in [DATA_FILE, HISTORY_FILE, ML_MODEL_FILE]:
        if not Path(file).exists():
            Path(file).touch()
    
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
