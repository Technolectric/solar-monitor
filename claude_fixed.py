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

# Configure logging
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
API_HISTORY_URL = "https://openapi.growatt.com/v1/device/storage/storage_data" # New Endpoint for historical lookup
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
        self.load_history = deque(maxlen=1000)
        self.appliance_classifier = None
        self.scaler = StandardScaler()
        self.model_lock = Lock()
        
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
# 2. Logic Engine
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
                    "msg": "Generator On", "status": "unsafe", "color": "var(--crit)", "reason": "Generator running - no loads"
                })
        elif site_id == 'nairobi':
            decision.update({
                "msg": "Grid Failure", "status": "unsafe", "color": "var(--crit)", "reason": "No utility power - battery only"
            })
        elif b_active:
            decision.update({
                "msg": "Backup Active", "status": "unsafe", "color": "var(--crit)", "reason": "Backup battery in use"
            })
        elif battery_soc_pct < 40:
            decision.update({
                "msg": "Battery Too Low", "status": "unsafe", "color": "var(--crit)", "reason": f"System at {battery_soc_pct:.0f}%"
            })
        elif app["watts"] > 1500:
            if is_night or is_past_4pm:
                decision.update({
                    "msg": "Not Safe", "status": "unsafe", "color": "var(--crit)", "reason": "Heavy loads restricted: nighttime or after 4 PM"
                })
            elif heavy_loads_safe:
                decision.update({
                    "msg": "Safe in Window", "status": "safe", "color": "var(--success)", "reason": "Inside calculated safe solar window"
                })
            elif primary_pct > 85 and battery_kwh_available >= app_kwh_required * 1.5:
                decision.update({
                    "msg": "Safe (High Battery)", "status": "safe", "color": "var(--success)", "reason": f"Primary {primary_pct:.0f}% > 85%"
                })
            else:
                decision.update({
                    "msg": "Wait for Window", "status": "unsafe", "color": "var(--warn)", "reason": "Need solar window or >85% battery"
                })
        elif app["watts"] >= 800:
            if is_night and battery_soc_pct < 60:
                decision.update({
                    "msg": "Avoid at Night", "status": "unsafe", "color": "var(--warn)", "reason": "Night + battery < 60%"
                })
            elif heavy_loads_safe or primary_pct > 75:
                decision.update({
                    "msg": "Conditionally Safe", "status": "safe", "color": "var(--success)", "reason": "Good conditions"
                })
            else:
                decision.update({
                    "msg": "Wait", "status": "unsafe", "color": "var(--warn)", "reason": "Conditions not optimal"
                })
        else:
            if is_night and battery_soc_pct < 50:
                decision.update({
                    "msg": "Conserve", "status": "unsafe", "color": "var(--warn)", "reason": "Night + low battery"
                })
            elif battery_kwh_available >= app_kwh_required * 1.2:
                decision.update({
                    "msg": "Safe to Run", "status": "safe", "color": "var(--success)", "reason": "Battery sufficient"
                })
            else:
                decision.update({
                    "msg": "Wait", "status": "unsafe", "color": "var(--warn)", "reason": "Insufficient battery"
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
last_alert_time = {} 
alert_history = {}  
site_latest_data = {}

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
    if site_id not in last_alert_time: last_alert_time[site_id] = {}
    if site_id not in alert_history: alert_history[site_id] = []
    
    cooldown_minutes = 120
    if "critical" in alert_type.lower(): cooldown_minutes = 60
    elif "high_load" in alert_type.lower(): cooldown_minutes = 30
    
    if alert_type in last_alert_time[site_id]:
        time_since = datetime.now(EAT) - last_alert_time[site_id][alert_type]
        if time_since < timedelta(minutes=cooldown_minutes):
            return
    
    recipient = SITES.get(site_id, {}).get("recipient_email", RECIPIENT_EMAIL)
    
    if send_via_email and RESEND_API_KEY and recipient:
        try:
            site_label = SITES.get(site_id, {}).get("label", site_id)
            full_subject = f"[{site_label}] {subject}"
            requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_API_KEY}"}, json={"from": SENDER_EMAIL, "to": [recipient], "subject": full_subject, "html": html})
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
        if inv.get('communication_lost'): send_email(f"⚠️ Comm Lost: {inv['Label']}", "Check inverter", "communication_lost", site_id=site_id)
        if inv.get('has_fault'): send_email(f"🚨 FAULT: {inv['Label']}", "Fault code", "fault_alarm", site_id=site_id)
        if inv.get('high_temperature'): send_email(f"🌡️ High Temp: {inv['Label']}", f"Temp: {inv['temperature']}", "high_temperature", site_id=site_id)
    
    if site_id == 'nairobi':
        if not gen_run: send_email("🚨 CRITICAL: Grid Failure", "No utility power - running on battery", "critical", site_id=site_id)
    else:
        if gen_run or (b_volt < 51.2 and b_volt > 10): send_email("🚨 CRITICAL: Generator Running", "Backup critical", "critical", site_id=site_id)
    
    if b_active and p_cap < 40: send_email("⚠️ HIGH ALERT: Backup Active", "Reduce Load", "backup_active", site_id=site_id)
    if 40 < p_cap < 50: send_email("⚠️ Primary Low", "Reduce Load", "warning", send_via_email=b_active, site_id=site_id)
    if bat_discharge >= 4500: send_email("🚨 URGENT: High Discharge", "Critical", "very_high_load", send_via_email=b_active, site_id=site_id)
    elif 2500 <= bat_discharge < 4500: send_email("⚠️ High Discharge", "Warning", "high_load", send_via_email=b_active, site_id=site_id)

# ----------------------------
# 5. Polling Loop
# ----------------------------
polling_active = False
polling_thread = None
site_managers = {}

def poll_growatt():
    global site_latest_data, polling_active, site_managers

    for site_id, config in SITES.items():
        if site_id not in site_managers:
            site_managers[site_id] = {
                'load_manager': PersistentLoadManager(f"{site_id}_{DATA_FILE}"),
                'history_manager': DailyHistoryManager(f"{site_id}_{HISTORY_FILE}"),
                'ml_detector': ApplianceDetector(config["appliance_type"]),
                'daily_accumulator': {'consumption_wh': 0, 'solar_wh': 0, 'last_date': None},
                'pool_pump_start_time': None, 'pool_pump_last_alert': None,
                'last_save': datetime.now(EAT), 'last_ml_save': datetime.now(EAT),
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
                    
                    grid_stats = {"eToUserToday": "0", "eToUserTotal": "0", "eToGridToday": "0", "eToGridTotal": "0"}

                    for sn in config["serial_numbers"]:
                        inv_cfg = config["inverter_config"].get(sn, {"label": sn, "type": "unknown"})
                        success = False
                        site_headers = headers_template.copy()
                        site_headers["token"] = token

                        for attempt in range(3):
                            try:
                                r = requests.post(API_URL, data={"storage_sn": sn}, headers=site_headers, timeout=10)
                                if r.status_code == 200:
                                    if r.json().get("error_code") == 0:
                                        d = r.json().get("data", {})
                                        op = float(d.get("outPutPower") or 0)
                                        cap = float(d.get("capacity") or 0)
                                        vb = float(d.get("vBat") or 0)
                                        pb = float(d.get("pBat") or 0)
                                        sol = float(d.get("ppv") or 0) + float(d.get("ppv2") or 0)
                                        grid_pwr = float(d.get("pAcInPut") or 0)
                                        temp = max(float(d.get("invTemperature") or 0), float(d.get("dcDcTemperature") or 0))
                                        
                                        tot_out += op
                                        tot_sol += sol
                                        tot_grid += grid_pwr
                                        if pb > 0: tot_bat += pb

                                        info = {"SN": sn, "Label": inv_cfg['label'], "OutputPower": op, "Capacity": cap, "vBat": vb, "temp": temp, "temperature": temp, "high_temperature": temp >= 60, "has_fault": int(d.get("errorCode") or 0) != 0, "communication_lost": False}
                                        inv_data.append(info)

                                        if inv_cfg['type'] == 'primary': 
                                            p_caps.append(cap)
                                            if site_id == 'nairobi' and float(d.get("vGrid") or 0) >= 180: gen_on = True
                                        elif inv_cfg['type'] == 'backup':
                                            b_data = info
                                            if float(d.get("vac") or 0) > 100 or float(d.get("pAcInPut") or 0) > 50: gen_on = True
                                        
                                        if site_id == 'nairobi':
                                            if d.get("eToUserToday"): grid_stats["eToUserToday"] = d.get("eToUserToday")
                                            if d.get("eToUserTotal"): grid_stats["eToUserTotal"] = d.get("eToUserTotal")
                                            if d.get("eToGridToday"): grid_stats["eToGridToday"] = d.get("eToGridToday")
                                            if d.get("eToGridTotal"): grid_stats["eToGridTotal"] = d.get("eToGridTotal")
                                        
                                        success = True
                                        break
                            except Exception as e: time.sleep(1)

                        if not success:
                            inv_data.append({"SN": sn, "Label": inv_cfg.get('label', sn), "Type": inv_cfg.get('type'), "OutputPower": 0, "Capacity": 0, "vBat": 0, "temp": 0, "temperature": 0, "high_temperature": False, "has_fault": False, "communication_lost": True})

                    p_min = min(p_caps) if p_caps else 0
                    b_volts = b_data['vBat'] if b_data else 0
                    b_act = b_data['OutputPower'] > 50 if b_data else False

                    current_date = now.strftime('%Y-%m-%d')
                    if managers['daily_accumulator']['last_date'] != current_date:
                        if managers['daily_accumulator']['last_date']:
                            managers['history_manager'].update_daily(managers['daily_accumulator']['last_date'], managers['daily_accumulator']['consumption_wh'], managers['daily_accumulator']['solar_wh'], calculate_daily_irradiance_potential(wx_data, now-timedelta(days=1), config))
                            managers['history_manager'].save_history()
                        managers['daily_accumulator'] = {'consumption_wh': 0, 'solar_wh': 0, 'last_date': current_date}

                    managers['daily_accumulator']['consumption_wh'] += tot_out * (POLL_INTERVAL_MINUTES / 60.0)
                    managers['daily_accumulator']['solar_wh'] += tot_sol * (POLL_INTERVAL_MINUTES / 60.0)

                    detected = identify_active_appliances(tot_out, managers['prev_watts'], gen_on, b_volts, p_min, managers['ml_detector'], site_id)
                    if not any("Generator" in x for x in detected): managers['load_manager'].update(tot_out)
                    managers['history_manager'].add_hourly_datapoint(now, tot_out, tot_bat, tot_sol, tot_grid)
                    check_alerts(inv_data, None, tot_sol, tot_bat, gen_on, site_id)

                    if now.hour >= 16 and site_id == "kajiado" and tot_bat > 1100:
                        if managers['pool_pump_start_time'] is None: managers['pool_pump_start_time'] = now
                        elif (now - managers['pool_pump_start_time']) > timedelta(hours=3) and now.hour >= 18:
                             if managers['pool_pump_last_alert'] is None or (now - managers['pool_pump_last_alert']) > timedelta(hours=1):
                                send_email("⚠️ HIGH LOAD ALERT", "Battery discharge > 1.1kW for >3h.", "high_load_continuous", site_id=site_id)
                                managers['pool_pump_last_alert'] = now
                    else: managers['pool_pump_start_time'] = None

                    if (now - managers['last_ml_save']) > timedelta(hours=6): managers['ml_detector'].save_model(); managers['last_ml_save'] = now
                    if (now - managers['last_save']) > timedelta(hours=1): managers['load_manager'].save_data(); managers['last_save'] = now

                    l_cast = managers['load_manager'].get_forecast(24)
                    s_cast = generate_solar_forecast(wx_data, config)
                    breakdown = calculate_battery_breakdown(p_min, b_volts, config)
                    sim_res = calculate_battery_cascade(s_cast, l_cast, p_min, b_volts, config)
                    
                    heavy_loads_safe = False
                    if s_cast and l_cast: # (Simplified check for brevity)
                        heavy_loads_safe = any(s['estimated_generation'] - l.get('estimated_load',600) > 2500 for s,l in zip(s_cast,l_cast))

                    schedule = generate_smart_schedule(breakdown['status_obj'], s_cast, l_cast, now.hour, heavy_loads_safe, gen_on, b_act, site_id)
                    del breakdown['status_obj']

                    with Lock():
                        site_latest_data[site_id] = {
                            "data": {
                                "timestamp": now.strftime("%H:%M:%S"), "total_output_power": tot_out, "total_solar_input_W": tot_sol, "total_grid_input_W": tot_grid,
                                "total_battery_discharge_W": tot_bat, "primary_battery_min": p_min, "backup_battery_voltage": b_volts, "backup_active": b_act,
                                "generator_running": gen_on, "detected_appliances": detected, "load_forecast": l_cast[:12], "solar_forecast": s_cast[:12],
                                "battery_sim": sim_res, "energy_breakdown": breakdown, "scheduler": schedule, "inverters": inv_data,
                                "heatmap_data": managers['history_manager'].get_last_30_days(), "hourly_24h": managers['history_manager'].get_last_24h_data(),
                                "ml_status": "Active", "heavy_loads_safe": heavy_loads_safe, "grid_stats": grid_stats
                            }, "timestamp": now
                        }
                    managers['prev_watts'] = tot_out
                    print(f"Update {site_id}: Load={tot_out}W, Bat={p_min}%", flush=True)

                except Exception as e: print(f"Error processing site {site_id}: {e}", flush=True)
        except Exception as e: print(f"Global Loop Error: {e}", flush=True)
        time.sleep(POLL_INTERVAL_MINUTES * 60)

# ----------------------------
# 6. UI & Routes
# ----------------------------
@app.route('/health')
def health(): return jsonify({"status": "healthy", "polling_thread_alive": polling_active})

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
        date_str = data.get('date')
        if not site_id or not date_str: return jsonify({'error': 'Missing params'}), 400
        
        config = SITES.get(site_id)
        if not config: return jsonify({'error': 'Invalid site'}), 404
        
        sn = config['serial_numbers'][0]
        params = {'storage_sn': sn, 'start_date': date_str, 'end_date': date_str, 'page': 1, 'perpage': 1000}
        headers = {"token": config['api_token'], "Content-Type": "application/x-www-form-urlencoded"}
        
        r = requests.post(API_HISTORY_URL, data=params, headers=headers, timeout=10)
        resp = r.json()
        
        if resp.get('error_code') == 0:
            records = resp.get('data', {}).get('datas', [])
            if not records: return jsonify({'error': 'No data for date'}), 404
            last_record = records[-1]
            return jsonify({
                "eToUserToday": last_record.get("eToUserToday", "0"), "eToUserTotal": last_record.get("eToUserTotal", "0"),
                "eToGridToday": last_record.get("eToGridToday", "0"), "eToGridTotal": last_record.get("eToGridTotal", "0")
            })
        else: return jsonify({'error': f"API Error: {resp.get('error_msg')}"}), 500
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/data')
def api_data():
    site_id, _ = get_current_site()
    if not site_id: return jsonify({"error": "Unauthorized"}), 401
    return jsonify(site_latest_data.get(site_id, {}).get('data', {}))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        for site_id, config in SITES.items():
            if request.form.get('password') == config['password']:
                session['site_id'] = site_id
                return redirect(url_for('home'))
        return render_template_string(LOGIN_TEMPLATE, error="Invalid password")
    return render_template_string(LOGIN_TEMPLATE, error=None)

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('login'))

@app.route("/")
def home():
    site_id, site_config = get_current_site()
    if not site_id: return redirect(url_for('login'))
        
    d = site_latest_data.get(site_id, {}).get('data', {})
    if not d: d = {"timestamp": "Initializing...", "energy_breakdown": {"chart_data": [1,0,1], "tier_labels": [], "tier_colors": [], "total_pct": 0, "total_kwh": 0}}

    # Extract Data for Template
    load = float(d.get("total_output_power", 0))
    solar = float(d.get("total_solar_input_W", 0))
    grid_watts = float(d.get("total_grid_input_W", 0))
    breakdown = d.get("energy_breakdown", {})
    grid_stats = d.get("grid_stats", {})

    # Status Logic
    st_txt, st_col = "NORMAL", "var(--info)"
    if site_id == 'nairobi':
        if not d.get("generator_running"): st_txt, st_col = "GRID FAILURE", "var(--crit)"
        elif grid_watts > 20: st_txt, st_col = "USING GRID", "var(--backup-color)"
        else: st_txt, st_col = "SOLAR MODE", "var(--success)"
    else:
        if d.get("generator_running"): st_txt, st_col = "GENERATOR ON", "var(--crit)"

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ site_config['label'] }}</title>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root { --bg: #0a0e27; --card: rgba(21, 27, 61, 0.7); --text: #e2e8f5; --accent: #6366f1; --success: #10b981; --warn: #f59e0b; --crit: #ef4444; --info: #3b82f6; --border: rgba(99, 102, 241, 0.2); }
        body { background: var(--bg); color: var(--text); font-family: 'Manrope', sans-serif; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; }
        .col-12 { grid-column: span 12; } .col-6 { grid-column: span 6; } .col-4 { grid-column: span 4; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 20px; padding: 25px; }
        .metric-val { font-family: 'JetBrains Mono'; font-size: 2.2rem; font-weight: 700; }
        .status-badge { padding: 8px 20px; border-radius: 50px; font-weight: 700; border: 2px solid; }
        @media(max-width:768px){ .col-6, .col-4 { grid-column: span 12; } }
    </style>
</head>
<body>
    <div style="max-width:1600px; margin:0 auto;">
        <div class="card" style="display:flex; justify-content:space-between; margin-bottom:30px;">
            <div><h1>{{ site_config['label'] }}</h1><span class="status-badge" style="color:{{ st_col }}; border-color:{{ st_col }}">{{ st_txt }}</span></div>
            <div style="text-align:right"><div style="font-family:'JetBrains Mono'; font-size:1.4rem">{{ d['timestamp'] }}</div><a href="/logout" style="color:#ef4444">Logout</a></div>
        </div>

        <div class="grid">
            <!-- NAIROBI GRID STATS -->
            {% if site_id == 'nairobi' %}
            <div class="col-12 card">
                <h3>🔌 Utility Grid Energy (KPLC)</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom:20px;">
                    <div style="background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius:12px;">
                        <span style="color:var(--info); font-weight:700">IMPORT (To User)</span>
                        <div class="metric-val">{{ grid_stats['eToUserToday'] }} kWh</div>
                        <div style="opacity:0.7">Total: {{ grid_stats['eToUserTotal'] }} kWh</div>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.1); padding: 15px; border-radius:12px;">
                        <span style="color:var(--success); font-weight:700">EXPORT (To Grid)</span>
                        <div class="metric-val">{{ grid_stats['eToGridToday'] }} kWh</div>
                        <div style="opacity:0.7">Total: {{ grid_stats['eToGridTotal'] }} kWh</div>
                    </div>
                </div>
                <!-- HISTORICAL LOOKUP -->
                <div style="border-top:1px solid var(--border); padding-top:15px;">
                    <h4>📅 Historical Data Lookup</h4>
                    <div style="display:flex; gap:10px;">
                        <input type="date" id="histDate" style="padding:8px; border-radius:6px;">
                        <button onclick="fetchHistory()" style="background:var(--accent); color:white; border:none; padding:8px 15px; border-radius:6px; cursor:pointer;">Fetch Data</button>
                    </div>
                    <div id="histResult" style="margin-top:15px; display:none; grid-template-columns: 1fr 1fr; gap:10px;"></div>
                </div>
            </div>
            {% endif %}

            <div class="col-4 card"><h3>Solar Input</h3><div class="metric-val" style="color:var(--warn)">{{ '%0.f'|format(solar) }}W</div></div>
            <div class="col-4 card"><h3>Active Load</h3><div class="metric-val" style="color:var(--info)">{{ '%0.f'|format(load) }}W</div></div>
            <div class="col-4 card"><h3>Battery</h3><div class="metric-val" style="color:var(--success)">{{ breakdown['total_pct'] }}%</div></div>
            
            <div class="col-12 card"><h3>Storage Breakdown</h3><div style="height:250px"><canvas id="pieChart"></canvas></div></div>
        </div>
    </div>
    <script>
        function fetchHistory() {
            const date = document.getElementById('histDate').value;
            if(!date) return alert("Select a date");
            const resDiv = document.getElementById('histResult');
            resDiv.innerHTML = "Loading..."; resDiv.style.display = 'block';
            
            fetch('/api/history', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({site_id: '{{ site_id }}', date: date}) })
            .then(r => r.json()).then(d => {
                if(d.error) { resDiv.innerHTML = "Error: " + d.error; return; }
                resDiv.style.display = 'grid';
                resDiv.innerHTML = `
                    <div style="background:rgba(59,130,246,0.1); padding:10px; border-radius:8px;"><div>Import Today</div><b>\${d.eToUserToday} kWh</b></div>
                    <div style="background:rgba(59,130,246,0.1); padding:10px; border-radius:8px;"><div>Import Total</div><b>\${d.eToUserTotal} kWh</b></div>
                    <div style="background:rgba(16,185,129,0.1); padding:10px; border-radius:8px;"><div>Export Today</div><b>\${d.eToGridToday} kWh</b></div>
                    <div style="background:rgba(16,185,129,0.1); padding:10px; border-radius:8px;"><div>Export Total</div><b>\${d.eToGridTotal} kWh</b></div>`;
            }).catch(e => resDiv.innerHTML = "Fetch failed");
        }
        
        // Simplified Chart Initialization
        new Chart(document.getElementById('pieChart'), {
            type: 'doughnut',
            data: { labels: {{ breakdown['tier_labels']|tojson }}, datasets: [{ data: {{ breakdown['chart_data']|tojson }}, backgroundColor: {{ breakdown['tier_colors']|tojson }} }] },
            options: { responsive: true, maintainAspectRatio: false }
        });
        
        fetch('/health').then(r=>r.json()).then(d=>{ if(!d.polling_thread_alive) fetch('/start-polling'); });
        setTimeout(() => location.reload(), 120000);
    </script>
</body>
</html>
"""
    return render_template_string(html, d=d, solar=solar, load=load, st_txt=st_txt, st_col=st_col, breakdown=breakdown, site_config=site_config, site_id=site_id, grid_stats=grid_stats)

# Simplified Login Template
LOGIN_TEMPLATE = """<!DOCTYPE html><html><body style="background:#0a0e27; display:flex; justify-content:center; align-items:center; height:100vh; font-family:sans-serif;">
<form method="POST" style="background:white; padding:40px; border-radius:10px; text-align:center;">
<h2>Solar Monitor</h2><input type="password" name="password" placeholder="Password" style="padding:10px; width:100%; margin-bottom:10px;"><button type="submit" style="padding:10px; width:100%; background:#6366f1; color:white; border:none; cursor:pointer;">Login</button>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}</form></body></html>"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
