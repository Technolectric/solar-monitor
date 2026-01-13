import os
import time
import requests
import json
from datetime import datetime, timedelta, timezone
from threading import Thread
from flask import Flask, render_template_string, request, jsonify
import numpy as np
from collections import deque
from pathlib import Path

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Config & Environment
# ----------------------------
API_URL = "https://openapi.growatt.com/v1/device/storage/storage_last_data"
TOKEN = os.getenv("API_TOKEN")
SERIAL_NUMBERS = os.getenv("SERIAL_NUMBERS", "").split(",")
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", 5))
DATA_FILE = "load_patterns.json"

print(f"🔧 Configuration: TOKEN={'SET' if TOKEN else 'NOT SET'}, SERIALS={len(SERIAL_NUMBERS)}")

# ----------------------------
# Inverter Configuration
# ----------------------------
INVERTER_CONFIG = {
    "RKG3B0400T": {"label": "Inverter 1", "type": "primary", "datalog": "DDD0B021CC", "display_order": 1},
    "KAM4N5W0AG": {"label": "Inverter 2", "type": "primary", "datalog": "DDD0B02121", "display_order": 2},
    "JNK1CDR0KQ": {"label": "Inverter 3 (Backup)", "type": "backup", "datalog": "DDD0B0221H", "display_order": 3}
}

# Specs
PRIMARY_BATTERY_THRESHOLD = 40
BACKUP_VOLTAGE_THRESHOLD = 51.2
TOTAL_SOLAR_CAPACITY_KW = 10
PRIMARY_BATTERY_CAPACITY_WH = 30000 
BACKUP_BATTERY_DEGRADED_WH = 21000   
BACKUP_DEGRADATION = 0.70
SOLAR_EFFICIENCY_FACTOR = 0.85
FORECAST_HOURS = 12

# Location & Email
LATITUDE = -1.85238
LONGITUDE = 36.77683
RESEND_API_KEY = os.getenv('RESEND_API_KEY')
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')
EAT = timezone(timedelta(hours=3))

# ----------------------------
# 1. Persistence Manager
# ----------------------------
class PersistentLoadManager:
    """Saves load patterns to disk so predictions improve over time."""
    def __init__(self, filename):
        self.filename = filename
        self.patterns = self.load_data()
        
    def load_data(self):
        if Path(self.filename).exists():
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except: pass
        return {"weekday": {str(h): [] for h in range(24)}, "weekend": {str(h): [] for h in range(24)}}

    def save_data(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.patterns, f)
        except: pass

    def update(self, load_watts):
        now = datetime.now(EAT)
        hour = str(now.hour)
        day_type = "weekend" if now.weekday() >= 5 else "weekday"
        
        self.patterns[day_type][hour].append(load_watts)
        if len(self.patterns[day_type][hour]) > 100:
            self.patterns[day_type][hour] = self.patterns[day_type][hour][-100:]
            
    def get_forecast(self, hours_ahead=12):
        forecast = []
        now = datetime.now(EAT)
        for i in range(hours_ahead):
            ft = now + timedelta(hours=i)
            f_hour = str(ft.hour)
            f_day = "weekend" if ft.weekday() >= 5 else "weekday"
            
            history = self.patterns[f_day][f_hour]
            if history:
                est = sum(history) / len(history)
            else:
                if 18 <= int(f_hour) <= 21: est = 2500
                elif 0 <= int(f_hour) <= 5: est = 600
                else: est = 1200
            forecast.append({'time': ft, 'hour': ft.hour, 'estimated_load': est})
        return forecast

load_manager = PersistentLoadManager(DATA_FILE)

# ----------------------------
# 2. Smart Appliance Detection
# ----------------------------
def identify_active_appliances(current, previous, gen_active, backup_volts, primary_pct):
    detected = []
    delta = current - previous

    # --- Generator Logic ---
    if gen_active:
        # High Primary (>42%) implies Manual Water Heating
        if primary_pct > 42:
            detected.append("🚿 Water Heating (Manual Start)")
            return detected
        else:
            detected.append("⚡ Generator (System Charging)")
            return detected

    # --- Inverter Load Logic ---
    if current < 400: detected.append("🌙 Idle (Fridges/WiFi)")
    elif 1000 <= current <= 1350: detected.append("🏊 Pool Pump")
    elif current > 1800: detected.append("🍳 Cooking (Oven/Stove)")
    elif 400 <= current < 1000: detected.append("📺 TV / Lighting")

    if delta > 1500: detected.append("☕ Kettle/Microwave (Started)")
        
    return detected

# ----------------------------
# Globals & State
# ----------------------------
headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"} if TOKEN else {}
last_alert_time, alert_history, last_communication = {}, [], {}

latest_data = {
    "timestamp": "Initializing...", "total_output_power": 0, "total_battery_discharge_W": 0,
    "total_solar_input_W": 0, "primary_battery_min": 0, "backup_battery_voltage": 0,
    "backup_voltage_status": "Unknown", "backup_active": False, "backup_percent_calc": 0,
    "generator_running": False, "inverters": [], "detected_appliances": [], 
    "solar_forecast": [], "load_forecast": [],
    "battery_life_prediction": None, "weather_source": "Initializing...",
    "usable_energy": {"primary_kwh": 0, "backup_kwh": 0, "total_kwh": 0, "total_pct": 0}
}
load_history, battery_history = [], []
weather_forecast = {}
solar_conditions_cache = None
solar_generation_pattern = deque(maxlen=5000)

# ----------------------------
# Helper Functions
# ----------------------------
def calculate_usable_energy(primary_pct, backup_pct):
    primary_available_wh = max(0, ((primary_pct - 40) / 100) * PRIMARY_BATTERY_CAPACITY_WH)
    primary_available_kwh = primary_available_wh / 1000
    backup_actual_capacity_wh = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION
    backup_available_wh = max(0, ((backup_pct - 20) / 100) * backup_actual_capacity_wh)
    backup_available_kwh = backup_available_wh / 1000
    total_available_kwh = primary_available_kwh + backup_available_kwh
    total_usable_capacity_wh = (PRIMARY_BATTERY_CAPACITY_WH * 0.6) + (backup_actual_capacity_wh * 0.8)
    total_usable_capacity_kwh = total_usable_capacity_wh / 1000
    total_available_pct = (total_available_kwh / total_usable_capacity_kwh) * 100 if total_usable_capacity_kwh > 0 else 0
    return {'primary_kwh': round(primary_available_kwh, 1), 'backup_kwh': round(backup_available_kwh, 1), 'total_kwh': round(total_available_kwh, 1), 'total_pct': round(total_available_pct, 1), 'total_usable_capacity': round(total_usable_capacity_kwh, 1)}

# --- All 4 Weather Sources ---
def get_weather_from_openmeteo():
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=cloud_cover,shortwave_radiation&timezone=Africa/Nairobi&forecast_days=2"
        response = requests.get(url, timeout=10)
        return {'times': response.json()['hourly']['time'], 'cloud_cover': response.json()['hourly']['cloud_cover'], 'solar_radiation': response.json()['hourly']['shortwave_radiation'], 'source': 'Open-Meteo'}
    except: return None

def get_weather_from_weatherapi():
    try:
        WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY") 
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={LATITUDE},{LONGITUDE}&days=2"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            times, cloud, solar = [], [], []
            for day in data.get('forecast', {}).get('forecastday', []):
                for hour in day.get('hour', []):
                    times.append(hour['time'])
                    cloud.append(hour['cloud'])
                    solar.append(hour.get('uv', 0) * 120) 
            if times: return {'times': times, 'cloud_cover': cloud, 'solar_radiation': solar, 'source': 'WeatherAPI'}
    except: pass
    return None
        
def get_weather_from_7timer():
    try:
        url = f"http://www.7timer.info/bin/api.pl?lon={LONGITUDE}&lat={LATITUDE}&product=civil&output=json"
        response = requests.get(url, timeout=15)
        data = response.json()
        times, cloud, solar = [], [], []
        base = datetime.now(EAT)
        for item in data.get('dataseries', [])[:48]:
            t = base + timedelta(hours=item.get('timepoint', 0))
            times.append(t.strftime('%Y-%m-dT%H:%M'))
            c_pct = min((item.get('cloudcover', 5) * 12), 100)
            cloud.append(c_pct)
            solar.append(max(800 * (1 - c_pct/100), 0))
        if times: return {'times': times, 'cloud_cover': cloud, 'solar_radiation': solar, 'source': '7Timer'}
    except: pass
    return None

def get_fallback_weather():
    times, clouds, rads = [], [], []
    now = datetime.now(EAT).replace(minute=0, second=0, microsecond=0)
    for i in range(48):
        t = now + timedelta(hours=i)
        times.append(t.isoformat())
        clouds.append(20)
        h = t.hour
        rads.append(max(0, 1000 - (abs(12 - h) * 150)) if 6 <= h <= 18 else 0)
    return {'times': times, 'cloud_cover': clouds, 'solar_radiation': rads, 'source': 'Synthetic (Offline)'}

def get_weather_forecast():
    for src, func in [("Open-Meteo", get_weather_from_openmeteo), ("WeatherAPI", get_weather_from_weatherapi), ("7Timer", get_weather_from_7timer)]:
        f = func()
        if f and len(f.get('times', [])) > 0:
            return f
    return get_fallback_weather()

def analyze_solar_conditions(forecast):
    if not forecast: return None
    try:
        now = datetime.now(EAT)
        h = now.hour
        is_night = h < 6 or h >= 18
        if is_night:
            start = (now + timedelta(days=1)).replace(hour=6, minute=0)
            end = (now + timedelta(days=1)).replace(hour=18, minute=0)
            label = "Tomorrow's Daylight"
        else:
            start = now
            end = now.replace(hour=18, minute=0)
            label = "Today's Remaining Daylight"
        
        c_sum, s_sum, count = 0, 0, 0
        for i, t_str in enumerate(forecast['times']):
            try:
                ft = datetime.fromisoformat(t_str.replace('Z', '')) if 'T' in t_str else datetime.strptime(t_str, '%Y-%m-%d %H:%M')
                ft = ft.replace(tzinfo=EAT) if ft.tzinfo is None else ft.astimezone(EAT)
                if start <= ft <= end and 6 <= ft.hour <= 18:
                    c_sum += forecast['cloud_cover'][i]
                    s_sum += forecast['solar_radiation'][i]
                    count += 1
            except: continue
        
        if count > 0:
            return {
                'avg_cloud_cover': c_sum/count,
                'avg_solar_radiation': s_sum/count,
                'poor_conditions': (c_sum/count) > 70 or (s_sum/count) < 200,
                'analysis_period': label
            }
    except: pass
    return None

def analyze_historical_solar_pattern():
    if len(solar_generation_pattern) < 3: return None
    pattern, hour_map = [], {}
    for d in solar_generation_pattern:
        h = d['hour']
        if h not in hour_map: hour_map[h] = []
        hour_map[h].append(d['generation'] / d.get('max_possible', TOTAL_SOLAR_CAPACITY_KW * 1000))
    for h, v in hour_map.items(): pattern.append((h, np.mean(v)))
    return pattern

def get_hourly_weather_forecast(weather_data, num_hours=12):
    hourly = []
    now = datetime.now(EAT)
    if not weather_data: return hourly
    w_times = []
    for i, t_str in enumerate(weather_data['times']):
        try:
            ft = datetime.fromisoformat(t_str.replace('Z', '')) if 'T' in t_str else datetime.strptime(t_str, '%Y-%m-%d %H:%M')
            ft = ft.replace(tzinfo=EAT) if ft.tzinfo is None else ft.astimezone(EAT)
            w_times.append({'time': ft, 'cloud': weather_data['cloud_cover'][i], 'solar': weather_data['solar_radiation'][i]})
        except: continue
    w_times.sort(key=lambda x: x['time'])
    for i in range(num_hours):
        ft = now + timedelta(hours=i)
        closest = min(w_times, key=lambda x: abs(x['time'] - ft))
        hourly.append({'time': ft, 'hour': ft.hour, 'cloud_cover': closest['cloud'], 'solar_radiation': closest['solar']})
    return hourly

def apply_solar_curve(gen, hour):
    if hour < 6 or hour >= 19: return 0.0
    curve = np.sin(((hour - 6) / 13.0) * np.pi) ** 2
    return gen * curve * (0.7 if hour <= 7 or hour >= 18 else 1.0)

def generate_solar_forecast(weather_data, pattern):
    forecast = []
    hourly = get_hourly_weather_forecast(weather_data, FORECAST_HOURS)
    max_gen = TOTAL_SOLAR_CAPACITY_KW * 1000
    for d in hourly:
        h = d['hour']
        if h < 6 or h >= 19:
            est = 0.0
        else:
            theo = (d['solar_radiation'] / 1000) * max_gen * SOLAR_EFFICIENCY_FACTOR
            curved = apply_solar_curve(theo, h)
            if pattern:
                p_val = next((v for ph, v in pattern if ph == h), 0)
                est = (curved * 0.6 + (p_val * max_gen) * 0.4)
            else: est = curved
        forecast.append({'time': d['time'], 'hour': h, 'estimated_generation': max(0, est)})
    return forecast

# --- Battery Simulation Logic ---
def calculate_battery_cascade(solar, load, p_pct, b_active=False):
    if not solar or not load: return None
    
    p_daily_wh = max(0, ((p_pct/100)*30000) - 12000)
    b_wh = max(0, (21000 * 0.9) - 4200)
    
    trace = [((p_daily_wh + b_wh) / 34800) * 100]
    gen_needed, empty_time, switch_occurred = False, None, False
    acc_gen_wh = 0
    
    for i in range(min(len(solar), len(load))):
        net = load[i]['estimated_load'] - solar[i]['estimated_generation']
        step = net * 1.0
        
        if step > 0:
            if p_daily_wh >= step: p_daily_wh -= step
            else:
                rem = step - p_daily_wh
                p_daily_wh = 0
                switch_occurred = True
                if b_wh >= rem: b_wh -= rem
                else:
                    b_wh = 0
                    gen_needed = True
                    acc_gen_wh += (rem - b_wh)
                    if not empty_time: empty_time = solar[i]['time'].strftime("%I:%M %p")
        else:
            surplus = abs(step)
            space_p = 18000 - p_daily_wh
            if surplus <= space_p: p_daily_wh += surplus
            else:
                p_daily_wh = 18000
                surplus -= space_p
                if surplus <= (16800 - b_wh): b_wh += surplus
                else: b_wh = 16800
        
        trace.append(((p_daily_wh + b_wh) / 34800) * 100)
    
    return {'trace_total_pct': trace, 'generator_needed': gen_needed, 'time_empty': empty_time, 'switchover_occurred': switch_occurred, 'genset_hours': acc_gen_wh/5000}

def send_email(subject, html, alert_type="general", send_via_email=True):
    global last_alert_time, alert_history
    cooldown = 120
    if "critical" in alert_type: cooldown = 60
    
    if alert_type in last_alert_time and (datetime.now(EAT) - last_alert_time[alert_type]) < timedelta(minutes=cooldown):
        return False
    
    if send_via_email and RESEND_API_KEY:
        try:
            requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_API_KEY}"}, json={"from": SENDER_EMAIL, "to": [RECIPIENT_EMAIL], "subject": subject, "html": html})
        except: pass
        
    now = datetime.now(EAT)
    last_alert_time[alert_type] = now
    alert_history.append({"timestamp": now, "type": alert_type, "subject": subject})
    # Keep last 20
    alert_history[:] = [a for a in alert_history if a['timestamp'] >= (now - timedelta(days=1))]
    return True

# ----------------------------
# Polling Loop
# ----------------------------
polling_active = False
polling_thread = None

def poll_growatt():
    global latest_data, load_history, battery_history, weather_forecast, solar_conditions_cache
    global polling_active, alert_history
    
    if not TOKEN: return
    
    weather_forecast = get_weather_forecast()
    if weather_forecast: solar_conditions_cache = analyze_solar_conditions(weather_forecast)
    
    last_wx = datetime.now(EAT)
    last_save_time = datetime.now(EAT)
    previous_load_watts = 0
    polling_active = True
    
    print("🚀 Polling Started: Full Features + Alert History + Persistence")
    
    while polling_active:
        try:
            now = datetime.now(EAT)
            
            # Weather Refresh
            if (now - last_wx) > timedelta(minutes=30):
                weather_forecast = get_weather_forecast()
                if weather_forecast: solar_conditions_cache = analyze_solar_conditions(weather_forecast)
                last_wx = now
                
            tot_out, tot_bat, tot_sol = 0, 0, 0
            inv_data, p_caps = [], []
            b_data, gen_on = None, False
            
            # API Calls
            for sn in SERIAL_NUMBERS:
                try:
                    r = requests.post(API_URL, data={"storage_sn": sn}, headers=headers, timeout=20)
                    if r.json().get("error_code") == 0:
                        d = r.json().get("data", {})
                        
                        op = float(d.get("outPutPower") or 0)
                        cap = float(d.get("capacity") or 0)
                        vb = float(d.get("vBat") or 0)
                        pb = float(d.get("pBat") or 0)
                        sol = float(d.get("ppv") or 0) + float(d.get("ppv2") or 0)
                        temp = float(d.get("temperature") or 0)
                        flt = int(d.get("errorCode") or 0) != 0
                        
                        tot_out += op
                        tot_sol += sol
                        if pb > 0: tot_bat += pb
                        
                        cfg = INVERTER_CONFIG.get(sn, {"label": sn, "type": "unknown", "display_order": 99})
                        info = {
                            "SN": sn, "Label": cfg['label'], "OutputPower": op, "Capacity": cap, "vBat": vb, "temperature": temp, 
                            "has_fault": flt, "communication_lost": False
                        }
                        inv_data.append(info)
                        
                        if cfg['type'] == 'primary': p_caps.append(cap)
                        elif cfg['type'] == 'backup':
                            b_data = info
                            if float(d.get("vac") or 0) > 100 or float(d.get("pAcInPut") or 0) > 50: gen_on = True
                        
                        if flt: send_email(f"FAULT: {cfg['label']}", "Inverter reports fault", "fault")
                        if temp > 60: send_email(f"High Temp: {cfg['label']}", f"{temp}C", "temp")
                        
                except Exception as e:
                    print(f"Error sn {sn}: {e}")
            
            # Data Processing
            p_min = min(p_caps) if p_caps else 0
            b_volts = b_data['vBat'] if b_data else 0
            b_act = b_data['OutputPower'] > 50 if b_data else False
            
            # --- Alert Logic ---
            if gen_on: send_email("Generator Running", "Generator detected", "gen")
            if b_act: send_email("Backup Active", "Primary depleted", "backup")
            
            # --- Smart Logic ---
            detected_apps = identify_active_appliances(tot_out, previous_load_watts, gen_on, b_volts, p_min)
            
            is_manual_gen_run = any("Manual" in app or "Water" in app for app in detected_apps)
            if not is_manual_gen_run:
                load_manager.update(tot_out)
            
            if (now - last_save_time) > timedelta(hours=1):
                load_manager.save_data()
                last_save_time = now
            
            # Forecasting
            l_cast = load_manager.get_forecast(12)
            
            # Solar Pattern
            now_h = now.hour
            solar_generation_pattern.append({'hour': now_h, 'generation': tot_sol, 'max_possible': 10000})
            s_pat = analyze_historical_solar_pattern()
            s_cast = generate_solar_forecast(weather_forecast, s_pat)
            
            # Simulation
            pred = calculate_battery_cascade(s_cast, l_cast, p_min, b_act)
            usable = calculate_usable_energy(p_min, max(0, min(100, (b_volts - 51.0) / 2.0 * 100)))
            
            load_history.append((now, tot_out))
            load_history[:] = [(t, p) for t, p in load_history if t >= (now - timedelta(days=14))]
            battery_history.append((now, tot_bat))
            
            previous_load_watts = tot_out
            
            latest_data = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S EAT"),
                "total_output_power": tot_out,
                "total_battery_discharge_W": tot_bat,
                "total_solar_input_W": tot_sol,
                "primary_battery_min": p_min,
                "backup_battery_voltage": b_volts,
                "backup_active": b_act,
                "generator_running": gen_on,
                "inverters": inv_data,
                "detected_appliances": detected_apps,
                "solar_forecast": s_cast,
                "load_forecast": l_cast,
                "battery_life_prediction": pred,
                "usable_energy": usable
            }
            
            print(f"Update: {tot_out}W | Gen: {gen_on} | Alerts: {len(alert_history)}")
            
        except Exception as e: print(f"Poll Error: {e}")
        
        if polling_active:
            for _ in range(POLL_INTERVAL_MINUTES * 60):
                if not polling_active: break
                time.sleep(1)

# ----------------------------
# Routes
# ----------------------------
@app.route('/health')
def health(): return jsonify({"status": "healthy"})

@app.route('/start-polling')
def start_polling():
    global polling_active, polling_thread
    if not polling_active:
        polling_thread = Thread(target=poll_growatt, daemon=True)
        polling_thread.start()
    return jsonify({"status": "started"})

@app.route('/api/data')
def api_data(): return jsonify(latest_data)

@app.route("/")
def home():
    def _num(val): return float(val) if val is not None else 0
    d = latest_data
    
    # Data extraction
    p_bat = _num(d.get("primary_battery_min", 0))
    b_volt = _num(d.get("backup_battery_voltage", 0))
    b_act = d.get("backup_active", False)
    gen_on = d.get("generator_running", False)
    tot_load = _num(d.get("total_output_power", 0))
    tot_sol = _num(d.get("total_solar_input_W", 0))
    detected = d.get("detected_appliances", [])
    usable = d.get("usable_energy", {"total_pct": 0})
    
    # Status Logic
    app_st, app_sub, app_col, status_icon = "ℹ️ NORMAL", "System running optimally", "normal", "ℹ️"
    if gen_on:
        if any("Water" in x for x in detected):
            app_st, app_sub, app_col, status_icon = "🚿 WATER HEATING", "Manual Generator Run detected", "warning", "🚿"
        elif any("System Charging" in x for x in detected):
            app_st, app_sub, app_col, status_icon = "⚡ AUTO-CHARGE", "Low Battery Protection Active", "critical", "⚡"
        else:
            app_st, app_sub, app_col, status_icon = "⚠️ GENERATOR ON", "Generator running", "critical", "⚙️"
    elif b_act:
        app_st, app_sub, app_col, status_icon = "⚠️ BACKUP ACTIVE", "Primary depleted", "critical", "🔋"
    elif any("Pool" in x for x in detected):
        app_st, app_sub, app_col, status_icon = "🏊 POOL CYCLE", "Pool pump running", "normal", "🏊"
    elif usable['total_pct'] > 95:
        app_st, app_sub, app_col, status_icon = "✅ BATTERY FULL", "Fully Charged", "good", "🔋"

    # Charts Data
    l_fc = d.get("load_forecast", [])
    s_fc = d.get("solar_forecast", [])
    pred = d.get("battery_life_prediction")
    
    chart_labels = [x['time'].strftime('%H:%M') for x in l_fc] if l_fc else []
    chart_load = [x['estimated_load'] for x in l_fc] if l_fc else []
    chart_solar = [x['estimated_generation'] for x in s_fc[:len(l_fc)]] if s_fc else []
    
    sim_t = ["Now"] + [d['time'].strftime('%H:%M') for d in s_fc] if s_fc else []
    trace_pct = pred.get('trace_total_pct', []) if pred else []
    
    # Get recent alerts (reverse order)
    alerts = sorted(alert_history, key=lambda x: x['timestamp'], reverse=True)[:10]

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tulia House Solar</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root { --bg: #0a0e13; --surface: #151922; --surface-2: #1d232e; --border: rgba(58, 70, 89, 0.5); --text: #e6edf5; --primary: #3fb950; --warning: #f0883e; --danger: #f85149; --info: #58a6ff; --radius: 16px; }
        body { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; margin: 0; padding: 1rem; }
        .container { max-width: 1600px; margin: 0 auto; display: grid; gap: 1.5rem; grid-template-columns: repeat(12, 1fr); }
        .span-12 { grid-column: span 12; } .span-6 { grid-column: span 12; } .span-4 { grid-column: span 12; } .span-3 { grid-column: span 6; }
        @media(min-width:768px){ .span-6 { grid-column: span 6; } .span-4 { grid-column: span 4; } .span-3 { grid-column: span 3; } }
        
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; }
        .hero { text-align: center; padding: 2rem; border-radius: var(--radius); background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%); border: 1px solid var(--border); }
        .hero.critical { border-color: var(--danger); background: linear-gradient(135deg, rgba(248,81,73,0.15), rgba(21,25,34,0.95)); }
        .hero.warning { border-color: var(--warning); background: linear-gradient(135deg, rgba(240,136,62,0.15), rgba(21,25,34,0.95)); }
        .hero.good { border-color: var(--primary); background: linear-gradient(135deg, rgba(63,185,80,0.15), rgba(21,25,34,0.95)); }
        
        .metric { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 600; }
        
        .app-grid { display: flex; flex-wrap: wrap; gap: 0.8rem; }
        .tag { padding: 0.6rem 1.2rem; border-radius: 50px; font-size: 0.9rem; font-weight: 600; background: rgba(255,255,255,0.05); border: 1px solid var(--border); display: flex; align-items: center; gap: 0.5rem; }
        .tag.water { border-color: var(--warning); color: var(--warning); }
        .tag.pool { border-color: var(--info); color: var(--info); }
        .tag.cook { border-color: var(--danger); color: var(--danger); }
        
        .flow-container { height: 300px; display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; justify-items: center; position: relative; }
        .node { width: 80px; height: 80px; background: var(--surface-2); border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px solid var(--border); z-index: 2; }
        .node.active { animation: pulse 2s infinite; border-color: var(--primary); }
        @keyframes pulse { 0%{box-shadow: 0 0 0 0 rgba(63,185,80,0.4)} 70%{box-shadow: 0 0 0 10px rgba(63,185,80,0)} 100%{box-shadow: 0 0 0 0 rgba(63,185,80,0)} }
        
        .alert-row { display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .alert-time { font-family: 'Space Mono'; opacity: 0.7; }
        .alert-type.gen { color: var(--danger); }
        .alert-type.backup { color: var(--warning); }
    </style>
</head>
<body>
    <div class="container">
        <!-- Status Hero -->
        <div class="span-12 hero {{ app_col }}">
            <div style="font-size: 3rem">{{ status_icon }}</div>
            <h1 style="margin:0.5rem 0">{{ app_st }}</h1>
            <div style="opacity:0.8">{{ app_sub }}</div>
        </div>

        <!-- NEW: Detected Activity -->
        <div class="span-12 card">
            <h2>🔎 Detected Activity</h2>
            <div class="app-grid">
                {% if detected %}
                    {% for app in detected %}
                        <div class="tag {{ 'water' if 'Water' in app else '' }} {{ 'pool' if 'Pool' in app else '' }} {{ 'cook' if 'Cooking' in app else '' }}">
                            {{ app }}
                        </div>
                    {% endfor %}
                {% else %}
                    <div style="opacity: 0.5">System nominal. No heavy loads.</div>
                {% endif %}
            </div>
        </div>

        <!-- Metrics -->
        <div class="span-3 card"><h3>Load</h3><div class="metric">{{ '%0.f'|format(tot_load) }}W</div></div>
        <div class="span-3 card"><h3>Solar</h3><div class="metric text-success">{{ '%0.f'|format(tot_sol) }}W</div></div>
        <div class="span-3 card"><h3>Primary</h3><div class="metric">{{ '%0.f'|format(p_bat) }}%</div></div>
        <div class="span-3 card"><h3>Backup</h3><div class="metric">{{ '%0.1f'|format(b_volt) }}V</div></div>

        <!-- Power Flow -->
        <div class="span-12 card">
            <h2>⚡ Power Flow</h2>
            <div class="flow-container">
                <div class="node" style="grid-column:1; grid-row:2">☀️<br>{{ '%0.f'|format(tot_sol) }}W</div>
                <div class="node {{ 'active' if gen_on else '' }}" style="grid-column:2; grid-row:1">⚙️<br>GEN</div>
                <div class="node" style="grid-column:2; grid-row:2; width:100px; height:100px; border-color:var(--info)">⚡<br>INV</div>
                <div class="node" style="grid-column:2; grid-row:3">🔋<br>{{ usable.total_pct }}%</div>
                <div class="node {{ 'active' if tot_load > 2000 else '' }}" style="grid-column:3; grid-row:2">🏠<br>{{ '%0.f'|format(tot_load) }}W</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="span-6 card">
            <h2>🔮 12-Hour Forecast</h2>
            <div style="height:300px"><canvas id="fcChart"></canvas></div>
        </div>
        <div class="span-6 card">
            <h2>🔋 Capacity Simulation</h2>
            <div style="height:300px"><canvas id="simChart"></canvas></div>
        </div>
        
        <!-- RESTORED: Alert History -->
        <div class="span-12 card">
            <h2>🔔 Recent Alerts</h2>
            {% if alerts %}
                {% for alert in alerts %}
                    <div class="alert-row">
                        <div class="alert-type {{ alert.type }}"><strong>{{ alert.subject }}</strong></div>
                        <div class="alert-time">{{ alert.timestamp.strftime('%H:%M') }}</div>
                    </div>
                {% endfor %}
            {% else %}
                <div style="opacity:0.5; padding:10px">No active alerts.</div>
            {% endif %}
        </div>
    </div>
    
    <script>
        new Chart(document.getElementById('fcChart'), {
            type: 'line',
            data: {
                labels: {{ chart_labels|tojson }},
                datasets: [
                    { label: 'Predicted Load', data: {{ chart_load|tojson }}, borderColor: '#58a6ff', borderDash: [5,5], tension: 0.4 },
                    { label: 'Solar Forecast', data: {{ chart_solar|tojson }}, borderColor: '#3fb950', fill: true, backgroundColor: 'rgba(63,185,80,0.1)', tension: 0.4 }
                ]
            }, options: { maintainAspectRatio: false }
        });
        
        new Chart(document.getElementById('simChart'), {
            type: 'line',
            data: {
                labels: {{ sim_t|tojson }},
                datasets: [{ label: 'Battery %', data: {{ trace_pct|tojson }}, borderColor: '#f0883e', fill: true, tension: 0.4 }]
            }, options: { maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
        });
        
        fetch('/health').then(r=>r.json()).then(d=>{ if(!d.polling_thread_alive) fetch('/start-polling'); });
        setTimeout(()=>location.reload(), 60000);
    </script>
</body>
</html>
    """
    return render_template_string(html, 
        app_st=app_st, app_sub=app_sub, app_col=app_col, status_icon=status_icon,
        detected=detected, tot_load=tot_load, tot_sol=tot_sol,
        p_bat=p_bat, b_volt=b_volt, usable=usable,
        chart_labels=chart_labels, chart_load=chart_load, chart_solar=chart_solar,
        sim_t=sim_t, trace_pct=trace_pct, gen_on=gen_on, alerts=alerts
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
