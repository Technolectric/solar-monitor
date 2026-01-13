import os
import time
import requests
import json
from datetime import datetime, timedelta, timezone
from threading import Thread
from flask import Flask, render_template_string, request, jsonify
import numpy as np
from collections import deque

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Growatt Config (from env)
# ----------------------------
API_URL = "https://openapi.growatt.com/v1/device/storage/storage_last_data"
TOKEN = os.getenv("API_TOKEN")
SERIAL_NUMBERS = os.getenv("SERIAL_NUMBERS", "").split(",")
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", 5))

print(f"🔧 Configuration: TOKEN={'SET' if TOKEN else 'NOT SET'}, SERIALS={len(SERIAL_NUMBERS)}")

# ----------------------------
# Inverter Configuration
# ----------------------------
INVERTER_CONFIG = {
    "RKG3B0400T": {"label": "Inverter 1", "type": "primary", "datalog": "DDD0B021CC", "display_order": 1},
    "KAM4N5W0AG": {"label": "Inverter 2", "type": "primary", "datalog": "DDD0B02121", "display_order": 2},
    "JNK1CDR0KQ": {"label": "Inverter 3 (Backup)", "type": "backup", "datalog": "DDD0B0221H", "display_order": 3}
}

# Thresholds & Battery Specs
PRIMARY_BATTERY_THRESHOLD = 40
BACKUP_VOLTAGE_THRESHOLD = 51.2
TOTAL_SOLAR_CAPACITY_KW = 10
PRIMARY_INVERTER_CAPACITY_W = 10000
BACKUP_INVERTER_CAPACITY_W = 5000

BACKUP_VOLTAGE_GOOD = 53.0
BACKUP_VOLTAGE_MEDIUM = 52.3
BACKUP_VOLTAGE_LOW = 52.0

INVERTER_TEMP_WARNING = 60
INVERTER_TEMP_CRITICAL = 70
COMMUNICATION_TIMEOUT_MINUTES = 10

# Battery Specs (LiFePO4)
PRIMARY_BATTERY_CAPACITY_WH = 30000 
PRIMARY_DAILY_MIN_PCT = 40 
BACKUP_BATTERY_DEGRADED_WH = 21000   
BACKUP_DEGRADATION = 0.70  # 70% State of Health
BACKUP_CUTOFF_PCT = 20
TOTAL_SYSTEM_USABLE_WH = 34800 

# ----------------------------
# Location & Email
# ----------------------------
LATITUDE = -1.85238
LONGITUDE = 36.77683
RESEND_API_KEY = os.getenv('RESEND_API_KEY')
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')

# ----------------------------
# Globals
# ----------------------------
headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"} if TOKEN else {}
last_alert_time = {}
latest_data = {
    "timestamp": "Initializing...",
    "total_output_power": 0,
    "total_battery_discharge_W": 0,
    "total_solar_input_W": 0,
    "primary_battery_min": 0,
    "backup_battery_voltage": 0,
    "backup_voltage_status": "Unknown",
    "backup_active": False,
    "backup_percent_calc": 0,
    "generator_running": False,
    "inverters": [],
    "solar_forecast": [],
    "load_forecast": [],
    "battery_life_prediction": None,
    "weather_source": "Initializing...",
    "usable_energy": {
        "primary_kwh": 0,
        "backup_kwh": 0,
        "total_kwh": 0,
        "total_pct": 0,
        "total_usable_capacity": 29.76
    }
}
load_history = []
battery_history = []
weather_forecast = {}
weather_source = "Initializing..."
solar_conditions_cache = None
alert_history = []
last_communication = {}

pool_pump_start_time = None
pool_pump_last_alert = None

solar_forecast = []
solar_generation_pattern = deque(maxlen=5000)
load_demand_pattern = deque(maxlen=5000)
SOLAR_EFFICIENCY_FACTOR = 0.85
FORECAST_HOURS = 12
EAT = timezone(timedelta(hours=3))

# ----------------------------
# Health Check Endpoint (CRITICAL FOR RAILWAY)
# ----------------------------
@app.route("/health")
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(EAT).strftime("%Y-%m-%d %H:%M:%S EAT"),
        "service": "Tulia House Solar Monitor",
        "polling_active": 'poll_thread' in globals() and poll_thread.is_alive(),
        "data_age": "initializing" if latest_data.get('timestamp') == 'Initializing...' else "active"
    }), 200

# ----------------------------
# Battery Calculation Function
# ----------------------------
def calculate_usable_energy(primary_pct, backup_pct):
    """Calculate actual usable energy considering cutoffs and degradation"""
    
    # Primary battery - usable down to 40% (switchover point)
    primary_available_wh = max(0, ((primary_pct - 40) / 100) * PRIMARY_BATTERY_CAPACITY_WH)
    primary_available_kwh = primary_available_wh / 1000
    
    # Backup battery - degraded to 70% SoH, usable down to 20%
    backup_actual_capacity_wh = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION  # 14700 Wh
    backup_available_wh = max(0, ((backup_pct - 20) / 100) * backup_actual_capacity_wh)
    backup_available_kwh = backup_available_wh / 1000
    
    # Total available
    total_available_kwh = primary_available_kwh + backup_available_kwh
    
    # Total usable capacity (60% of primary + 80% of degraded backup)
    total_usable_capacity_wh = (PRIMARY_BATTERY_CAPACITY_WH * 0.6) + (backup_actual_capacity_wh * 0.8)
    total_usable_capacity_kwh = total_usable_capacity_wh / 1000  # 29.76 kWh
    
    # Percentage of usable capacity
    total_available_pct = (total_available_kwh / total_usable_capacity_kwh) * 100 if total_usable_capacity_kwh > 0 else 0
    
    return {
        'primary_kwh': round(primary_available_kwh, 1),
        'backup_kwh': round(backup_available_kwh, 1),
        'total_kwh': round(total_available_kwh, 1),
        'total_pct': round(total_available_pct, 1),
        'total_usable_capacity': round(total_usable_capacity_kwh, 1)
    }

# ----------------------------
# Weather Functions
# ----------------------------
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
    global weather_source
    print("🌤️ Fetching weather forecast...")
    for src, func in [("Open-Meteo", get_weather_from_openmeteo), ("WeatherAPI", get_weather_from_weatherapi), ("7Timer", get_weather_from_7timer)]:
        f = func()
        if f and len(f.get('times', [])) > 0:
            weather_source = f['source']
            return f
    weather_source = "Synthetic (Offline)"
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
                'analysis_period': label,
                'is_nighttime': is_night
            }
    except: pass
    return None

# Helper Functions
def get_backup_voltage_status(voltage):
    if voltage >= BACKUP_VOLTAGE_GOOD: return "Good", "green"
    elif voltage >= BACKUP_VOLTAGE_MEDIUM: return "Medium", "orange"
    else: return "Low", "red"

def check_generator_running(backup_data):
    if not backup_data: return False
    return float(backup_data.get('vac', 0) or 0) > 100 or float(backup_data.get('pAcInPut', 0) or 0) > 50

def analyze_historical_solar_pattern():
    if len(solar_generation_pattern) < 3: return None
    pattern, hour_map = [], {}
    for d in solar_generation_pattern:
        h = d['hour']
        if h not in hour_map: hour_map[h] = []
        hour_map[h].append(d['generation'] / d.get('max_possible', TOTAL_SOLAR_CAPACITY_KW * 1000))
    for h, v in hour_map.items(): pattern.append((h, np.mean(v)))
    return pattern

def analyze_historical_load_pattern():
    if len(load_demand_pattern) < 3: return None
    pattern, hour_map = [], {}
    for d in load_demand_pattern:
        h = d['hour']
        if h not in hour_map: hour_map[h] = []
        hour_map[h].append(d['load'])
    for h, v in hour_map.items(): pattern.append((h, 0, np.mean(v)))
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

def calculate_moving_average_load(mins=45):
    cutoff = datetime.now(EAT) - timedelta(minutes=mins)
    recent = [p for t, p in load_history if t >= cutoff]
    return sum(recent) / len(recent) if recent else 0

def generate_load_forecast(pattern, current_avg=0):
    """Generate load forecast with proper fallback to time-based averages"""
    forecast = []
    now = datetime.now(EAT)
    
    for i in range(FORECAST_HOURS):
        ft = now + timedelta(hours=i)
        h = ft.hour
        
        # Start with time-based defaults
        if 0 <= h < 5: base = 600
        elif 5 <= h < 8: base = 1800
        elif 8 <= h < 17: base = 1200
        elif 17 <= h < 22: base = 2800
        else: base = 1000
        
        # Override with historical pattern if available
        if pattern:
            match = next((l for ph, _, l in pattern if ph == h), None)
            if match is not None: base = match
        
        is_spike = current_avg > (base * 1.5)
        
        if current_avg > 0:
            if i == 0: val = (current_avg * 0.8) + (base * 0.2)
            elif i == 1: val = (current_avg * 0.3) + (base * 0.7) if is_spike else (current_avg * 0.5) + (base * 0.5)
            elif i == 2: val = base if is_spike else (current_avg * 0.2) + (base * 0.8)
            else: val = base
        else: 
            val = base
            
        forecast.append({'time': ft, 'hour': h, 'estimated_load': val})
    return forecast

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

def update_patterns(solar, load):
    now = datetime.now(EAT)
    h = now.hour
    clean_s = 0.0 if (h < 6 or h >= 19) else solar
    solar_generation_pattern.append({'timestamp': now, 'hour': h, 'generation': clean_s, 'max_possible': 10000})
    load_demand_pattern.append({'timestamp': now, 'hour': h, 'load': load})

def send_email(subject, html, alert_type="general", send_via_email=True):
    global last_alert_time, alert_history
    cooldown = 120
    if "critical" in alert_type: cooldown = 60
    elif "very_high" in alert_type: cooldown = 30
    
    if alert_type in last_alert_time and (datetime.now(EAT) - last_alert_time[alert_type]) < timedelta(minutes=cooldown):
        return False
        
    success = False
    if send_via_email and all([RESEND_API_KEY, SENDER_EMAIL, RECIPIENT_EMAIL]):
        try:
            r = requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_API_KEY}"}, json={"from": SENDER_EMAIL, "to": [RECIPIENT_EMAIL], "subject": subject, "html": html})
            if r.status_code == 200: success = True
        except: pass
    else: success = True
    
    if success:
        now = datetime.now(EAT)
        last_alert_time[alert_type] = now
        alert_history.append({"timestamp": now, "type": alert_type, "subject": subject})
        alert_history[:] = [a for a in alert_history if a['timestamp'] >= (now - timedelta(hours=12))]
        return True
    return False

def check_alerts(inv_data, solar, total_solar, bat_discharge, gen_run):
    inv1 = next((i for i in inv_data if i['SN'] == 'RKG3B0400T'), None)
    inv2 = next((i for i in inv_data if i['SN'] == 'KAM4N5W0AG'), None)
    inv3 = next((i for i in inv_data if i['SN'] == 'JNK1CDR0KQ'), None)
    if not all([inv1, inv2, inv3]): return
    
    p_cap = min(inv1['Capacity'], inv2['Capacity'])
    b_active = inv3['OutputPower'] > 50
    b_volt = inv3['vBat']
    
    for inv in inv_data:
        if inv.get('communication_lost'): send_email(f"⚠️ Comm Lost: {inv['Label']}", "Check inverter", "communication_lost")
        if inv.get('has_fault'): send_email(f"🚨 FAULT: {inv['Label']}", "Fault code", "fault_alarm")
        if inv.get('high_temperature'): send_email(f"🌡️ High Temp: {inv['Label']}", f"Temp: {inv['temperature']}", "high_temperature")
        
    if gen_run or b_volt < 51.2:
        send_email("🚨 CRITICAL: Generator Running", "Backup critical", "critical")
        return
    if b_active and p_cap < 40:
        send_email("⚠️ HIGH ALERT: Backup Active", "Reduce Load", "backup_active")
        return
    if 40 < p_cap < 50:
        send_email("⚠️ Primary Low", "Reduce Load", "warning", send_via_email=b_active)
    
    if bat_discharge >= 4500: send_email("🚨 URGENT: High Discharge", "Critical", "very_high_load", send_via_email=b_active)
    elif 2500 <= bat_discharge < 3500: send_email("⚠️ High Discharge", "Warning", "high_load", send_via_email=b_active)
    elif 1500 <= bat_discharge < 2000 and p_cap < 50: send_email("ℹ️ Moderate Discharge", "Info", "moderate_load", send_via_email=b_active)

# ----------------------------
# Polling Loop
# ----------------------------
def poll_growatt():
    global latest_data, load_history, battery_history, weather_forecast, last_communication, solar_conditions_cache
    global pool_pump_start_time, pool_pump_last_alert

    print("🚀 Starting polling thread...")
    
    if not TOKEN:
        print("❌ ERROR: API_TOKEN environment variable not set! Polling disabled.")
        return
    
    weather_forecast = get_weather_forecast()
    if weather_forecast: solar_conditions_cache = analyze_solar_conditions(weather_forecast)
    last_wx = datetime.now(EAT)
    
    while True:
        try:
            now = datetime.now(EAT)
            alert_history[:] = [a for a in alert_history if a['timestamp'] >= (now - timedelta(hours=12))]
            
            if (now - last_wx) > timedelta(minutes=30):
                weather_forecast = get_weather_forecast()
                if weather_forecast: solar_conditions_cache = analyze_solar_conditions(weather_forecast)
                last_wx = now
                
            tot_out, tot_bat, tot_sol = 0, 0, 0
            inv_data, p_caps = [], []
            b_data, gen_on = None, False
            
            for sn in SERIAL_NUMBERS:
                try:
                    r = requests.post(API_URL, data={"storage_sn": sn}, headers=headers, timeout=20)
                    r.raise_for_status()
                    data = r.json()
                    
                    # Check for error_code instead of just getting "data"
                    api_code = data.get("error_code", data.get("code", -1))
                    
                    if api_code == 0:  # Success
                        d = data.get("data", {})
                        last_communication[sn] = now
                        cfg = INVERTER_CONFIG.get(sn, {"label": sn, "type": "unknown", "display_order": 99})
                        
                        op = float(d.get("outPutPower") or 0)
                        cap = float(d.get("capacity") or 0)
                        vb = float(d.get("vBat") or 0)
                        pb = float(d.get("pBat") or 0)
                        sol = float(d.get("ppv") or 0) + float(d.get("ppv2") or 0)
                        tmp = max(float(d.get("invTemperature") or 0), float(d.get("dcDcTemperature") or 0), float(d.get("temperature") or 0))
                        flt = int(d.get("errorCode") or 0) != 0
                        
                        tot_out += op
                        tot_sol += sol
                        if pb > 0: tot_bat += pb
                        
                        info = {
                            "SN": sn, "Label": cfg['label'], "Type": cfg['type'], "DisplayOrder": cfg['display_order'],
                            "OutputPower": op, "Capacity": cap, "vBat": vb, "pBat": pb, "ppv": sol, "temperature": tmp,
                            "high_temperature": tmp >= 60, "Status": d.get("statusText", "Unknown"), "has_fault": flt,
                            "last_seen": now.strftime("%Y-%m-%d %H:%M:%S"), "communication_lost": False
                        }
                        inv_data.append(info)
                        
                        if cfg['type'] == 'primary' and cap > 0: p_caps.append(cap)
                        elif cfg['type'] == 'backup':
                            b_data = info
                            if float(d.get("vac") or 0) > 100 or float(d.get("pAcInPut") or 0) > 50: gen_on = True
                    else:
                        print(f"❌ API error for {sn}: Code {api_code}")
                        if sn in last_communication and (now - last_communication[sn]) > timedelta(minutes=10):
                            cfg = INVERTER_CONFIG.get(sn, {})
                            inv_data.append({"SN": sn, "Label": cfg.get('label', sn), "Type": cfg.get('type'), "DisplayOrder": 99, "communication_lost": True})
                except:
                    if sn in last_communication and (now - last_communication[sn]) > timedelta(minutes=10):
                        cfg = INVERTER_CONFIG.get(sn, {})
                        inv_data.append({"SN": sn, "Label": cfg.get('label', sn), "Type": cfg.get('type'), "DisplayOrder": 99, "communication_lost": True})
            
            inv_data.sort(key=lambda x: x.get('DisplayOrder', 99))
            update_patterns(tot_sol, tot_out)
            
            load_history.append((now, tot_out))
            load_history[:] = [(t, p) for t, p in load_history if t >= (now - timedelta(days=14))]
            battery_history.append((now, tot_bat))
            battery_history[:] = [(t, p) for t, p in battery_history if t >= (now - timedelta(days=14))]
            
            s_pat = analyze_historical_solar_pattern()
            l_pat = analyze_historical_load_pattern()
            s_cast = generate_solar_forecast(weather_forecast, s_pat)
            avg_load = calculate_moving_average_load(45)
            l_cast = generate_load_forecast(l_pat, avg_load)
            
            p_min = min(p_caps) if p_caps else 0
            b_volts = b_data['vBat'] if b_data else 0
            b_act = b_data['OutputPower'] > 50 if b_data else False
            b_pct = max(0, min(100, (b_volts - 51.0) / 2.0 * 100))
            
            # Calculate usable energy with correct logic
            usable = calculate_usable_energy(p_min, b_pct)
            
            pred = calculate_battery_cascade(s_cast, l_cast, p_min, b_act)

            if now.hour >= 16:
                if tot_bat > 1100:
                    if pool_pump_start_time is None:
                        pool_pump_start_time = now
                    
                    duration = now - pool_pump_start_time
                    if duration > timedelta(hours=3) and now.hour >= 18:
                        if pool_pump_last_alert is None or (now - pool_pump_last_alert) > timedelta(hours=1):
                            send_email(
                                "⚠️ HIGH LOAD ALERT: Pool Pumps?", 
                                f"Battery discharge has been over 1.1kW for {duration.seconds//3600} hours. Did you leave the pool pumps on?", 
                                "high_load_continuous"
                            )
                            pool_pump_last_alert = now
                else:
                    pool_pump_start_time = None
            else:
                pool_pump_start_time = None
            
            latest_data = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S EAT"),
                "total_output_power": tot_out,
                "total_battery_discharge_W": tot_bat,
                "total_solar_input_W": tot_sol,
                "primary_battery_min": p_min,
                "backup_battery_voltage": b_volts,
                "backup_voltage_status": get_backup_voltage_status(b_volts)[0],
                "backup_active": b_act,
                "backup_percent_calc": b_pct,
                "generator_running": gen_on,
                "inverters": inv_data,
                "solar_forecast": s_cast,
                "load_forecast": l_cast,
                "battery_life_prediction": pred,
                "weather_source": weather_source,
                "usable_energy": usable
            }
            
            print(f"{latest_data['timestamp']} | Load={tot_out:.0f}W | Solar={tot_sol:.0f}W | Battery={usable['total_pct']:.0f}%")
            check_alerts(inv_data, solar_conditions_cache, tot_sol, tot_bat, gen_on)
        except Exception as e: 
            print(f"Error in polling: {e}")
        time.sleep(POLL_INTERVAL_MINUTES * 60)

# ----------------------------
# API Endpoints
# ----------------------------
@app.route("/api/data")
def api_data():
    """Real-time data endpoint for AJAX updates"""
    p_bat = latest_data.get("primary_battery_min", 0)
    b_volt = latest_data.get("backup_battery_voltage", 0)
    tot_load = latest_data.get("total_output_power", 0)
    tot_sol = latest_data.get("total_solar_input_W", 0)
    tot_dis = latest_data.get("total_battery_discharge_W", 0)
    
    return jsonify({
        "timestamp": latest_data.get('timestamp'),
        "load": tot_load,
        "solar": tot_sol,
        "discharge": tot_dis,
        "primary_battery": p_bat,
        "backup_voltage": b_volt,
        "generator_running": latest_data.get("generator_running", False),
        "backup_active": latest_data.get("backup_active", False),
        "inverters": latest_data.get("inverters", []),
        "usable_energy": latest_data.get("usable_energy", {}),
        "alerts": [{"time": a['timestamp'].strftime("%H:%M"), "subject": a['subject'], "type": a['type']} for a in alert_history[-10:]]
    })

# ----------------------------
# Web Interface
# ----------------------------
@app.route("/")
def home():
    def _num(val):
        """Safe number conversion"""
        try:
            return float(val) if val is not None else 0
        except (ValueError, TypeError):
            return 0
    
    # Extract data safely
    p_bat = _num(latest_data.get("primary_battery_min", 0))
    b_volt = _num(latest_data.get("backup_battery_voltage", 0))
    b_stat = latest_data.get("backup_voltage_status", "Unknown")
    b_active = latest_data.get("backup_active", False)
    gen_on = latest_data.get("generator_running", False)
    tot_load = _num(latest_data.get("total_output_power", 0))
    tot_sol = _num(latest_data.get("total_solar_input_W", 0))
    tot_dis = _num(latest_data.get("total_battery_discharge_W", 0))
    
    # Get corrected usable energy
    usable = latest_data.get("usable_energy", {
        "primary_kwh": 0,
        "backup_kwh": 0,
        "total_kwh": 0,
        "total_pct": 0,
        "total_usable_capacity": 29.76
    })
    
    b_pct = _num(latest_data.get("backup_percent_calc", 0))
    
    sol_cond = solar_conditions_cache
    weather_bad = sol_cond and sol_cond['poor_conditions']
    surplus_power = tot_sol - tot_load

    # Status determination
    if gen_on:
        app_st, app_sub, app_col = "⚠️ GENERATOR RUNNING", "Stop all heavy loads immediately", "critical"
        status_icon = "🚨"
    elif b_active:
        app_st, app_sub, app_col = "⚠️ BACKUP ACTIVE", "Primary depleted - conserve power", "critical"
        status_icon = "⚠️"
    elif p_bat < 45 and tot_sol < tot_load:
        app_st, app_sub, app_col = "⚠️ REDUCE LOADS", "Battery low & discharging", "warning"
        status_icon = "⚠️"
    elif usable['total_pct'] > 95:
        app_st, app_sub, app_col = "✅ BATTERY FULL", "System fully charged", "good"
        status_icon = "🔋"
    elif tot_sol > 2000 and (tot_sol > tot_load * 0.9):
        app_st, app_sub, app_col = "✅ SOLAR POWERING", "Solar covering loads", "good"
        status_icon = "☀️"
    elif (usable['total_pct'] > 75 and surplus_power > 3000):
        app_st, app_sub, app_col = "✅ HIGH SURPLUS", f"Heavy loads safe", "good"
        status_icon = "⚡"
    elif weather_bad and usable['total_pct'] > 80:
        app_st, app_sub, app_col = "⚡ USE POWER NOW", "Poor forecast - cook now", "good"
        status_icon = "⚡"
    elif weather_bad and usable['total_pct'] < 70:
        app_st, app_sub, app_col = "☁️ CONSERVE POWER", "Low solar expected", "warning"
        status_icon = "☁️"
    elif surplus_power > 100:
        app_st, app_sub, app_col = "🔋 CHARGING", f"System recovering", "normal"
        status_icon = "🔋"
    else:
        app_st, app_sub, app_col = "ℹ️ NORMAL", "System running", "normal"
        status_icon = "ℹ️"
    
    # Chart data
    if not load_history:
        times = [datetime.now(EAT).strftime('%d %b %H:%M')]
        l_vals = [tot_load]
        b_vals = [tot_dis]
    else:
        total_points = len(load_history)
        step = max(1, total_points // 150)
        times = [t.strftime('%d %b %H:%M') for i, (t, p) in enumerate(load_history) if i % step == 0]
        l_vals = [p for i, (t, p) in enumerate(load_history) if i % step == 0]
        b_vals = [p for i, (t, p) in enumerate(battery_history) if i % step == 0]
    
    pred = latest_data.get("battery_life_prediction")
    sim_t = ["Now"] + [d['time'].strftime('%H:%M') for d in latest_data.get("solar_forecast", [])]
    trace_pct = pred.get('trace_total_pct', []) if pred else []
    
    s_forecast = latest_data.get("solar_forecast", [])
    l_forecast = latest_data.get("load_forecast", [])
    
    if s_forecast and l_forecast:
        forecast_times = [d['time'].strftime('%H:%M') for d in s_forecast[:12]]
        forecast_solar = [d['estimated_generation'] for d in s_forecast[:12]]
        forecast_load = [d['estimated_load'] for d in l_forecast[:12]]
    else:
        now = datetime.now(EAT)
        forecast_times = []
        forecast_solar = []
        forecast_load = []
        for i in range(12):
            hour = (now.hour + i) % 24
            forecast_times.append((now + timedelta(hours=i)).strftime('%H:%M'))
            if 6 <= hour <= 18:
                forecast_solar.append(3000 - abs(12 - hour) * 200)
            else:
                forecast_solar.append(0)
            forecast_load.append(1200)

    # Power flow states - determine which icons should pulse
    solar_active = tot_sol > 100
    battery_charging = surplus_power > 100
    battery_discharging = tot_dis > 100
    
    # Calculate line widths for power flow (proportional to power)
    solar_line_width = max(2, min(8, tot_sol / 1000))
    load_line_width = max(2, min(8, tot_load / 1000))
    battery_line_width = max(2, min(8, tot_dis / 1000))
    
    # Inverter temperature
    inverter_temps = [inv.get('temperature', 0) for inv in latest_data.get('inverters', [])]
    inverter_temp = f"{(sum(inverter_temps) / len(inverter_temps)):.0f}" if inverter_temps else "0"
    
    # Trends
    load_trend_icon = "↑" if tot_load > 2000 else "→" if tot_load > 1000 else "↓"
    load_trend_text = "High" if tot_load > 2000 else "Moderate" if tot_load > 1000 else "Low"
    
    solar_trend_icon = "☀️" if tot_sol > 5000 else "⛅" if tot_sol > 2000 else "☁️"
    solar_trend_text = "Excellent" if tot_sol > 5000 else "Good" if tot_sol > 2000 else "Low"
    
    primary_color = "text-success" if p_bat > 60 else "text-warning" if p_bat > 40 else "text-danger"
    backup_color = "text-success" if b_volt > 52.3 else "text-warning" if b_volt > 51.5 else "text-danger"
    
    # Battery bar color based on usable percentage
    if usable['total_pct'] >= 60:
        battery_bar_color = "success"
    elif usable['total_pct'] >= 25:
        battery_bar_color = "warning"
    else:
        battery_bar_color = "danger"
    
    alerts = [{"time": a['timestamp'].strftime("%H:%M"), "subject": a['subject'], "type": a['type']} 
              for a in reversed(alert_history[-10:])]
    
    # Smart Recommendations - UPDATED LOGIC: only recommend heavy loads when primary battery > 75%
    recommendation_items = []
    
    safe_statuses = ["USE POWER NOW", "HIGH SURPLUS", "BATTERY FULL", "SOLAR POWERING"]
    is_safe_now = any(s in app_st for s in safe_statuses)
    
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
    elif is_safe_now and p_bat > 75:  # Only recommend heavy loads when primary battery > 75%
        recommendation_items.append({
            'icon': '✅',
            'title': 'SAFE TO USE HEAVY LOADS',
            'description': f'Primary battery: {p_bat:.0f}% (>75%) | Surplus: {surplus_power:.0f}W',
            'class': 'good'
        })
    elif usable['total_pct'] < 50 and tot_sol < tot_load:
        recommendation_items.append({
            'icon': '⚠️',
            'title': 'CONSERVE POWER',
            'description': f'Battery low ({usable["total_pct"]:.0f}%) and not charging well',
            'class': 'warning'
        })
    elif p_bat <= 75 and is_safe_now:
        recommendation_items.append({
            'icon': '⚠️',
            'title': 'LIMIT HEAVY LOADS',
            'description': f'Primary battery {p_bat:.0f}% (≤75%) - use moderate loads only',
            'class': 'warning'
        })
    else:
        recommendation_items.append({
            'icon': 'ℹ️',
            'title': 'MONITOR USAGE',
            'description': 'Check schedule below for optimal times',
            'class': 'normal'
        })
    
    # Schedule items
    schedule_items = []
    
    if s_forecast:
        best_start, best_end, current_run = None, None, 0
        temp_start = None
        for d in s_forecast:
            gen = d['estimated_generation']
            if gen > 2000:
                if current_run == 0: 
                    temp_start = d['time']
                current_run += 1
            else:
                if current_run > 0:
                    if best_start is None or current_run > ((best_end.hour if best_end else 0) - (best_start.hour if best_start else 0)):
                        best_start = temp_start
                        best_end = d['time']
                    current_run = 0
        
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
                'title': 'No High Solar Window',
                'time': 'Avoid heavy loads today',
                'class': 'warning'
            })
        
        # Cloud warnings
        next_3_gen = sum([d['estimated_generation'] for d in s_forecast[:3]]) / 3 if len(s_forecast) >= 3 else 0
        current_hour = datetime.now(EAT).hour
        if next_3_gen < 500 and 8 <= current_hour <= 16:
            schedule_items.append({
                'icon': '☁️',
                'title': 'Cloud Warning',
                'time': 'Low solar next 3 hours',
                'class': 'warning'
            })
    
    # Calculate runtime estimate
    if tot_load > 0 and usable['total_kwh'] > 0:
        runtime_hours = (usable['total_kwh'] * 1000) / tot_load
    else:
        runtime_hours = 0

    # HTML template
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tulia House Solar</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border-color: #30363d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
            --info: #58a6ff;
            --good-bg: rgba(63, 185, 80, 0.15);
            --warning-bg: rgba(210, 153, 34, 0.15);
            --critical-bg: rgba(248, 81, 73, 0.15);
            --normal-bg: rgba(88, 166, 255, 0.15);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'DM Sans', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 20px;
        }
        .span-12 { grid-column: span 12; }
        .span-9 { grid-column: span 9; }
        .span-6 { grid-column: span 6; }
        .span-4 { grid-column: span 4; }
        .span-3 { grid-column: span 3; }
        header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }
        h1 {
            font-family: 'Space Mono', monospace;
            font-size: 2.5rem;
            background: linear-gradient(90deg, var(--info), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        .card:hover { border-color: var(--text-muted); }
        .metric-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            font-family: 'Space Mono', monospace;
        }
        .metric-unit {
            font-size: 1.2rem;
            color: var(--text-secondary);
            margin-left: 4px;
        }
        .status-hero {
            padding: 30px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 20px;
        }
        .status-hero.good { background: var(--good-bg); border: 2px solid var(--success); }
        .status-hero.warning { background: var(--warning-bg); border: 2px solid var(--warning); }
        .status-hero.critical { background: var(--critical-bg); border: 2px solid var(--danger); }
        .status-hero.normal { background: var(--normal-bg); border: 2px solid var(--info); }
        .status-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .text-success { color: var(--success); }
        .text-warning { color: var(--warning); }
        .text-danger { color: var(--danger); }
        .text-info { color: var(--info); }
        .chart-container {
            height: 300px;
            position: relative;
            margin-top: 20px;
        }
        .progress-bar {
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            transition: width 0.5s ease;
        }
        .progress-success { background: var(--success); }
        .progress-warning { background: var(--warning); }
        .progress-danger { background: var(--danger); }
        .alert-item {
            padding: 12px;
            border-left: 4px solid;
            margin-bottom: 10px;
            background: var(--bg-tertiary);
            border-radius: 0 8px 8px 0;
        }
        .alert-critical { border-color: var(--danger); }
        .alert-warning { border-color: var(--warning); }
        .alert-info { border-color: var(--info); }
        .inverter-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .inverter-card {
            padding: 15px;
            border-radius: 8px;
            background: var(--bg-tertiary);
        }
        .power-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 40px;
            padding: 20px;
            background: var(--bg-tertiary);
            border-radius: 12px;
            margin: 20px 0;
        }
        .flow-icon {
            font-size: 3rem;
            position: relative;
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }
        .flow-line {
            height: 4px;
            flex-grow: 1;
            background: var(--border-color);
            position: relative;
        }
        .flow-line.active {
            background: linear-gradient(90deg, var(--success), transparent);
            animation: flow 2s linear infinite;
        }
        @keyframes flow {
            0% { background-position: -100% 0; }
            100% { background-position: 200% 0; }
        }
        .recommendation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .recommendation-item {
            padding: 15px;
            border-radius: 8px;
            background: var(--bg-tertiary);
            border-left: 4px solid;
        }
        .recommendation-item.good { border-color: var(--success); }
        .recommendation-item.warning { border-color: var(--warning); }
        .recommendation-item.critical { border-color: var(--danger); }
        .recommendation-item.normal { border-color: var(--info); }
        .update-time {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }
        @media (max-width: 1200px) {
            .dashboard-grid { grid-template-columns: repeat(6, 1fr); }
            .span-12, .span-9, .span-6, .span-4, .span-3 { grid-column: span 6; }
        }
        @media (max-width: 768px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .span-12, .span-9, .span-6, .span-4, .span-3 { grid-column: 1; }
            h1 { font-size: 2rem; }
            .power-flow { flex-direction: column; gap: 20px; }
            .flow-line { width: 4px; height: 40px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="dashboard-grid">
            <header class="span-12">
                <h1>TULIA HOUSE SOLAR</h1>
                <div class="subtitle">{{ timestamp }}</div>
            </header>
            
            <!-- Status Hero -->
            <div class="span-12 status-hero {{ app_col }}">
                <div style="font-size: 3rem; margin-bottom: 0.5rem">{{ status_icon }}</div>
                <div class="status-title">{{ app_st }}</div>
                <div style="font-size: 1.1rem; opacity: 0.9">{{ app_sub }}</div>
            </div>
            
            <!-- Key Metrics (Row of 4) -->
            <div class="card span-3">
                <div class="metric-label">Current Load</div>
                <div class="metric-value text-info">{{ '%0.f'|format(tot_load) }}<span class="metric-unit">W</span></div>
                <div style="font-size: 0.85rem; color: var(--text-muted)">{{ load_trend_icon }} {{ load_trend_text }} demand</div>
            </div>
            
            <div class="card span-3">
                <div class="metric-label">Solar Output</div>
                <div class="metric-value text-success">{{ '%0.f'|format(tot_sol) }}<span class="metric-unit">W</span></div>
                <div style="font-size: 0.85rem; color: var(--text-muted)">{{ solar_trend_icon }} {{ solar_trend_text }} production</div>
            </div>
            
            <div class="card span-3">
                <div class="metric-label">Primary Battery</div>
                <div class="metric-value {{ primary_color }}">{{ '%0.f'|format(p_bat) }}<span class="metric-unit">%</span></div>
                <div style="font-size: 0.85rem; color: var(--text-muted)">Raw system reading</div>
            </div>
            
            <div class="card span-3">
                <div class="metric-label">Backup Voltage</div>
                <div class="metric-value {{ backup_color }}">{{ '%0.1f'|format(b_volt) }}<span class="metric-unit">V</span></div>
                <div style="font-size: 0.85rem; color: var(--text-muted)">Status: {{ b_stat }}</div>
            </div>
            
            <!-- Usable Energy -->
            <div class="card span-6">
                <div class="metric-label">Usable Energy (Corrected)</div>
                <div class="metric-value text-success">{{ '%0.1f'|format(usable.total_kwh) }}<span class="metric-unit">kWh</span></div>
                <div style="font-size: 1.2rem; margin-top: 5px; color: var(--text-secondary)">
                    {{ '%0.f'|format(usable.total_pct) }}% of {{ usable.total_usable_capacity }} kWh usable capacity
                </div>
                <div class="progress-bar">
                    <div class="progress-fill progress-{{ battery_bar_color }}" style="width: {{ usable.total_pct }}%"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 15px; font-size: 0.9rem;">
                    <div>Primary: {{ '%0.1f'|format(usable.primary_kwh) }} kWh</div>
                    <div>Backup: {{ '%0.1f'|format(usable.backup_kwh) }} kWh</div>
                    <div>Runtime: {{ '%0.1f'|format(runtime_hours) }}h</div>
                </div>
            </div>
            
            <!-- System Status -->
            <div class="card span-6">
                <div class="metric-label">System Status</div>
                <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px;">
                    <div style="flex: 1; min-width: 120px;">
                        <div style="font-size: 0.9rem; color: var(--text-secondary)">Generator</div>
                        <div style="font-size: 1.5rem; color: {{ 'var(--danger)' if gen_on else 'var(--success)' }}">
                            {{ '🚨 RUNNING' if gen_on else '✅ OFF' }}
                        </div>
                    </div>
                    <div style="flex: 1; min-width: 120px;">
                        <div style="font-size: 0.9rem; color: var(--text-secondary)">Backup Active</div>
                        <div style="font-size: 1.5rem; color: {{ 'var(--warning)' if b_active else 'var(--success)' }}">
                            {{ '⚠️ ACTIVE' if b_active else '✅ INACTIVE' }}
                        </div>
                    </div>
                    <div style="flex: 1; min-width: 120px;">
                        <div style="font-size: 0.9rem; color: var(--text-secondary)">Inverter Temp</div>
                        <div style="font-size: 1.5rem; color: {{ 'var(--danger)' if inverter_temp|float > 60 else 'var(--warning)' if inverter_temp|float > 50 else 'var(--success)' }}">
                            {{ inverter_temp }}°C
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Power Flow Visualization -->
            <div class="card span-12">
                <div class="metric-label">Power Flow</div>
                <div class="power-flow">
                    <div class="flow-icon {{ 'pulse' if solar_active else '' }}">☀️</div>
                    <div class="flow-line {{ 'active' if solar_active else '' }}" style="width: {{ solar_line_width }}px"></div>
                    
                    <div class="flow-icon">🔋</div>
                    <div class="flow-line {{ 'active' if battery_charging or battery_discharging else '' }}" style="width: {{ battery_line_width }}px"></div>
                    
                    <div class="flow-icon">🏠</div>
                    <div class="flow-line active" style="width: {{ load_line_width }}px"></div>
                    
                    <div class="flow-icon">⚡</div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 15px; text-align: center;">
                    <div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary)">Solar</div>
                        <div style="font-size: 1.2rem">{{ '%0.f'|format(tot_sol) }}W</div>
                    </div>
                    <div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary)">Battery</div>
                        <div style="font-size: 1.2rem; color: {{ 'var(--success)' if battery_charging else 'var(--warning)' if battery_discharging else 'var(--text-secondary)' }}">
                            {{ '%0.f'|format(tot_dis) }}W {{ '↑' if battery_charging else '↓' if battery_discharging else '→' }}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary)">Load</div>
                        <div style="font-size: 1.2rem">{{ '%0.f'|format(tot_load) }}W</div>
                    </div>
                    <div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary)">Net</div>
                        <div style="font-size: 1.2rem; color: {{ 'var(--success)' if surplus_power > 0 else 'var(--warning)' }}">
                            {{ '%0.f'|format(surplus_power) }}W
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Solar & Load Forecast -->
            <div class="card span-9">
                <div class="metric-label">Solar & Load Forecast (Next 12 Hours)</div>
                <div class="chart-container">
                    <canvas id="forecastChart"></canvas>
                </div>
                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 10px;">
                    Weather Source: {{ weather_source }}
                </div>
            </div>
            
            <!-- Smart Recommendations -->
            <div class="card span-3">
                <div class="metric-label">Smart Recommendations</div>
                <div class="recommendation-grid">
                    {% for item in recommendation_items %}
                    <div class="recommendation-item {{ item.class }}">
                        <div style="font-size: 1.5rem; margin-bottom: 5px">{{ item.icon }}</div>
                        <div style="font-weight: 600; margin-bottom: 5px">{{ item.title }}</div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary)">{{ item.description }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <!-- Inverter Status -->
            <div class="card span-6">
                <div class="metric-label">Inverter Status</div>
                <div class="inverter-grid">
                    {% for inv in latest_data.inverters %}
                    <div class="inverter-card" style="border-left: 4px solid {{ 'var(--danger)' if inv.communication_lost else 'var(--warning)' if inv.high_temperature or inv.has_fault else 'var(--success)' }}">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div>
                                <div style="font-weight: 600">{{ inv.Label }}</div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary)">{{ inv.Type|title }} • {{ inv.SN[:8] }}...</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.5rem; font-family: 'Space Mono'">{{ '%0.f'|format(inv.OutputPower) }}W</div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary)">Output</div>
                            </div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Battery:</span>
                                <span>{{ '%0.f'|format(inv.Capacity) if inv.Capacity else '?' }}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Solar:</span>
                                <span>{{ '%0.f'|format(inv.ppv) }}W</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Temp:</span>
                                <span style="color: {{ 'var(--danger)' if inv.temperature|float > 60 else 'var(--warning)' if inv.temperature|float > 50 else 'inherit' }}">
                                    {{ '%0.f'|format(inv.temperature) }}°C
                                </span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Status:</span>
                                <span style="color: {{ 'var(--danger)' if inv.communication_lost else 'var(--warning)' if inv.has_fault else 'var(--success)' }}">
                                    {{ '🔴 Offline' if inv.communication_lost else '⚠️ Fault' if inv.has_fault else '✅ ' + inv.Status }}
                                </span>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <!-- Load Schedule & Battery Simulation -->
            <div class="card span-6">
                <div class="metric-label">Optimal Load Schedule</div>
                <div class="recommendation-grid">
                    {% for item in schedule_items %}
                    <div class="recommendation-item {{ item.class }}">
                        <div style="font-size: 1.5rem; margin-bottom: 5px">{{ item.icon }}</div>
                        <div style="font-weight: 600; margin-bottom: 5px">{{ item.title }}</div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary)">{{ item.time }}</div>
                    </div>
                    {% endfor %}
                </div>
                
                {% if trace_pct %}
                <div style="margin-top: 20px;">
                    <div class="metric-label">Battery Simulation</div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 10px;">
                        {% set pred = latest_data.battery_life_prediction %}
                        {% if pred.generator_needed %}
                        ⚠️ Generator needed at {{ pred.time_empty }} ({{ '%0.1f'|format(pred.genset_hours) }} hours)
                        {% elif pred.switchover_occurred %}
                        ℹ️ Backup battery will be used
                        {% else %}
                        ✅ Battery sufficient for forecast period
                        {% endif %}
                    </div>
                    <div class="chart-container">
                        <canvas id="batteryChart"></canvas>
                    </div>
                </div>
                {% endif %}
            </div>
            
            <!-- Recent Alerts -->
            {% if alerts %}
            <div class="card span-12">
                <div class="metric-label">Recent Alerts</div>
                <div style="margin-top: 15px;">
                    {% for alert in alerts %}
                    <div class="alert-item alert-{{ alert.type.split('_')[0] if '_' in alert.type else 'info' }}">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div style="font-weight: 600">{{ alert.subject[:50] }}{{ '...' if alert.subject|length > 50 else '' }}</div>
                            <div style="font-size: 0.8rem; color: var(--text-secondary)">{{ alert.time }}</div>
                        </div>
                        <div style="font-size: 0.9rem; color: var(--text-muted)">{{ alert.type|replace('_', ' ')|title }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
            
            <div class="span-12 update-time">
                Data updates every {{ POLL_INTERVAL_MINUTES }} minutes. Last API poll: {{ latest_data.timestamp }}
                {% if not TOKEN %}
                <div style="color: var(--danger); margin-top: 10px;">
                    ⚠️ API_TOKEN not set. Polling disabled. Set environment variable in Railway.
                </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <script>
        // Chart Config
        Chart.defaults.color = '#8a95a8';
        Chart.defaults.borderColor = 'rgba(58, 70, 89, 0.4)';
        Chart.defaults.font.family = "'DM Sans', sans-serif";
        
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', align: 'end', labels: { boxWidth: 10, usePointStyle: true, font: { size: 11 } } }
            },
            interaction: { mode: 'index', intersect: false }
        };

        // Forecast Chart
        new Chart(document.getElementById('forecastChart'), {
            type: 'line',
            data: {
                labels: {{ forecast_times|tojson }},
                datasets: [
                    { 
                        label: 'Solar', 
                        data: {{ forecast_solar|tojson }}, 
                        borderColor: '#3fb950', 
                        backgroundColor: 'rgba(63, 185, 80, 0.15)', 
                        fill: true, 
                        tension: 0.4,
                        borderWidth: 2
                    },
                    { 
                        label: 'Load', 
                        data: {{ forecast_load|tojson }}, 
                        borderColor: '#58a6ff', 
                        backgroundColor: 'rgba(88, 166, 255, 0.15)', 
                        fill: true, 
                        tension: 0.4,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Power (W)' } },
                    x: { title: { display: true, text: 'Time' } }
                }
            }
        });
        
        // Battery Simulation Chart
        {% if trace_pct %}
        new Chart(document.getElementById('batteryChart'), {
            type: 'line',
            data: {
                labels: {{ sim_t|tojson }},
                datasets: [{
                    label: 'Battery Level',
                    data: {{ trace_pct|tojson }},
                    borderColor: '#d29922',
                    backgroundColor: 'rgba(210, 153, 34, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    y: { 
                        beginAtZero: true, 
                        max: 100,
                        title: { display: true, text: 'Battery %' },
                        ticks: { callback: value => value + '%' }
                    },
                    x: { title: { display: true, text: 'Time' } }
                },
                plugins: {
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                yMin: 40,
                                yMax: 40,
                                borderColor: '#f85149',
                                borderWidth: 1,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: 'Primary cutoff',
                                    position: 'end'
                                }
                            },
                            line2: {
                                type: 'line',
                                yMin: 20,
                                yMax: 20,
                                borderColor: '#d29922',
                                borderWidth: 1,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: 'Backup cutoff',
                                    position: 'end'
                                }
                            }
                        }
                    }
                }
            }
        });
        {% endif %}
        
        // Auto Refresh
        setInterval(() => {
            fetch('/api/data').then(r => r.json()).then(d => {
                if(d.timestamp !== "{{ latest_data.timestamp }}") {
                    location.reload();
                }
            }).catch(err => {
                console.log('Auto-refresh failed:', err);
            });
        }, 60000);
        
        // Check if polling is active
        fetch('/health').then(r => r.json()).then(data => {
            if(!data.polling_active && data.status === 'healthy') {
                console.warn('Polling thread not active but app is healthy');
            }
        });
    </script>
</body>
</html>
    """
    
    from flask import render_template_string
    return render_template_string(
        html_template,
        timestamp=latest_data.get('timestamp', 'Initializing...'),
        status_icon=status_icon,
        app_st=app_st,
        app_sub=app_sub,
        app_col=app_col,
        tot_load=tot_load,
        tot_sol=tot_sol,
        tot_dis=tot_dis,
        p_bat=p_bat,
        b_volt=b_volt,
        b_pct=b_pct,
        b_stat=b_stat,
        usable=usable,
        load_trend_icon=load_trend_icon,
        load_trend_text=load_trend_text,
        solar_trend_icon=solar_trend_icon,
        solar_trend_text=solar_trend_text,
        primary_color=primary_color,
        backup_color=backup_color,
        battery_bar_color=battery_bar_color,
        solar_active=solar_active,
        battery_charging=battery_charging,
        battery_discharging=battery_discharging,
        gen_on=gen_on,
        b_active=b_active,
        inverter_temp=inverter_temp,
        solar_line_width=solar_line_width,
        load_line_width=load_line_width,
        battery_line_width=battery_line_width,
        recommendation_items=recommendation_items,
        schedule_items=schedule_items,
        forecast_times=forecast_times,
        forecast_solar=forecast_solar,
        forecast_load=forecast_load,
        sim_t=sim_t,
        trace_pct=trace_pct,
        times=times,
        l_vals=l_vals,
        b_vals=b_vals,
        latest_data=latest_data,
        alerts=alerts,
        runtime_hours=runtime_hours,
        POLL_INTERVAL_MINUTES=POLL_INTERVAL_MINUTES,
        weather_source=weather_source
    )

# ================ RAILWAY/PRODUCTION SETTINGS ================
if __name__ == "__main__":
    import threading
    
    # Debug environment
    print("=" * 50)
    print("RAILWAY DEPLOYMENT STARTING")
    print(f"PORT: {os.getenv('PORT', '10000')}")
    print(f"API_TOKEN set: {'YES' if TOKEN else 'NO'}")
    print(f"SERIAL_NUMBERS: {SERIAL_NUMBERS}")
    print("=" * 50)
    
    # Start polling thread only if token is set
    if TOKEN and SERIAL_NUMBERS and SERIAL_NUMBERS[0]:
        poll_thread = threading.Thread(target=poll_growatt, daemon=True)
        poll_thread.start()
        print(f"✅ Polling thread started with interval {POLL_INTERVAL_MINUTES} minutes")
    else:
        print("⚠️ WARNING: API_TOKEN not set or SERIAL_NUMBERS empty, polling disabled")
        print("⚠️ Add API_TOKEN and SERIAL_NUMBERS in Railway environment variables")
    
    port = int(os.getenv("PORT", 10000))
    
    # Run with Railway-compatible settings
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=False, 
        threaded=True,
        use_reloader=False
    )
