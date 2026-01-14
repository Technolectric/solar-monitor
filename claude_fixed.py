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

# ----------------------------
# 1. Logic Engine: Persistence & Detection
# ----------------------------
class PersistentLoadManager:
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
        self.hourly_data = []  # Store last 24 hours of granular data
        
    def load_history(self):
        if Path(self.filename).exists():
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except: pass
        return {}
    
    def save_history(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.history, f)
        except: pass
    
    def add_hourly_datapoint(self, timestamp, load_w, battery_discharge_w, solar_w):
        """Add a datapoint for 24-hour detailed chart"""
        self.hourly_data.append({
            'timestamp': timestamp.isoformat(),
            'load': load_w,
            'battery_discharge': battery_discharge_w,
            'solar': solar_w
        })
        # Keep only last 24 hours (at 5min intervals = 288 points)
        if len(self.hourly_data) > 288:
            self.hourly_data = self.hourly_data[-288:]
    
    def get_last_24h_data(self):
        """Return last 24 hours of granular data"""
        return self.hourly_data
    
    def update_daily(self, date_str, total_consumption_wh, total_solar_wh, max_solar_potential_wh):
        """Update daily totals"""
        if date_str not in self.history:
            self.history[date_str] = {
                'consumption': 0,
                'solar': 0,
                'potential': max_solar_potential_wh
            }
        self.history[date_str]['consumption'] = total_consumption_wh
        self.history[date_str]['solar'] = total_solar_wh
        
        # Keep only last 30 days
        dates = sorted(self.history.keys())
        if len(dates) > 30:
            for old_date in dates[:-30]:
                del self.history[old_date]
    
    def get_last_30_days(self):
        """Return last 30 days of data for heatmap"""
        now = datetime.now(EAT)
        result = []
        for i in range(29, -1, -1):
            date = now - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            data = self.history.get(date_str, {'consumption': 0, 'solar': 0, 'potential': 0})
            
            # Calculate efficiency (how much of potential was used)
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
    detected = []
    delta = current - previous
    if gen_active:
        if primary_pct > 42: detected.append("Water Heating")
        else: detected.append("System Charging")
    if current < 400: detected.append("Idle")
    elif 1000 <= current <= 1350: detected.append("Pool Pump")
    elif current > 1800: detected.append("Cooking")
    elif 400 <= current < 1000: detected.append("TV/Lights")
    if delta > 1500: detected.append("Kettle")
    return detected

# ----------------------------
# 2. Physics & Scheduler Engine
# ----------------------------

APPLIANCE_PROFILES = [
    {"id": "pool", "name": "Pool Pump", "watts": 1200, "hours": 4, "icon": "🏊"},
    {"id": "wash", "name": "Washer", "watts": 800, "hours": 1.5, "icon": "🧺"},
    {"id": "oven", "name": "Oven", "watts": 2500, "hours": 1.5, "icon": "🍳"}
]

def generate_smart_schedule(p_pct, s_forecast, l_forecast):
    """Generates advice with status codes for UI styling."""
    advice = []
    is_safe_now = p_pct > 60
    
    for app in APPLIANCE_PROFILES:
        decision = {}
        if is_safe_now:
             decision = {"msg": "Safe to Run", "status": "safe", "color": "var(--success)"}
        else:
             decision = {"msg": "Wait for Solar", "status": "unsafe", "color": "var(--warn)"}
             
        advice.append({**app, **decision})
    return advice

def calculate_battery_breakdown(p_pct, b_volts):
    """
    Calculates breakdown for circular chart with tiered discharge strategy:
    1. Primary Battery: 100% → 40% (18 kWh usable)
    2. Backup Battery: 53.0V → 51.2V (16.8 kWh usable, 80% of degraded capacity)
    3. Emergency Primary: 40% → 20% (6 kWh emergency reserve)
    
    Voltage formula for backup: 51.0V = 0%, 53.0V = 100%
    So: battery_pct = (voltage - 51.0) / 2.0 * 100
    """
    # Backup battery percentage from voltage
    b_pct = max(0, min(100, (b_volts - 51.0) / 2.0 * 100))
    
    # Total capacities
    p_total_wh = PRIMARY_BATTERY_CAPACITY_WH  # 30,000 Wh
    b_total_wh = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION  # 21,000 * 0.7 = 14,700 Wh
    
    # Current energy stored
    curr_p_wh = (p_pct / 100) * p_total_wh
    curr_b_wh = (b_pct / 100) * b_total_wh
    
    # Tiered availability calculation
    # Tier 1: Primary from 100% down to 40% (18 kWh available)
    primary_tier1_capacity = p_total_wh * 0.60  # 18,000 Wh
    primary_tier1_threshold = p_total_wh * 0.40  # 12,000 Wh
    
    if curr_p_wh > primary_tier1_threshold:
        primary_tier1_available = curr_p_wh - primary_tier1_threshold
    else:
        primary_tier1_available = 0
    
    # Tier 2: Backup battery 80% usable (16.8 kWh, down to 20% before generator kicks in)
    backup_capacity = b_total_wh * 0.80  # 11,760 Wh usable
    backup_threshold = b_total_wh * 0.20  # 2,940 Wh (generator trigger point)
    
    if curr_b_wh > backup_threshold:
        backup_available = curr_b_wh - backup_threshold
    else:
        backup_available = 0
    
    # Tier 3: Emergency Primary from 40% down to 20% (6 kWh for backup inverter overload)
    emergency_capacity = p_total_wh * 0.20  # 6,000 Wh
    emergency_threshold = p_total_wh * 0.20  # 6,000 Wh
    
    if curr_p_wh <= primary_tier1_threshold and curr_p_wh > emergency_threshold:
        emergency_available = curr_p_wh - emergency_threshold
    else:
        emergency_available = 0
    
    # Total system usable capacity
    total_system_capacity = primary_tier1_capacity + backup_capacity + emergency_capacity  # 18 + 11.76 + 6 = 35.76 kWh
    
    # Current total available
    total_available = primary_tier1_available + backup_available + emergency_available
    
    # Overall percentage
    total_pct = (total_available / total_system_capacity * 100) if total_system_capacity > 0 else 0
    
    return {
        'chart_data': [
            round(primary_tier1_available / 1000, 1),  # Primary Tier 1
            round(backup_available / 1000, 1),          # Backup
            round(emergency_available / 1000, 1)        # Reserve
        ],
        'tier_labels': ['Primary (100-40%)', 'Backup (80-20%)', 'Reserve (40-20%)'],
        'total_pct': round(total_pct, 1),
        'total_kwh': round(total_available / 1000, 1),
        'primary_pct': p_pct,
        'backup_voltage': round(b_volts, 1),  # Format to 1 decimal
        'backup_pct': round(b_pct, 1)
    }

def calculate_battery_cascade(solar, load, p_pct, b_volts, b_active=False):
    """
    Simulates battery levels over forecast period using tiered discharge:
    1. Primary 100% → 40% (18 kWh)
    2. Backup 80% → 20% (based on voltage)
    3. Emergency Primary 40% → 20% (6 kWh)
    """
    if not solar or not load: return {'labels': [], 'data': []}
    
    # System capacities
    p_total_wh = PRIMARY_BATTERY_CAPACITY_WH  # 30,000 Wh
    b_total_wh = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION  # 14,700 Wh
    
    # Starting state
    curr_p_wh = (p_pct / 100.0) * p_total_wh
    b_pct = max(0, min(100, (b_volts - 51.0) / 2.0 * 100))
    curr_b_wh = (b_pct / 100.0) * b_total_wh
    
    sim_data, sim_labels = [], []
    
    for i in range(min(len(solar), len(load))):
        net = solar[i]['estimated_generation'] - load[i]['estimated_load']
        
        if net > 0:  # Charging
            # Charge primary first up to 100%
            space_in_primary = p_total_wh - curr_p_wh
            if net <= space_in_primary:
                curr_p_wh += net
            else:
                curr_p_wh = p_total_wh
                # Overflow charges backup
                overflow = net - space_in_primary
                curr_b_wh = min(b_total_wh, curr_b_wh + overflow)
        else:  # Discharging
            drain = abs(net)
            
            # Tier 1: Use primary from current level down to 40%
            primary_min = p_total_wh * 0.40  # 12,000 Wh
            available_tier1 = max(0, curr_p_wh - primary_min)
            
            if available_tier1 >= drain:
                curr_p_wh -= drain
                drain = 0
            else:
                curr_p_wh = primary_min
                drain -= available_tier1
            
            # Tier 2: Use backup if drain remains
            if drain > 0:
                backup_min = b_total_wh * 0.20  # 2,940 Wh (generator trigger)
                available_backup = max(0, curr_b_wh - backup_min)
                
                if available_backup >= drain:
                    curr_b_wh -= drain
                    drain = 0
                else:
                    curr_b_wh = backup_min
                    drain -= available_backup
            
            # Tier 3: Emergency primary (40% → 20%)
            if drain > 0:
                emergency_min = p_total_wh * 0.20  # 6,000 Wh
                available_emergency = max(0, curr_p_wh - emergency_min)
                
                if available_emergency >= drain:
                    curr_p_wh -= drain
                else:
                    curr_p_wh = emergency_min
        
        # Calculate total available across all tiers
        primary_tier1_avail = max(0, curr_p_wh - (p_total_wh * 0.40))
        backup_avail = max(0, curr_b_wh - (b_total_wh * 0.20))
        emergency_avail = max(0, min(curr_p_wh, p_total_wh * 0.40) - (p_total_wh * 0.20))
        
        total_capacity = (p_total_wh * 0.60) + (b_total_wh * 0.80) + (p_total_wh * 0.20)
        total_available = primary_tier1_avail + backup_avail + emergency_avail
        
        percentage = (total_available / total_capacity) * 100 if total_capacity > 0 else 0
        sim_data.append(percentage)
        sim_labels.append(solar[i]['time'].strftime('%H:%M'))
    
    return {'labels': sim_labels, 'data': sim_data}

# ----------------------------
# 3. Helpers
# ----------------------------
headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"} if TOKEN else {}
last_alert_time, alert_history = {}, []
daily_accumulator = {'consumption_wh': 0, 'solar_wh': 0, 'last_date': None}
latest_data = {
    "timestamp": "Initializing...", "total_output_power": 0, "total_solar_input_W": 0,
    "primary_battery_min": 0, "backup_battery_voltage": 0, "backup_active": False,
    "generator_running": False, "inverters": [], "detected_appliances": [], 
    "solar_forecast": [], "load_forecast": [], 
    "battery_sim": {"labels": [], "data": []},
    "energy_breakdown": {"chart_data": [1, 0, 1], "total_pct": 0, "total_kwh": 0},
    "scheduler": [],
    "heatmap_data": [],
    "hourly_24h": []
}

def get_weather_forecast():
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=shortwave_radiation&timezone=Africa/Nairobi&forecast_days=2"
        r = requests.get(url, timeout=5).json()
        return {'times': r['hourly']['time'], 'rad': r['hourly']['shortwave_radiation']}
    except: return None

def generate_solar_forecast(weather_data):
    forecast = []
    now = datetime.now(EAT)
    if not weather_data: return forecast
    w_map = {t: r for t, r in zip(weather_data['times'], weather_data['rad'])}
    for i in range(24):
        ft = now + timedelta(hours=i)
        key = ft.strftime('%Y-%m-%dT%H:00')
        rad = w_map.get(key, 0)
        est = (rad / 1000.0) * (TOTAL_SOLAR_CAPACITY_KW * 1000) * SOLAR_EFFICIENCY_FACTOR
        forecast.append({'time': ft, 'estimated_generation': est})
    return forecast

def send_email(subject, html, alert_type="general"):
    global last_alert_time, alert_history
    if alert_type in last_alert_time and (datetime.now(EAT) - last_alert_time[alert_type]) < timedelta(minutes=60):
        return
    if RESEND_API_KEY:
        try:
            requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_API_KEY}"}, json={"from": SENDER_EMAIL, "to": [RECIPIENT_EMAIL], "subject": subject, "html": html})
        except: pass
    now = datetime.now(EAT)
    last_alert_time[alert_type] = now
    alert_history.insert(0, {"timestamp": now, "type": alert_type, "subject": subject})
    alert_history = alert_history[:20]

# ----------------------------
# 4. Polling Loop
# ----------------------------
polling_active = False
polling_thread = None

def poll_growatt():
    global latest_data, polling_active, daily_accumulator
    if not TOKEN: return

    wx_data = get_weather_forecast()
    prev_watts = 0 
    last_save = datetime.now(EAT)
    polling_active = True
    
    print("🚀 System Started: Enhanced Dashboard Mode")

    while polling_active:
        try:
            now = datetime.now(EAT)
            tot_out, tot_sol, tot_bat = 0, 0, 0
            inv_data, p_caps = [], []
            b_data, gen_on = None, False
            
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
                        
                        tot_out += op
                        tot_sol += sol
                        if pb > 0: tot_bat += pb
                        
                        cfg = INVERTER_CONFIG.get(sn, {"label": sn, "type": "unknown"})
                        info = {"SN": sn, "Label": cfg['label'], "OutputPower": op, "Capacity": cap, "vBat": vb, "temp": temp}
                        inv_data.append(info)
                        
                        if cfg['type'] == 'primary': p_caps.append(cap)
                        elif cfg['type'] == 'backup':
                            b_data = info
                            if float(d.get("vac") or 0) > 100 or float(d.get("pAcInPut") or 0) > 50: gen_on = True
                except: pass

            p_min = min(p_caps) if p_caps else 0
            b_volts = b_data['vBat'] if b_data else 0
            b_act = b_data['OutputPower'] > 50 if b_data else False
            
            # Daily accumulation
            current_date = now.strftime('%Y-%m-%d')
            if daily_accumulator['last_date'] != current_date:
                # New day - save yesterday's data
                if daily_accumulator['last_date']:
                    history_manager.update_daily(
                        daily_accumulator['last_date'],
                        daily_accumulator['consumption_wh'],
                        daily_accumulator['solar_wh'],
                        TOTAL_SOLAR_CAPACITY_KW * 1000 * 10  # ~10 hours avg potential
                    )
                    history_manager.save_history()
                # Reset for new day
                daily_accumulator = {'consumption_wh': 0, 'solar_wh': 0, 'last_date': current_date}
            
            # Accumulate (Wh = W * hours, so W * (minutes/60))
            interval_hours = POLL_INTERVAL_MINUTES / 60.0
            daily_accumulator['consumption_wh'] += tot_out * interval_hours
            daily_accumulator['solar_wh'] += tot_sol * interval_hours
            
            detected = identify_active_appliances(tot_out, prev_watts, gen_on, b_volts, p_min)
            is_manual_gen = any("Water" in x for x in detected)
            if not is_manual_gen: load_manager.update(tot_out)
            
            # Add granular datapoint for 24h chart
            history_manager.add_hourly_datapoint(now, tot_out, tot_bat, tot_sol)
            
            if gen_on: send_email("Generator ON", "Generator running", "gen")
            if p_min < 30: send_email("Battery Critical", f"Primary at {p_min}%", "crit")
            
            if (now - last_save) > timedelta(hours=1):
                load_manager.save_data()
                last_save = now

            l_cast = load_manager.get_forecast(24)
            s_cast = generate_solar_forecast(wx_data)
            sim_res = calculate_battery_cascade(s_cast, l_cast, p_min, b_volts, b_act)
            breakdown = calculate_battery_breakdown(p_min, b_volts)
            schedule = generate_smart_schedule(p_min, s_cast, l_cast)
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
                "hourly_24h": hourly_24h
            }
            print(f"Update: Load={tot_out}W")
            
        except Exception as e: print(f"Error: {e}")
        if polling_active:
            for _ in range(POLL_INTERVAL_MINUTES * 60):
                if not polling_active: break
                time.sleep(1)

# ----------------------------
# 5. UI & Routes
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
    d = latest_data
    def _n(k): return float(d.get(k, 0) or 0)
    
    load = _n("total_output_power")
    solar = _n("total_solar_input_W")
    bat_dis = _n("total_battery_discharge_W")
    p_pct = _n("primary_battery_min")
    b_volt = _n("backup_battery_voltage")
    gen_on = d.get("generator_running", False)
    detected = d.get("detected_appliances", [])
    
    breakdown = d.get("energy_breakdown") or {
        "chart_data": [1,0,1], 
        "tier_labels": ['Primary (100-40%)', 'Backup (80-20%)', 'Reserve (40-20%)'],
        "total_pct": 0, 
        "total_kwh": 0,
        "primary_pct": 0,
        "backup_voltage": 0,
        "backup_pct": 0
    }
    sim = d.get("battery_sim") or {"labels": [], "data": []}
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
    
    c_labels = [x['time'].strftime('%H:%M') for x in l_fc] if l_fc else []
    c_load = [x['estimated_load'] for x in l_fc] if l_fc else []
    c_solar = [x['estimated_generation'] for x in s_fc[:len(l_fc)]] if s_fc else []
    alerts = alert_history[:8]
    tier_labels = breakdown.get('tier_labels', ['Primary (100-40%)', 'Backup (80-20%)', 'Reserve (40-20%)'])
    primary_pct = breakdown.get('primary_pct', 0)
    backup_voltage = breakdown.get('backup_voltage', 0)
    backup_pct = breakdown.get('backup_pct', 0)

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solar Monitor Pro</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
    <style>
        :root { 
            --bg: #0a0e27; --bg-secondary: #151b3d; --card: rgba(21, 27, 61, 0.7); 
            --border: rgba(99, 102, 241, 0.2); --border-hover: rgba(99, 102, 241, 0.4);
            --text: #e2e8f0; --text-muted: #94a3b8; --text-dim: #64748b;
            --success: #10b981; --warn: #f59e0b; --crit: #ef4444; --info: #3b82f6;
            --accent: #6366f1; --accent-glow: rgba(99, 102, 241, 0.3);
            --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --gradient-solar: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        }
        
        * { box-sizing: border-box; }
        
        body { 
            background: var(--bg);
            background-image: 
                radial-gradient(ellipse at top, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(245, 158, 11, 0.1) 0%, transparent 50%);
            color: var(--text); 
            font-family: 'Manrope', sans-serif; 
            margin: 0; 
            padding: 20px; 
            min-height: 100vh;
        }
        
        .container { max-width: 1600px; margin: 0 auto; }
        
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
            background: linear-gradient(135deg, var(--accent) 0%, var(--info) 100%);
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
            .col-3 { grid-column: span 3; }
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
            background: var(--gradient-1);
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
        
        /* Tooltip */
        .tooltip {
            position: absolute;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 0.75rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 100;
            backdrop-filter: blur(10px);
        }
        
        .heatmap-cell:hover .tooltip {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>⚡ SOLAR MONITOR PRO</h1>
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
            <div class="col-3 card">
                <div class="card-title">Solar Generation</div>
                <div class="metric-val" style="color:var(--warn)">{{ '%0.f'|format(solar) }}<span style="font-size:1.2rem">W</span></div>
                <div class="metric-unit">Current Input</div>
            </div>
            <div class="col-3 card">
                <div class="card-title">Home Consumption</div>
                <div class="metric-val" style="color:var(--info)">{{ '%0.f'|format(load) }}<span style="font-size:1.2rem">W</span></div>
                <div class="metric-unit">Active Load</div>
            </div>
            <div class="col-3 card">
                <div class="card-title">Battery Status</div>
                <div class="metric-val" style="color:var(--success)">{{ breakdown['total_pct'] }}<span style="font-size:1.2rem">%</span></div>
                <div class="metric-unit">{{ breakdown['total_kwh'] }} kWh Usable</div>
            </div>
            <div class="col-3 card">
                <div class="card-title">Grid Status</div>
                <div class="metric-val" style="color:{{ 'var(--crit)' if gen_on else 'var(--text-dim)' }}">{{ 'ON' if gen_on else 'OFF' }}</div>
                <div class="metric-unit">Generator/Grid</div>
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
                        <div class="tile-status" style="color: {{ s.color }}">{{ s.msg }}</div>
                    </div>
                    {% endfor %}
                </div>
                <div style="margin-top:15px; text-align:right; font-size:0.9rem; color: var(--text-muted)">
                    Simulated Additional Load: <span id="sim-val" style="color: var(--accent); font-weight: 700">0W</span>
                </div>
            </div>

            <!-- BATTERY PROJECTION -->
            <div class="col-8 card">
                <div class="card-title">24-Hour Battery Projection</div>
                <div style="height:280px"><canvas id="simChart"></canvas></div>
            </div>

            <!-- STORAGE BREAKDOWN -->
            <div class="col-4 card">
                <div class="card-title">Storage Breakdown</div>
                <div style="height:200px"><canvas id="pieChart"></canvas></div>
                <div style="text-align:center; margin-top:15px; font-size:1.1rem; color: var(--success); font-weight: 700">
                    {{ breakdown['total_pct'] }}% Charged
                </div>
                <div style="text-align:center; color: var(--text-muted); font-size:0.85rem">
                    {{ breakdown['total_kwh'] }} kWh Available
                </div>
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
                <div class="card-title">Current Activity</div>
                <div style="margin-bottom:20px">
                    {% if detected %}
                        {% for a in detected %}
                        <span class="tag">{{ a }}</span>
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
                        <div class="alert-time">{{ a.timestamp.strftime('%H:%M') }}</div>
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
        const sForecast = {{ s_fc|tojson }};
        const lForecast = {{ l_fc|tojson }};
        const pieData = {{ breakdown['chart_data']|tojson }};
        const tierLabels = {{ tier_labels|tojson }};
        const hourly24h = {{ hourly_24h|tojson }};
        
        // Sim State
        let activeSims = {};
        
        // Chart.js Global Config
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.2)';
        Chart.defaults.font.family = "'Manrope', sans-serif";
        
        // --- 1. Simulation Chart ---
        const ctx = document.getElementById('simChart');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Current Projection',
                        data: baseData,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
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
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
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

        // --- 2. Storage Pie Chart (3 Tiers) ---
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

            // Simplified JS Physics matching Python tiered logic
            const P_TOTAL = 30000;
            const B_TOTAL = 21000 * 0.7;
            const TOTAL_CAPACITY = (P_TOTAL * 0.60) + (B_TOTAL * 0.80) + (P_TOTAL * 0.20);
            
            let currentPct = baseData[0] || 0;
            let currentWh = (currentPct / 100) * TOTAL_CAPACITY;
            let simCurve = [];
            
            for(let i = 0; i < labels.length; i++) {
                let baseL = (lForecast[i] ? lForecast[i].estimated_load : 1000);
                let sol = (sForecast[i] ? sForecast[i].estimated_generation : 0);
                let net = sol - (baseL + totalSimWatts);
                
                if(net > 0) {
                    currentWh = Math.min(TOTAL_CAPACITY, currentWh + net);
                } else {
                    currentWh = Math.max(0, currentWh - Math.abs(net));
                }
                
                simCurve.push((currentWh / TOTAL_CAPACITY) * 100);
            }
            
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
        backup_voltage=backup_voltage, backup_pct=backup_pct
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
