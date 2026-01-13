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

# Inverter Mapping
INVERTER_CONFIG = {
    "RKG3B0400T": {"label": "Inverter 1", "type": "primary", "display_order": 1},
    "KAM4N5W0AG": {"label": "Inverter 2", "type": "primary", "display_order": 2},
    "JNK1CDR0KQ": {"label": "Inverter 3 (Backup)", "type": "backup", "display_order": 3}
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

load_manager = PersistentLoadManager(DATA_FILE)

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
# 2. Physics & Scheduler Engine (ROBUST VERSION)
# ----------------------------

# Define Heavy Loads with DURATION
APPLIANCE_PROFILES = [
    {"name": "Pool Pump", "watts": 1200, "hours": 4, "icon": "🏊"},
    {"name": "Washing Machine", "watts": 800, "hours": 1.5, "icon": "🧺"},
    {"name": "Dishwasher", "watts": 1500, "hours": 2, "icon": "🍽️"},
    {"name": "Oven/Cooking", "watts": 2500, "hours": 1.5, "icon": "🍳"}
]

def simulate_run_impact(start_index, duration_hours, appliance_watts, current_p_wh, s_forecast, l_forecast):
    """
    Simulates the battery state if we add an appliance load on top of base load.
    Returns the MINIMUM battery percentage reached during the run.
    """
    p_cap = PRIMARY_BATTERY_CAPACITY_WH
    running_p = current_p_wh
    min_pct_encountered = 100
    
    # We need forecasts for the duration of the run
    # Ensure lists are long enough
    if len(s_forecast) < (start_index + duration_hours + 1) or len(l_forecast) < (start_index + duration_hours + 1):
        return 0 # Fail safe: not enough data
        
    # Simulate hour by hour
    for i in range(int(duration_hours) + 1): # +1 to cover partial hours
        idx = start_index + i
        if idx >= len(s_forecast): break
        
        # Base Scenario
        solar_gen = s_forecast[idx]['estimated_generation']
        base_load = l_forecast[idx]['estimated_load']
        
        # Add Appliance Load?
        # If it's the last hour and duration is 1.5, we only add half load
        load_factor = 1.0
        if i == int(duration_hours): 
            load_factor = duration_hours % 1
            if load_factor == 0: continue # Loop finished
            
        total_load = base_load + (appliance_watts * load_factor)
        
        # Physics Step
        net = solar_gen - total_load
        
        if net > 0:
            running_p = min(p_cap, running_p + net)
        else:
            running_p = max(0, running_p - abs(net))
            
        current_pct = (running_p / p_cap) * 100
        if current_pct < min_pct_encountered:
            min_pct_encountered = current_pct
            
    return min_pct_encountered

def generate_smart_schedule(p_pct, s_forecast, l_forecast):
    """
    Robust scheduler that runs a forward simulation for every appliance.
    """
    advice = []
    current_p_wh = (p_pct / 100) * PRIMARY_BATTERY_CAPACITY_WH
    
    # Determine safe threshold (Buffer above 40% cutoff)
    SAFE_THRESHOLD = 45.0 
    
    for app in APPLIANCE_PROFILES:
        watts = app['watts']
        hours = app['hours']
        
        # 1. Simulate running NOW (Index 0)
        min_pct_now = simulate_run_impact(0, hours, watts, current_p_wh, s_forecast, l_forecast)
        
        decision = {}
        
        if min_pct_now >= SAFE_THRESHOLD:
            # It's safe, but is it "Free"? (Ending state higher than starting?)
            if min_pct_now > p_pct:
                 decision = {"status": "now", "msg": "Run Now (Free Solar)", "color": "var(--success)"}
            elif min_pct_now > 60:
                 decision = {"status": "ok", "msg": "OK (Safe Buffer)", "color": "var(--info)"}
            else:
                 decision = {"status": "ok", "msg": "OK (Bat Drains)", "color": "var(--warn)"}
        else:
            # UNSAFE NOW. Find best future time.
            best_hour = -1
            best_min_pct = -1
            
            # Check next 12 hours
            for h in range(1, 12):
                # We need to estimate battery state at hour h BEFORE starting simulation
                # (Simple linear projection for starting point of future simulation)
                # This is complex, so we simplify: Find time of max solar
                pass 
            
            # Simplified "Find Best Time": Look for highest solar generation block
            best_start_time = "Tomorrow"
            max_solar_avg = 0
            
            for h in range(12): # Look 12 hours ahead
                # Sum solar for duration
                solar_sum = 0
                for d in range(int(hours) + 1):
                    if (h+d) < len(s_forecast):
                        solar_sum += s_forecast[h+d]['estimated_generation']
                
                if solar_sum > max_solar_avg:
                    max_solar_avg = solar_sum
                    if s_forecast[h]['estimated_generation'] > (watts * 0.8): # Ensure start time has solar
                        best_start_time = s_forecast[h]['time'].strftime("%I%p").lstrip('0')

            decision = {"status": "stop", "msg": f"Unsafe! Wait for {best_start_time}", "color": "var(--crit)"}
            
            # Special Case: If battery is critically full (>95), allow it even if solar drops later
            if p_pct > 95:
                 decision = {"status": "now", "msg": "Run Now (Dump Energy)", "color": "var(--success)"}

        advice.append({**app, **decision})
        
    return advice

def calculate_battery_breakdown(p_pct, b_volts):
    b_pct = max(0, min(100, (b_volts - 51.0) / 2.0 * 100))
    p_cap = PRIMARY_BATTERY_CAPACITY_WH
    b_cap = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION
    curr_p = (p_pct / 100) * p_cap
    curr_b = (b_pct / 100) * b_cap
    p_avail = max(0, curr_p - (p_cap * 0.40))
    b_avail = max(0, curr_b - (b_cap * 0.20))
    reserve = (p_cap * 0.40) + (b_cap * 0.20)
    total_usable_cap = (p_cap * 0.60) + (b_cap * 0.80)
    total_current_usable = p_avail + b_avail
    total_pct = (total_current_usable / total_usable_cap * 100) if total_usable_cap > 0 else 0
    return {'chart_data': [round(p_avail/1000, 1), round(b_avail/1000, 1), round(reserve/1000, 1)], 'total_pct': round(total_pct, 1), 'total_kwh': round(total_current_usable/1000, 1)}

def calculate_battery_cascade(solar, load, p_pct, b_active=False):
    if not solar or not load: return {'labels': [], 'data': []}
    p_wh = PRIMARY_BATTERY_CAPACITY_WH
    b_wh = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION
    curr_p = (p_pct / 100.0) * p_wh
    start_b_pct = 100 if b_active else 0
    curr_b = (start_b_pct / 100.0) * b_wh 
    sim_data, sim_labels = [], []
    run_p, run_b = curr_p, curr_b
    total_sys_wh = p_wh + b_wh
    for i in range(min(len(solar), len(load))):
        net = solar[i]['estimated_generation'] - load[i]['estimated_load']
        if net > 0:
            space_p = p_wh - run_p
            if net <= space_p: run_p += net
            else:
                run_p = p_wh
                run_b = min(b_wh, run_b + (net - space_p))
        else:
            drain = abs(net)
            avail_p = max(0, run_p - (p_wh * 0.40))
            if avail_p >= drain: run_p -= drain
            else:
                run_p = max(run_p - avail_p, p_wh * 0.40)
                run_b = max(0, run_b - (drain - avail_p))
        sim_data.append(((run_p + run_b) / total_sys_wh) * 100)
        sim_labels.append(solar[i]['time'].strftime('%H:%M'))
    return {'labels': sim_labels, 'data': sim_data}

# ----------------------------
# 3. Helpers
# ----------------------------
headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"} if TOKEN else {}
last_alert_time, alert_history = {}, []
latest_data = {
    "timestamp": "Initializing...", "total_output_power": 0, "total_solar_input_W": 0,
    "primary_battery_min": 0, "backup_battery_voltage": 0, "backup_active": False,
    "generator_running": False, "inverters": [], "detected_appliances": [], 
    "solar_forecast": [], "load_forecast": [], 
    "battery_sim": {"labels": [], "data": []},
    "energy_breakdown": {"chart_data": [1, 1, 1], "total_pct": 0, "total_kwh": 0},
    "scheduler": []
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
    global latest_data, polling_active
    if not TOKEN: return

    wx_data = get_weather_forecast()
    prev_watts = 0 
    last_save = datetime.now(EAT)
    polling_active = True
    
    print("🚀 System Started: Robust Physics Mode")

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
            
            detected = identify_active_appliances(tot_out, prev_watts, gen_on, b_volts, p_min)
            is_manual_gen = any("Water" in x for x in detected)
            if not is_manual_gen: load_manager.update(tot_out)
            
            if gen_on: send_email("Generator ON", "Generator running", "gen")
            if p_min < 30: send_email("Battery Critical", f"Primary at {p_min}%", "crit")
            
            if (now - last_save) > timedelta(hours=1):
                load_manager.save_data()
                last_save = now

            l_cast = load_manager.get_forecast(24)
            s_cast = generate_solar_forecast(wx_data)
            sim_res = calculate_battery_cascade(s_cast, l_cast, p_min, b_act)
            breakdown = calculate_battery_breakdown(p_min, b_volts)
            
            # --- Robust Scheduler (Physics Based) ---
            schedule = generate_smart_schedule(p_min, s_cast, l_cast)
            
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
                "inverters": inv_data
            }
            print(f"Update: Load={tot_out}W | Bat={p_min}%")
            
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
    
    # Defaults
    breakdown = d.get("energy_breakdown") or {"chart_data": [1,0,0], "total_pct": 0, "total_kwh": 0}
    sim = d.get("battery_sim") or {"labels": [], "data": []}
    s_fc = d.get("solar_forecast") or []
    l_fc = d.get("load_forecast") or []
    schedule = d.get("scheduler") or []
    
    st_txt, st_col = "NORMAL", "var(--info)"
    if gen_on:
        if "Water" in str(detected): st_txt, st_col = "WATER HEATING", "var(--warn)"
        else: st_txt, st_col = "GENERATOR ON", "var(--crit)"
    elif p_pct < 40: st_txt, st_col = "BACKUP ACTIVE", "var(--warn)"
    elif solar > load + 500: st_txt, st_col = "CHARGING", "var(--success)"

    is_charging = solar > (load + 100)
    is_discharging = bat_dis > 100 or load > solar
    c_labels = [x['time'].strftime('%H:%M') for x in l_fc] if l_fc else []
    c_load = [x['estimated_load'] for x in l_fc] if l_fc else []
    c_solar = [x['estimated_generation'] for x in s_fc[:len(l_fc)]] if s_fc else []
    alerts = alert_history[:8]

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solar Monitor</title>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root { 
            --bg: #000000; --card: rgba(25, 25, 25, 0.6); --border: rgba(255, 255, 255, 0.1); 
            --text: #ffffff; --text-muted: #888;
            --success: #3fb950; --warn: #ffa500; --crit: #ff0000; --info: #00bfff;
        }
        body { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; margin: 0; padding: 15px; }
        .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 15px; max-width: 1400px; margin: 0 auto; }
        .col-12 { grid-column: span 12; } .col-6 { grid-column: span 12; } .col-4 { grid-column: span 12; } .col-3 { grid-column: span 6; }
        @media(min-width:768px){ .col-6 { grid-column: span 6; } .col-4 { grid-column: span 4; } .col-3 { grid-column: span 3; } }
        
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 20px; padding: 20px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); backdrop-filter: blur(10px); }
        h2 { font-size: 0.9rem; text-transform: uppercase; color: var(--text-muted); margin: 0 0 15px 0; letter-spacing: 1px; }
        .metric-val { font-family: 'Space Mono'; font-size: 1.8rem; font-weight: 700; }
        .metric-unit { font-size: 0.9rem; color: var(--text-muted); }
        .tag { padding: 5px 12px; border-radius: 50px; font-size: 0.8rem; font-weight: 700; background: rgba(255,255,255,0.1); border: 1px solid var(--border); display: inline-flex; align-items: center; gap: 6px; margin-right: 5px; }
        
        .flow-diagram { position: relative; height: 300px; width: 100%; display: flex; justify-content: center; align-items: center; }
        .node { position: absolute; width: 80px; height: 80px; border-radius: 50%; background: #111; border: 2px solid #333; z-index: 2; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: 0 0 20px rgba(0,0,0,0.5); transition: all 0.3s ease; }
        .node-val { font-family: 'Space Mono'; font-size: 11px; font-weight: bold; }
        .n-solar { top: 10px; left: 50%; transform: translateX(-50%); border-color: var(--warn); }
        .n-inv   { top: 50%; left: 50%; transform: translate(-50%, -50%); width: 100px; height: 100px; border-color: var(--info); }
        .n-home  { top: 50%; right: 10%; transform: translateY(-50%); border-color: var(--info); }
        .n-bat   { bottom: 10px; left: 50%; transform: translateX(-50%); border-color: var(--success); }
        .n-gen   { top: 50%; left: 10%; transform: translateY(-50%); border-color: var(--crit); }
        
        .line { position: absolute; background: #333; z-index: 1; overflow: hidden; }
        .line-v { width: 4px; height: 80px; left: 50%; transform: translateX(-50%); }
        .l-solar { top: 90px; } .l-bat { bottom: 90px; }
        .line-h { height: 4px; width: 25%; top: 50%; transform: translateY(-50%); }
        .l-gen { left: 18%; } .l-home { right: 18%; }
        
        .dot { position: absolute; background: #fff; border-radius: 50%; width: 6px; height: 6px; box-shadow: 0 0 10px #fff; opacity: 0; }
        .flow-down .dot { left: -1px; animation: flowY 1.5s linear infinite; opacity: 1; }
        .flow-up .dot { left: -1px; animation: flowY-rev 1.5s linear infinite; opacity: 1; }
        .flow-right .dot { top: -1px; animation: flowX 1.5s linear infinite; opacity: 1; }
        @keyframes flowY { 0%{top:0%} 100%{top:100%} } @keyframes flowY-rev { 0%{top:100%} 100%{top:0%} } @keyframes flowX { 0%{left:0%} 100%{left:100%} }
        .pulse-g { animation: p-g 2s infinite; } .pulse-r { animation: p-r 2s infinite; } .pulse-y { animation: p-y 2s infinite; }
        @keyframes p-g { 0%{box-shadow:0 0 0 0 rgba(0,255,0,0.4)} 70%{box-shadow:0 0 0 15px rgba(0,255,0,0)} 100%{box-shadow:0 0 0 0 rgba(0,255,0,0)} }
        @keyframes p-r { 0%{box-shadow:0 0 0 0 rgba(255,0,0,0.4)} 70%{box-shadow:0 0 0 15px rgba(255,0,0,0)} 100%{box-shadow:0 0 0 0 rgba(255,0,0,0)} }
        @keyframes p-y { 0%{box-shadow:0 0 0 0 rgba(255,165,0,0.4)} 70%{box-shadow:0 0 0 15px rgba(255,165,0,0)} 100%{box-shadow:0 0 0 0 rgba(255,165,0,0)} }

        /* Scheduler */
        .sched-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
        .sched-card { background: rgba(255,255,255,0.05); padding: 10px; border-radius: 12px; text-align: center; border: 1px solid transparent; }
        .sched-stat { font-size: 0.8rem; font-weight: bold; margin-top: 5px; }
        
        .alert-row { display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding: 8px 0; font-size: 0.9rem; }
    </style>
</head>
<body>
    <div class="grid">
        <div class="col-12" style="display:flex; justify-content:space-between; align-items:center;">
            <div><h1 style="margin:0; font-size:1.5rem">SOLAR MONITOR</h1><div style="color:var(--text-muted); font-size:0.9rem">Status: <span style="color:{{ st_col }}">{{ st_txt }}</span></div></div>
            <div style="font-family:'Space Mono'; font-size:1.2rem">{{ d.timestamp.split(' ')[1] }}</div>
        </div>
        
        <!-- Power Flow -->
        <div class="col-12 card" style="padding:0">
            <div class="flow-diagram">
                <div class="line line-v l-solar {{ 'flow-down' if solar > 50 else '' }}"><div class="dot"></div></div>
                <div class="line line-v l-bat {{ 'flow-down' if is_charging else ('flow-up' if is_discharging else '') }}"><div class="dot"></div></div>
                <div class="line line-h l-home {{ 'flow-right' if load > 100 else '' }}"><div class="dot"></div></div>
                <div class="line line-h l-gen {{ 'flow-right' if gen_on else '' }}"><div class="dot"></div></div>
                
                <div class="node n-solar {{ 'pulse-y' if solar > 50 else '' }}"><div>☀️</div><div class="node-val">{{ '%0.f'|format(solar) }}W</div></div>
                <div class="node n-gen {{ 'pulse-r' if gen_on else '' }}"><div>⚙️</div><div class="node-val">{{ 'ON' if gen_on else 'OFF' }}</div></div>
                <div class="node n-inv"><div>⚡</div><div class="node-val">INV</div></div>
                <div class="node n-home {{ 'pulse-g' if load > 2000 else '' }}"><div>🏠</div><div class="node-val">{{ '%0.f'|format(load) }}W</div></div>
                <div class="node n-bat {{ 'pulse-g' if is_charging else ('pulse-r' if is_discharging else '') }}"><div>🔋</div><div class="node-val">{{ breakdown.total_pct }}%</div></div>
            </div>
        </div>

        <!-- Metrics -->
        <div class="col-3 card"><h2>Solar</h2><div class="metric-val" style="color:var(--warn)">{{ '%0.f'|format(solar) }}</div><div class="metric-unit">Watts</div></div>
        <div class="col-3 card"><h2>Load</h2><div class="metric-val" style="color:var(--info)">{{ '%0.f'|format(load) }}</div><div class="metric-unit">Watts</div></div>
        <div class="col-3 card"><h2>Primary</h2><div class="metric-val" style="color:var(--success)">{{ '%0.f'|format(p_pct) }}</div><div class="metric-unit">%</div></div>
        <div class="col-3 card"><h2>Backup</h2><div class="metric-val">{{ '%0.1f'|format(b_volt) }}</div><div class="metric-unit">V</div></div>

        <!-- NEW: Scheduler -->
        <div class="col-12 card">
            <h2>⚡ Smart Schedule</h2>
            <div class="sched-grid">
                {% for s in schedule %}
                <div class="sched-card" style="border-color: {{ s.color }}">
                    <div style="font-size:1.5rem">{{ s.icon }}</div>
                    <div style="font-weight:bold">{{ s.name }}</div>
                    <div class="sched-stat" style="color: {{ s.color }}">{{ s.msg }}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Charts -->
        <div class="col-4 card"><h2>Storage</h2><div style="height:200px"><canvas id="c1"></canvas></div><center style="margin-top:10px">{{ breakdown.total_pct }}% Usable</center></div>
        <div class="col-4 card"><h2>Forecast</h2><div style="height:200px"><canvas id="c2"></canvas></div></div>
        <div class="col-4 card"><h2>Prediction</h2><div style="height:200px"><canvas id="c3"></canvas></div></div>
        
        <!-- Detected -->
        <div class="col-6 card"><h2>Activity</h2>{% if detected %}{% for a in detected %}<div class="tag">{{ a }}</div>{% endfor %}{% else %}<div style="opacity:0.5">Idle</div>{% endif %}</div>
        <div class="col-6 card"><h2>Alerts</h2>{% for a in alerts %}<div class="alert-row"><div style="color:{{ 'var(--crit)' if 'crit' in a.type else 'var(--text)' }}">{{ a.subject }}</div><div class="alert-time">{{ a.timestamp.strftime('%H:%M') }}</div></div>{% endfor %}</div>
    </div>

    <script>
        const common = { responsive: true, maintainAspectRatio: false };
        new Chart(document.getElementById('c1'), { type: 'doughnut', data: { labels: ['Primary', 'Backup', 'Reserve'], datasets: [{ data: {{ breakdown.chart_data|tojson }}, backgroundColor: ['#3fb950', '#ffa500', '#333'], borderWidth: 0 }] }, options: { ...common, cutout: '70%', plugins: { legend: { display: false } } } });
        new Chart(document.getElementById('c2'), { type: 'line', data: { labels: {{ c_labels|tojson }}, datasets: [{ label: 'Load', data: {{ c_load|tojson }}, borderColor: '#00bfff' }, { label: 'Solar', data: {{ c_solar|tojson }}, borderColor: '#ffa500' }] }, options: common });
        new Chart(document.getElementById('c3'), { type: 'line', data: { labels: {{ sim.labels|tojson }}, datasets: [{ label: 'Bat %', data: {{ sim.data|tojson }}, borderColor: '#00ff00', fill:true }] }, options: { ...common, scales: { x:{display:false}, y:{min:0, max:100} } } });
        
        fetch('/health').then(r=>r.json()).then(d=>{ if(!d.polling_thread_alive) fetch('/start-polling'); });
        setTimeout(()=>location.reload(), 60000);
    </script>
</body>
</html>
    """
    return render_template_string(html, 
        d=d, solar=solar, load=load, p_pct=p_pct, b_volt=b_volt, 
        gen_on=gen_on, detected=detected, st_txt=st_txt, st_col=st_col,
        is_charging=is_charging, is_discharging=is_discharging,
        c_labels=c_labels, c_load=c_load, c_solar=c_solar, sim=sim, alerts=alerts,
        breakdown=breakdown, schedule=schedule
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
