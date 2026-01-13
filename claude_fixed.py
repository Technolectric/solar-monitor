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
    """Tracks historical usage to predict future battery drain."""
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
        # Keep last 100 readings per hour slot (approx 3 months history)
        if len(self.patterns[day_type][hour]) > 100:
            self.patterns[day_type][hour] = self.patterns[day_type][hour][-100:]
            
    def get_forecast(self, hours_ahead=12):
        forecast = []
        now = datetime.now(EAT)
        for i in range(hours_ahead):
            ft = now + timedelta(hours=i)
            day_type = "weekend" if ft.weekday() >= 5 else "weekday"
            hour = str(ft.hour)
            history = self.patterns[day_type][hour]
            
            # Default Profile if no history
            est = sum(history) / len(history) if history else (2500 if 18<=ft.hour<=21 else 1000)
            forecast.append({'time': ft, 'estimated_load': est})
        return forecast

load_manager = PersistentLoadManager(DATA_FILE)

def identify_active_appliances(current, previous, gen_active, backup_volts, primary_pct):
    """Detects active appliances and differentiates manual vs auto generator use."""
    detected = []
    delta = current - previous

    # --- Generator Logic ---
    if gen_active:
        # If Primary is healthy (>42%), Gen usage is Manual (Water Heating)
        if primary_pct > 42:
            detected.append("Water Heating") # Manual
        else:
            detected.append("System Charging") # Auto/Emergency
            
    # --- Inverter Load Logic ---
    if current < 400: detected.append("Idle")
    elif 1000 <= current <= 1350: detected.append("Pool Pump")
    elif current > 1800: detected.append("Cooking")
    elif 400 <= current < 1000: detected.append("TV/Lights")

    if delta > 1500: detected.append("Kettle")
    return detected

# ----------------------------
# 2. Physics Engine: Battery Simulation
# ----------------------------
def calculate_battery_cascade(solar, load, p_pct, b_active=False):
    """Simulates battery capacity over the next 24 hours."""
    if not solar or not load: return None
    
    # Capacities (Wh)
    p_wh = PRIMARY_BATTERY_CAPACITY_WH
    b_wh = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION
    total_sys_wh = p_wh + b_wh
    
    # Current State (Wh)
    curr_p = (p_pct / 100.0) * p_wh
    # Estimate Backup % from Voltage (Rough Calc) or assume 100 if active/charging
    start_b_pct = 100 if b_active else 0
    curr_b = (start_b_pct / 100.0) * b_wh 
    
    sim_data = []
    sim_labels = []
    
    run_p, run_b = curr_p, curr_b
    
    for i in range(min(len(solar), len(load))):
        gen = solar[i]['estimated_generation']
        dem = load[i]['estimated_load']
        net = gen - dem
        
        if net > 0: # Charging
            space_p = p_wh - run_p
            if net <= space_p: run_p += net
            else:
                run_p = p_wh
                run_b = min(b_wh, run_b + (net - space_p))
        else: # Discharging
            drain = abs(net)
            # Primary drains first until 40% (12000 Wh)
            avail_p = max(0, run_p - (p_wh * 0.40))
            if avail_p >= drain:
                run_p -= drain
            else:
                run_p = max(run_p - avail_p, p_wh * 0.40)
                run_b = max(0, run_b - (drain - avail_p))
                
        total_pct = ((run_p + run_b) / total_sys_wh) * 100
        sim_data.append(total_pct)
        sim_labels.append(solar[i]['time'].strftime('%H:%M'))
        
    return {'labels': sim_labels, 'data': sim_data}

# ----------------------------
# 3. Helpers (Weather, Email, Utils)
# ----------------------------
headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"} if TOKEN else {}
last_alert_time, alert_history = {}, []
latest_data = {
    "timestamp": "Initializing...", "total_output_power": 0, "total_solar_input_W": 0,
    "primary_battery_min": 0, "backup_battery_voltage": 0, "backup_active": False,
    "generator_running": False, "inverters": [], "detected_appliances": [], 
    "solar_forecast": [], "load_forecast": [], "battery_sim": None,
    "usable_energy": {"total_pct": 0, "total_kwh": 0}
}
load_history = [] 

def get_weather_forecast():
    """Robust weather fetcher with multiple sources."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=shortwave_radiation&timezone=Africa/Nairobi&forecast_days=2"
        r = requests.get(url, timeout=5).json()
        return {'times': r['hourly']['time'], 'rad': r['hourly']['shortwave_radiation']}
    except: return None # Add fallbacks here if preferred

def generate_solar_forecast(weather_data):
    forecast = []
    now = datetime.now(EAT)
    if not weather_data: return forecast
    
    # Map weather to hours
    w_map = {t: r for t, r in zip(weather_data['times'], weather_data['rad'])}
    
    for i in range(24):
        ft = now + timedelta(hours=i)
        key = ft.strftime('%Y-%m-%dT%H:00')
        rad = w_map.get(key, 0)
        # Rad (W/m2) * Capacity (m2 approx or kW conversion) * Efficiency
        est = (rad / 1000.0) * (TOTAL_SOLAR_CAPACITY_KW * 1000) * SOLAR_EFFICIENCY_FACTOR
        forecast.append({'time': ft, 'estimated_generation': est})
    return forecast

def calculate_usable_energy(primary_pct, backup_volts):
    # Estimate Backup % from Voltage (Linear approx between 51V and 53V)
    b_pct = max(0, min(100, (backup_volts - 51.0) / 2.0 * 100))
    
    p_kwh = max(0, ((primary_pct - 40) / 100) * (PRIMARY_BATTERY_CAPACITY_WH/1000))
    b_kwh = max(0, ((b_pct - 20) / 100) * (BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION / 1000))
    
    total_cap = (PRIMARY_BATTERY_CAPACITY_WH * 0.6 + BACKUP_BATTERY_DEGRADED_WH * 0.7 * 0.8) / 1000
    total = p_kwh + b_kwh
    pct = (total / total_cap) * 100 if total_cap > 0 else 0
    return {'total_kwh': round(total, 1), 'total_pct': round(pct, 1)}

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
    global latest_data, load_history, polling_active
    if not TOKEN: return

    wx_data = get_weather_forecast()
    prev_watts = 0 
    last_save = datetime.now(EAT)
    polling_active = True
    
    print("🚀 System Started: Tesla Dashboard Mode")

    while polling_active:
        try:
            now = datetime.now(EAT)
            tot_out, tot_sol, tot_bat = 0, 0, 0
            inv_data, p_caps = [], []
            b_data, gen_on = None, False
            
            # 1. Fetch Data
            for sn in SERIAL_NUMBERS:
                try:
                    r = requests.post(API_URL, data={"storage_sn": sn}, headers=headers, timeout=20)
                    if r.json().get("error_code") == 0:
                        d = r.json().get("data", {})
                        
                        op = float(d.get("outPutPower") or 0)
                        cap = float(d.get("capacity") or 0)
                        vb = float(d.get("vBat") or 0)
                        pb = float(d.get("pBat") or 0) # Discharge power
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

            # 2. Logic & Detection
            p_min = min(p_caps) if p_caps else 0
            b_volts = b_data['vBat'] if b_data else 0
            b_act = b_data['OutputPower'] > 50 if b_data else False
            
            detected = identify_active_appliances(tot_out, prev_watts, gen_on, b_volts, p_min)
            
            # 3. Persistence & Alerts
            is_manual_gen = any("Water" in x for x in detected)
            if not is_manual_gen: load_manager.update(tot_out)
            
            if gen_on: send_email("Generator ON", "Generator running", "gen")
            if p_min < 30: send_email("Battery Critical", f"Primary at {p_min}%", "crit")
            
            if (now - last_save) > timedelta(hours=1):
                load_manager.save_data()
                last_save = now

            # 4. Forecasts
            l_cast = load_manager.get_forecast(24)
            s_cast = generate_solar_forecast(wx_data)
            sim_res = calculate_battery_cascade(s_cast, l_cast, p_min, b_act)
            usable = calculate_usable_energy(p_min, b_volts)
            
            load_history.append((now, tot_out))
            load_history = load_history[-20:]
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
                "usable_energy": usable,
                "inverters": inv_data
            }
            
            print(f"Poll: Load={tot_out}W | Gen={gen_on} | Apps={detected}")
            
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
    # Safe Unpack
    def _n(k): return float(d.get(k, 0) or 0)
    
    load = _n("total_output_power")
    solar = _n("total_solar_input_W")
    bat_dis = _n("total_battery_discharge_W")
    p_pct = _n("primary_battery_min")
    b_volt = _n("backup_battery_voltage")
    gen_on = d.get("generator_running", False)
    detected = d.get("detected_appliances", [])
    
    # --- Status Logic ---
    st_txt, st_col = "NORMAL", "var(--info)"
    if gen_on:
        if "Water" in str(detected): st_txt, st_col = "WATER HEATING", "var(--warn)"
        else: st_txt, st_col = "GENERATOR ON", "var(--crit)"
    elif p_pct < 40: st_txt, st_col = "BACKUP ACTIVE", "var(--warn)"
    elif solar > load + 500: st_txt, st_col = "CHARGING", "var(--success)"

    # --- Animation Booleans ---
    is_charging = solar > (load + 100)
    is_discharging = bat_dis > 100 or load > solar
    is_solar = solar > 50
    
    # --- Chart Data ---
    s_fc = d.get("solar_forecast", [])
    l_fc = d.get("load_forecast", [])
    sim = d.get("battery_sim", {"labels": [], "data": []})
    
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
            --success: #00ff00; --warn: #ffa500; --crit: #ff0000; --info: #00bfff;
        }
        body { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; margin: 0; padding: 15px; }
        
        .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 15px; max-width: 1400px; margin: 0 auto; }
        .col-12 { grid-column: span 12; } .col-6 { grid-column: span 12; } .col-3 { grid-column: span 6; }
        @media(min-width:768px){ .col-6 { grid-column: span 6; } .col-3 { grid-column: span 3; } }
        
        /* Glassmorphism Card */
        .card { 
            background: var(--card); border: 1px solid var(--border); 
            backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
            border-radius: 20px; padding: 20px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        }
        
        h2 { font-size: 0.9rem; text-transform: uppercase; color: var(--text-muted); margin: 0 0 15px 0; letter-spacing: 1px; }
        
        /* Metric Styling */
        .metric-val { font-family: 'Space Mono'; font-size: 1.8rem; font-weight: 700; }
        .metric-unit { font-size: 0.9rem; color: var(--text-muted); }
        
        /* Tags */
        .tag { 
            padding: 5px 12px; border-radius: 50px; font-size: 0.8rem; font-weight: 700; 
            background: rgba(255,255,255,0.1); border: 1px solid var(--border); display: inline-flex; align-items: center; gap: 6px; 
            margin-right: 5px;
        }
        
        /* --- POWER FLOW DIAGRAM (Tesla Style) --- */
        .flow-diagram {
            position: relative; height: 350px; width: 100%;
            display: flex; justify-content: center; align-items: center;
        }
        
        /* Nodes */
        .node {
            position: absolute; width: 90px; height: 90px; border-radius: 50%;
            background: #111; border: 2px solid #333; z-index: 2;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            box-shadow: 0 0 20px rgba(0,0,0,0.5); transition: all 0.3s ease;
        }
        .node-icon { font-size: 24px; margin-bottom: 4px; }
        .node-val { font-family: 'Space Mono'; font-size: 12px; font-weight: bold; }
        
        /* Positioning */
        .n-solar { top: 10px; left: 50%; transform: translateX(-50%); border-color: var(--warn); }
        .n-inv   { top: 50%; left: 50%; transform: translate(-50%, -50%); width: 110px; height: 110px; border-color: var(--info); }
        .n-home  { top: 50%; right: 10%; transform: translateY(-50%); border-color: var(--info); }
        .n-bat   { bottom: 10px; left: 50%; transform: translateX(-50%); border-color: var(--success); }
        .n-gen   { top: 50%; left: 10%; transform: translateY(-50%); border-color: var(--crit); }
        
        /* Throbbing Animations */
        .pulse-solar { animation: pulse-yellow 2s infinite; }
        .pulse-green { animation: pulse-green 2s infinite; }
        .pulse-red   { animation: pulse-red 2s infinite; }
        
        @keyframes pulse-yellow { 0%{box-shadow: 0 0 0 0 rgba(255, 165, 0, 0.4)} 70%{box-shadow: 0 0 0 15px rgba(255, 165, 0, 0)} 100%{box-shadow: 0 0 0 0 rgba(255, 165, 0, 0)} }
        @keyframes pulse-green  { 0%{box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.4)} 70%{box-shadow: 0 0 0 15px rgba(0, 255, 0, 0)} 100%{box-shadow: 0 0 0 0 rgba(0, 255, 0, 0)} }
        @keyframes pulse-red    { 0%{box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4)} 70%{box-shadow: 0 0 0 15px rgba(255, 0, 0, 0)} 100%{box-shadow: 0 0 0 0 rgba(255, 0, 0, 0)} }

        /* Lines */
        .line {
            position: absolute; background: #333; z-index: 1; overflow: hidden;
        }
        /* Vertical Lines (Solar->Inv, Inv->Bat) */
        .line-v { width: 4px; height: 80px; left: 50%; transform: translateX(-50%); }
        .l-solar { top: 100px; }
        .l-bat   { bottom: 100px; }
        
        /* Horizontal Lines (Gen->Inv, Inv->Home) */
        .line-h { height: 4px; width: 25%; top: 50%; transform: translateY(-50%); }
        .l-gen  { left: 18%; }
        .l-home { right: 18%; }
        
        /* Moving Dots Animation */
        .dot {
            position: absolute; background: #fff; border-radius: 50%; width: 6px; height: 6px;
            box-shadow: 0 0 10px #fff; opacity: 0;
        }
        
        /* Flow Down (Solar -> Inv) */
        .flow-down .dot { left: -1px; animation: flowY 1.5s linear infinite; opacity: 1; }
        /* Flow Up (Bat -> Inv) */
        .flow-up .dot { left: -1px; animation: flowY-rev 1.5s linear infinite; opacity: 1; }
        /* Flow Right (Inv -> Home) */
        .flow-right .dot { top: -1px; animation: flowX 1.5s linear infinite; opacity: 1; }
        
        @keyframes flowY { 0%{top:0%} 100%{top:100%} }
        @keyframes flowY-rev { 0%{top:100%} 100%{top:0%} }
        @keyframes flowX { 0%{left:0%} 100%{left:100%} }
        
        /* Alerts Table */
        .alert-row { display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); padding: 8px 0; font-size: 0.9rem; }
        .alert-time { color: var(--text-muted); font-family: 'Space Mono'; }
        
    </style>
</head>
<body>

    <div class="grid">
        <!-- Header -->
        <div class="col-12" style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h1 style="margin:0; font-size:1.5rem">SOLAR MONITOR</h1>
                <div style="color:var(--text-muted); font-size:0.9rem">Status: <span style="color:{{ st_col }}">{{ st_txt }}</span></div>
            </div>
            <div style="text-align:right">
                <div style="font-family:'Space Mono'; font-size:1.2rem">{{ d.timestamp.split(' ')[1] }}</div>
            </div>
        </div>
        
        <!-- VISUAL DIAGRAM -->
        <div class="col-12 card" style="padding:0">
            <div class="flow-diagram">
                <!-- Lines & Animations -->
                <!-- Solar Line -->
                <div class="line line-v l-solar {{ 'flow-down' if solar > 50 else '' }}">
                    <div class="dot"></div><div class="dot" style="animation-delay:0.7s"></div>
                </div>
                <!-- Battery Line -->
                <div class="line line-v l-bat {{ 'flow-down' if is_charging else ('flow-up' if is_discharging else '') }}">
                     <div class="dot"></div>
                </div>
                <!-- Home Line -->
                <div class="line line-h l-home {{ 'flow-right' if load > 100 else '' }}">
                    <div class="dot"></div><div class="dot" style="animation-delay:0.5s"></div>
                </div>
                <!-- Gen Line -->
                <div class="line line-h l-gen {{ 'flow-right' if gen_on else '' }}">
                    <div class="dot"></div>
                </div>
                
                <!-- Nodes -->
                <div class="node n-solar {{ 'pulse-solar' if solar > 50 else '' }}">
                    <div class="node-icon">☀️</div>
                    <div class="node-val">{{ '%0.f'|format(solar) }}W</div>
                </div>
                
                <div class="node n-gen {{ 'pulse-red' if gen_on else '' }}">
                    <div class="node-icon">⚙️</div>
                    <div class="node-val">{{ 'ON' if gen_on else 'OFF' }}</div>
                </div>
                
                <div class="node n-inv">
                    <div class="node-icon">⚡</div>
                    <div class="node-val">INV</div>
                </div>
                
                <div class="node n-home {{ 'pulse-green' if load > 2000 else '' }}">
                    <div class="node-icon">
                        {% if 'Pool' in detected|string %}🏊
                        {% elif 'Water' in detected|string %}🚿
                        {% elif 'Cooking' in detected|string %}🍳
                        {% else %}🏠{% endif %}
                    </div>
                    <div class="node-val">{{ '%0.f'|format(load) }}W</div>
                </div>
                
                <div class="node n-bat {{ 'pulse-green' if is_charging else ('pulse-red' if is_discharging else '') }}">
                    <div class="node-icon">🔋</div>
                    <div class="node-val">{{ '%0.f'|format(p_pct) }}%</div>
                </div>
            </div>
        </div>

        <!-- Metrics Row -->
        <div class="col-3 card">
            <h2>Solar</h2>
            <div class="metric-val" style="color:var(--warn)">{{ '%0.f'|format(solar) }}</div>
            <div class="metric-unit">Watts</div>
        </div>
        <div class="col-3 card">
            <h2>Load</h2>
            <div class="metric-val" style="color:var(--info)">{{ '%0.f'|format(load) }}</div>
            <div class="metric-unit">Watts</div>
        </div>
        <div class="col-3 card">
            <h2>Primary</h2>
            <div class="metric-val" style="color:var(--success)">{{ '%0.f'|format(p_pct) }}</div>
            <div class="metric-unit">Percent</div>
        </div>
        <div class="col-3 card">
            <h2>Backup</h2>
            <div class="metric-val">{{ '%0.1f'|format(b_volt) }}</div>
            <div class="metric-unit">Volts</div>
        </div>

        <!-- Detected Activity -->
        <div class="col-6 card">
            <h2>Detected Appliances</h2>
            <div>
                {% if detected %}
                    {% for app in detected %}
                        <div class="tag" style="border-color:{{ 'var(--crit)' if 'Water' in app else 'var(--info)' }}">
                            {{ app }}
                        </div>
                    {% endfor %}
                {% else %}
                    <div style="opacity:0.5">System Idle</div>
                {% endif %}
            </div>
        </div>
        
        <!-- Alerts -->
        <div class="col-6 card">
            <h2>Recent Alerts</h2>
            {% for a in alerts %}
            <div class="alert-row">
                <div style="color:{{ 'var(--crit)' if 'crit' in a.type else 'var(--text)' }}">{{ a.subject }}</div>
                <div class="alert-time">{{ a.timestamp.strftime('%H:%M') }}</div>
            </div>
            {% endfor %}
        </div>

        <!-- Charts -->
        <div class="col-6 card">
            <h2>Forecast</h2>
            <div style="height:250px"><canvas id="c1"></canvas></div>
        </div>
        <div class="col-6 card">
            <h2>Battery Sim</h2>
            <div style="height:250px"><canvas id="c2"></canvas></div>
        </div>

    </div>

    <script>
        // Charts
        const common = { responsive: true, maintainAspectRatio: false, scales: { x:{grid:{display:false}}, y:{grid:{color:'rgba(255,255,255,0.1)'}} } };
        
        new Chart(document.getElementById('c1'), {
            type: 'line',
            data: {
                labels: {{ c_labels|tojson }},
                datasets: [
                    { label: 'Load', data: {{ c_load|tojson }}, borderColor: '#00bfff', borderDash:[5,5] },
                    { label: 'Solar', data: {{ c_solar|tojson }}, borderColor: '#ffa500', fill:true, backgroundColor:'rgba(255,165,0,0.1)' }
                ]
            }, options: common
        });
        
        new Chart(document.getElementById('c2'), {
            type: 'line',
            data: {
                labels: {{ sim.labels|tojson }},
                datasets: [{ label: 'Battery %', data: {{ sim.data|tojson }}, borderColor: '#00ff00', fill:true }]
            }, options: { ...common, scales: { y:{min:0, max:100} } }
        });
        
        // Auto-Poll
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
        c_labels=c_labels, c_load=c_load, c_solar=c_solar, sim=sim, alerts=alerts
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
