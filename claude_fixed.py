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

# API & Hardware Config
API_URL = "https://openapi.growatt.com/v1/device/storage/storage_last_data"
TOKEN = os.getenv("API_TOKEN")
SERIAL_NUMBERS = os.getenv("SERIAL_NUMBERS", "").split(",")
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", 5))
DATA_FILE = "load_patterns.json"

INVERTER_CONFIG = {
    "RKG3B0400T": {"label": "Inverter 1", "type": "primary"},
    "KAM4N5W0AG": {"label": "Inverter 2", "type": "primary"},
    "JNK1CDR0KQ": {"label": "Inverter 3 (Backup)", "type": "backup"}
}

# System Physics
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
# 1. Logic & Persistence
# ----------------------------
class PersistentLoadManager:
    def __init__(self, filename):
        self.filename = filename
        self.patterns = self.load_data()
        
    def load_data(self):
        if Path(self.filename).exists():
            try:
                with open(self.filename, 'r') as f: return json.load(f)
            except: pass
        return {"weekday": {str(h): [] for h in range(24)}, "weekend": {str(h): [] for h in range(24)}}

    def save_data(self):
        try:
            with open(self.filename, 'w') as f: json.dump(self.patterns, f)
        except: pass

    def update(self, load_watts):
        now = datetime.now(EAT)
        day = "weekend" if now.weekday() >= 5 else "weekday"
        h = str(now.hour)
        self.patterns[day][h].append(load_watts)
        if len(self.patterns[day][h]) > 100:
            self.patterns[day][h] = self.patterns[day][h][-100:]
            
    def get_forecast(self, hours_ahead=24):
        forecast = []
        now = datetime.now(EAT)
        for i in range(hours_ahead):
            ft = now + timedelta(hours=i)
            day = "weekend" if ft.weekday() >= 5 else "weekday"
            hist = self.patterns[day][str(ft.hour)]
            # Default Profile if no history: High in evening (Cooking), Low at night
            est = sum(hist) / len(hist) if hist else (2500 if 18<=ft.hour<=21 else 1000)
            forecast.append({'time': ft, 'estimated_load': est})
        return forecast

load_manager = PersistentLoadManager(DATA_FILE)

def identify_active_appliances(current, previous, gen_active, primary_pct):
    detected = []
    delta = current - previous
    if gen_active:
        if primary_pct > 42: detected.append("Water Heating (Gen)")
        else: detected.append("System Charging")
    # Inverter Loads
    if current < 400: detected.append("Idle")
    elif 1000 <= current <= 1350: detected.append("Pool Pump")
    elif current > 1800: detected.append("Cooking/Oven")
    elif 400 <= current < 1000: detected.append("TV/Lights")
    if delta > 1500: detected.append("Kettle")
    return detected

# ----------------------------
# 2. Physics & Calculation
# ----------------------------
def calculate_battery_breakdown(p_pct, b_volts):
    b_pct = max(0, min(100, (b_volts - 51.0) / 2.0 * 100))
    p_cap = PRIMARY_BATTERY_CAPACITY_WH
    b_cap = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION
    
    curr_p = (p_pct / 100) * p_cap
    curr_b = (b_pct / 100) * b_cap
    
    # 40% of Primary is cutoff, 20% of Backup is cutoff
    p_avail = max(0, curr_p - (p_cap * 0.40))
    b_avail = max(0, curr_b - (b_cap * 0.20))
    reserve = (p_cap * 0.40) + (b_cap * 0.20)
    
    total_usable = (p_cap * 0.60) + (b_cap * 0.80)
    current_usable = p_avail + b_avail
    total_pct = (current_usable / total_usable * 100) if total_usable > 0 else 0
    
    return {'chart_data': [round(p_avail/1000, 1), round(b_avail/1000, 1), round(reserve/1000, 1)], 'total_pct': round(total_pct, 1)}

def calculate_battery_cascade(solar, load, p_pct, b_active=False):
    if not solar or not load: return {'labels': [], 'data': []}
    p_wh = PRIMARY_BATTERY_CAPACITY_WH
    b_wh = BACKUP_BATTERY_DEGRADED_WH * BACKUP_DEGRADATION
    total_sys_wh = p_wh + b_wh
    
    curr_p = (p_pct / 100.0) * p_wh
    curr_b = (100 if b_active else 0) / 100.0 * b_wh 
    
    sim_data, sim_labels = [], []
    run_p, run_b = curr_p, curr_b
    
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
    "solar_forecast": [], "load_forecast": [], "battery_sim": {"labels": [], "data": []},
    "energy_breakdown": {"chart_data": [1, 1, 1], "total_pct": 0}
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
    if alert_type in last_alert_time and (datetime.now(EAT) - last_alert_time[alert_type]) < timedelta(minutes=60): return
    if RESEND_API_KEY:
        try: requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_API_KEY}"}, json={"from": SENDER_EMAIL, "to": [RECIPIENT_EMAIL], "subject": subject, "html": html})
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
    print("🚀 System Started: Scenario Planner Mode")

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
                        
                        tot_out += op
                        tot_sol += sol
                        if pb > 0: tot_bat += pb
                        
                        cfg = INVERTER_CONFIG.get(sn, {"label": sn})
                        info = {"SN": sn, "Label": cfg['label'], "OutputPower": op, "Capacity": cap, "vBat": vb}
                        inv_data.append(info)
                        if cfg['type'] == 'primary': p_caps.append(cap)
                        elif cfg['type'] == 'backup':
                            b_data = info
                            if float(d.get("vac") or 0) > 100 or float(d.get("pAcInPut") or 0) > 50: gen_on = True
                except: pass

            p_min = min(p_caps) if p_caps else 0
            b_volts = b_data['vBat'] if b_data else 0
            b_act = b_data['OutputPower'] > 50 if b_data else False
            
            detected = identify_active_appliances(tot_out, prev_watts, gen_on, p_min)
            is_manual_gen = any("Water" in x for x in detected)
            if not is_manual_gen: load_manager.update(tot_out)
            
            if gen_on: send_email("Generator ON", "Running", "gen")
            if p_min < 30: send_email("Battery Critical", f"{p_min}%", "crit")
            
            if (now - last_save) > timedelta(hours=1):
                load_manager.save_data()
                last_save = now

            l_cast = load_manager.get_forecast(24)
            s_cast = generate_solar_forecast(wx_data)
            sim_res = calculate_battery_cascade(s_cast, l_cast, p_min, b_act)
            breakdown = calculate_battery_breakdown(p_min, b_volts)
            
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
                "load_forecast": l_cast,
                "solar_forecast": s_cast,
                "battery_sim": sim_res,
                "energy_breakdown": breakdown,
                "inverters": inv_data
            }
            print(f"Update: {tot_out}W | Bat: {breakdown['total_pct']}%")
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
    p_pct = _n("primary_battery_min")
    b_volt = _n("backup_battery_voltage")
    gen_on = d.get("generator_running", False)
    detected = d.get("detected_appliances", [])
    
    breakdown = d.get("energy_breakdown") or {"chart_data": [1,0,0], "total_pct": 0}
    sim = d.get("battery_sim") or {"labels": [], "data": []}
    s_fc = d.get("solar_forecast") or []
    l_fc = d.get("load_forecast") or []
    
    st_txt, st_col = "NORMAL", "#00bfff"
    if gen_on: st_txt, st_col = "GENERATOR ON", "#ff0000"
    elif p_pct < 40: st_txt, st_col = "BACKUP ACTIVE", "#ffa500"
    
    c_labels = [x['time'].strftime('%H:%M') for x in l_fc][:12] if l_fc else []
    c_load = [x['estimated_load'] for x in l_fc][:12] if l_fc else []
    c_solar = [x['estimated_generation'] for x in s_fc][:12] if s_fc else []
    alerts = alert_history[:5]

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solar Scenario Planner</title>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root { --bg:#000; --card:rgba(30,30,30,0.7); --border:rgba(255,255,255,0.1); --text:#fff; --p:#3fb950; --w:#ffa500; --c:#ff4444; --i:#00bfff; }
        body { background:var(--bg); color:var(--text); font-family:'DM Sans'; margin:0; padding:15px; }
        .grid { display:grid; grid-template-columns:repeat(12, 1fr); gap:15px; max-width:1400px; margin:0 auto; }
        .c-12 { grid-column:span 12; } .c-6 { grid-column:span 12; } .c-4 { grid-column:span 12; } .c-3 { grid-column:span 6; }
        @media(min-width:768px){ .c-6 { grid-column:span 6; } .c-4 { grid-column:span 4; } .c-3 { grid-column:span 3; } }
        
        .card { background:var(--card); border:1px solid var(--border); border-radius:20px; padding:20px; backdrop-filter:blur(10px); }
        .metric-val { font-family:'Space Mono'; font-size:1.8rem; font-weight:700; }
        .metric-lbl { font-size:0.9rem; opacity:0.6; }
        
        /* Scenario Buttons */
        .scen-btn { 
            background:rgba(255,255,255,0.05); border:1px solid var(--border); color:#fff; 
            padding:12px; border-radius:12px; cursor:pointer; width:100%; text-align:left;
            display:flex; align-items:center; gap:10px; transition:all 0.2s;
        }
        .scen-btn:hover { background:rgba(255,255,255,0.1); border-color:var(--i); }
        .scen-btn.active { background:var(--i); color:#000; border-color:var(--i); box-shadow:0 0 15px rgba(0,191,255,0.4); }
        .scen-icon { font-size:1.4rem; }
        
        /* Chart Container */
        .chart-box { height:250px; position:relative; }
        
        /* Tags */
        .tag { padding:4px 10px; border-radius:50px; background:rgba(255,255,255,0.1); font-size:0.8rem; display:inline-block; margin:2px; }
    </style>
</head>
<body>
    <div class="grid">
        <div class="c-12" style="display:flex; justify-content:space-between; align-items:center">
            <div><h1 style="margin:0; font-size:1.4rem">SCENARIO PLANNER</h1><span style="color:{{ st_col }}">{{ st_txt }}</span></div>
            <div style="font-family:'Space Mono'; font-size:1.2rem">{{ d.timestamp.split(' ')[1] }}</div>
        </div>

        <!-- Metrics -->
        <div class="c-3 card">
            <div class="metric-lbl">Solar Input</div>
            <div class="metric-val" style="color:var(--w)">{{ '%0.f'|format(solar) }}<span style="font-size:1rem">W</span></div>
        </div>
        <div class="c-3 card">
            <div class="metric-lbl">Current Load</div>
            <div class="metric-val" style="color:var(--i)">{{ '%0.f'|format(load) }}<span style="font-size:1rem">W</span></div>
        </div>
        <div class="c-3 card">
            <div class="metric-lbl">Usable Battery</div>
            <div class="metric-val" style="color:var(--p)">{{ breakdown.total_pct }}<span style="font-size:1rem">%</span></div>
        </div>
        <div class="c-3 card">
            <div class="metric-lbl">Grid/Gen</div>
            <div class="metric-val" style="color:{{ 'var(--c)' if gen_on else '#555' }}">{{ 'ON' if gen_on else 'OFF' }}</div>
        </div>

        <!-- What-If Controls -->
        <div class="c-4 card">
            <h3 style="margin-top:0">Create Scenario</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px">
                <button class="scen-btn" onclick="toggleSim('oven', 2500)">
                    <span class="scen-icon">🍳</span> <div>Oven<br><small>2500W</small></div>
                </button>
                <button class="scen-btn" onclick="toggleSim('pool', 1200)">
                    <span class="scen-icon">🏊</span> <div>Pool<br><small>1200W</small></div>
                </button>
                <button class="scen-btn" onclick="toggleSim('wash', 800)">
                    <span class="scen-icon">🧺</span> <div>Washer<br><small>800W</small></div>
                </button>
                <button class="scen-btn" onclick="toggleSim('kettle', 2000)">
                    <span class="scen-icon">☕</span> <div>Kettle<br><small>2000W</small></div>
                </button>
            </div>
            <div style="margin-top:15px; border-top:1px solid var(--border); padding-top:10px">
                <div style="display:flex; justify-content:space-between">
                    <span>Simulated Load:</span>
                    <span id="sim-load-disp" style="font-family:'Space Mono'; color:var(--w)">0W</span>
                </div>
            </div>
        </div>

        <!-- Interactive Chart -->
        <div class="c-8 card">
            <h3 style="margin-top:0">Battery Projection (Real-Time)</h3>
            <div class="chart-box"><canvas id="simChart"></canvas></div>
        </div>

        <!-- Footer Info -->
        <div class="c-6 card">
            <h3 style="margin-top:0">Detected Now</h3>
            {% if detected %}{% for a in detected %}<span class="tag">{{ a }}</span>{% endfor %}{% else %}<span style="opacity:0.5">Idle</span>{% endif %}
        </div>
        <div class="c-6 card">
            <h3 style="margin-top:0">Battery Breakdown</h3>
            <div style="height:10px; background:#333; border-radius:5px; overflow:hidden; display:flex">
                <div style="width:{{ breakdown.chart_data[0] * 3 }}%; background:var(--p)" title="Primary"></div>
                <div style="width:{{ breakdown.chart_data[1] * 4 }}%; background:var(--w)" title="Backup"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-top:5px; opacity:0.7">
                <span>Primary</span><span>Backup</span><span>Reserve</span>
            </div>
        </div>
    </div>

    <script>
        // Data from Backend
        const labels = {{ sim.labels|tojson }};
        const baseData = {{ sim.data|tojson }};
        const sForecast = {{ s_fc|tojson }};
        const lForecast = {{ l_fc|tojson }};
        
        // Simulation State
        let activeSims = {};
        let totalSimWatts = 0;
        
        // Init Chart
        const ctx = document.getElementById('simChart');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Current Path',
                        data: baseData,
                        borderColor: '#3fb950', // Green
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: 'Scenario Path',
                        data: [], // Populated by JS
                        borderColor: '#ffa500', // Orange
                        borderDash: [5, 5],
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0,
                        hidden: true
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                scales: { 
                    y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.1)' } },
                    x: { grid: { display: false } }
                }
            }
        });

        function toggleSim(id, watts) {
            const btn = event.currentTarget;
            
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
            // Sum watts
            totalSimWatts = Object.values(activeSims).reduce((a, b) => a + b, 0);
            document.getElementById('sim-load-disp').innerText = "+" + totalSimWatts + "W";
            
            if (totalSimWatts === 0) {
                chart.data.datasets[1].hidden = true;
                chart.update();
                return;
            }

            // Run JS Physics Simulation (Simplified)
            // We take the current battery %, and for next 24h, apply (Base Load + Sim Load) vs Solar
            // Constants from Python (Hardcoded for JS speed)
            const BAT_WH = 30000 + (21000 * 0.7); // Total System Wh
            let currentPct = baseData[0]; // Start at current
            let currentWh = (currentPct / 100) * BAT_WH;
            
            let simCurve = [];
            
            for(let i=0; i<labels.length; i++) {
                // Get forecasted base load and solar
                let baseL = (lForecast[i] ? lForecast[i].estimated_load : 1000);
                let sol = (sForecast[i] ? sForecast[i].estimated_generation : 0);
                
                let totalL = baseL + totalSimWatts;
                let net = sol - totalL;
                
                // Physics step
                if(net > 0) {
                    currentWh = Math.min(BAT_WH, currentWh + net);
                } else {
                    currentWh = Math.max(0, currentWh - Math.abs(net));
                }
                
                simCurve.push((currentWh / BAT_WH) * 100);
            }
            
            chart.data.datasets[1].data = simCurve;
            chart.data.datasets[1].hidden = false;
            chart.update();
        }
        
        // Auto-Start
        fetch('/health').then(r=>r.json()).then(d=>{ if(!d.polling_thread_alive) fetch('/start-polling'); });
        setTimeout(()=>location.reload(), 60000);
    </script>
</body>
</html>
    """
    return render_template_string(html, 
        d=d, solar=solar, load=load, p_pct=p_pct, b_volt=b_volt, 
        gen_on=gen_on, detected=detected, st_txt=st_txt, st_col=st_col,
        s_fc=s_fc, l_fc=l_fc, sim=sim, breakdown=breakdown
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
