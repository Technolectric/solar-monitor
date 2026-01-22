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
warnings.filterwarnings('ignore')

print("🚀 Starting application initialization...", flush=True)

# ----------------------------
# 1. HTML TEMPLATE (Moved to top for safety)
# ----------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ site_config['label'] }}</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root { --bg: #0a0e27; --card: rgba(21, 27, 61, 0.7); --text: #e2e8f5; --success: #10b981; --warn: #f59e0b; --crit: #ef4444; --info: #3b82f6; }
        body { background: var(--bg); color: var(--text); font-family: 'Manrope', sans-serif; margin: 0; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; max-width: 1600px; margin: 0 auto; }
        .card { background: var(--card); padding: 20px; border-radius: 20px; border: 1px solid rgba(99, 102, 241, 0.2); }
        .col-4 { grid-column: span 12; } @media(min-width:768px){ .col-4 { grid-column: span 4; } }
        .col-12 { grid-column: span 12; }
        .metric-val { font-family: 'JetBrains Mono'; font-size: 2rem; font-weight: 700; }
        .rec-item { padding: 10px; margin-bottom: 5px; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid var(--info); }
        .tag { padding: 4px 10px; background: rgba(99, 102, 241, 0.2); border-radius: 20px; font-size: 0.8rem; margin-right: 5px; }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; margin-bottom:20px; max-width:1600px; margin:0 auto;">
        <h1>{{ site_config['label'] }}</h1>
        <div><a href="/update-kplc-billing" style="color:var(--info)">Check Bill</a> | <a href="/logout" style="color:var(--crit)">Logout</a></div>
    </div>

    <div class="grid">
        <div class="col-4 card">
            <div style="color:var(--warn); font-size:0.8rem">SOLAR INPUT</div>
            <div class="metric-val">{{ '%0.f'|format(solar) }}W</div>
        </div>
        <div class="col-4 card">
            <div style="color:var(--info); font-size:0.8rem">LOAD</div>
            <div class="metric-val">{{ '%0.f'|format(load) }}W</div>
        </div>
        <div class="col-4 card">
            <div style="color:var(--success); font-size:0.8rem">BATTERY</div>
            <div class="metric-val">{{ breakdown['total_pct'] }}%</div>
        </div>

        {% if site_id == 'nairobi' and billing_summary %}
        <div class="col-12 card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3>⚡ KPLC Bill Tracker</h3>
            {% if billing_summary.current_forecast %}
            <div style="display:flex; gap:20px; margin-bottom:15px;">
                <div>
                    <small>Accumulated</small>
                    <div style="font-size:1.5rem; font-weight:bold">{{ "%.1f"|format(billing_summary.current_forecast.accumulated_kwh) }} kWh</div>
                </div>
                <div>
                    <small>Est. Cost</small>
                    <div style="font-size:1.5rem; font-weight:bold">KES {{ "{:,.0f}"|format(billing_summary.current_forecast.estimated_amount) }}</div>
                </div>
            </div>
            {% endif %}
            
            {% if billing_summary.latest_bills %}
            <table style="width:100%; font-size:0.9rem; background:rgba(0,0,0,0.2); border-radius:8px;">
                <tr style="text-align:left; color:rgba(255,255,255,0.7)"><th>Month</th><th>KPLC</th><th>Growatt</th><th>Match</th><th>Cost</th></tr>
                {% for b in billing_summary.latest_bills %}
                <tr>
                    <td>{{ b.month }}</td>
                    <td>{{ "%.0f"|format(b.kplc_kwh) }}</td>
                    <td>{{ "%.0f"|format(b.growatt_kwh) }}</td>
                    <td>{{ "%.0f"|format(100 - b.match_accuracy) }}%</td>
                    <td>{{ "{:,.0f}"|format(b.kplc_amount) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </div>
        {% endif %}

        <div class="col-12 card">
            <div style="height:250px"><canvas id="mainChart"></canvas></div>
        </div>
        
        <div class="col-12 card">
            <div>Detected: 
                {% for d in detected %}<span class="tag">{{ d }}</span>{% endfor %}
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('mainChart');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: {{ hourly_24h|map(attribute='timestamp')|list|tojson }},
                datasets: [{
                    label: 'Load',
                    data: {{ hourly_24h|map(attribute='load')|list|tojson }},
                    borderColor: '#3b82f6', tension: 0.4
                }, {
                    label: 'Solar',
                    data: {{ hourly_24h|map(attribute='solar')|list|tojson }},
                    borderColor: '#f59e0b', tension: 0.4
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false } } }
        });
        
        // Auto-reload
        setTimeout(() => location.reload(), 60000);
        
        // Start polling if needed
        fetch('/health').then(r=>r.json()).then(d=>{ if(!d.polling_thread_alive) fetch('/start-polling') });
    </script>
</body>
</html>
"""

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { background: #0a0e27; display: flex; align-items: center; justify-content: center; height: 100vh; font-family: sans-serif; color: white; }
        form { background: rgba(255,255,255,0.1); padding: 40px; border-radius: 10px; text-align: center; }
        input { padding: 10px; border-radius: 5px; border: none; margin-bottom: 10px; width: 200px; }
        button { padding: 10px 20px; background: #3b82f6; color: white; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <form method="POST">
        <h2>Solar Monitor</h2>
        <input type="password" name="password" placeholder="Password" required autofocus><br>
        <button type="submit">Login</button>
    </form>
</body>
</html>
"""

# ----------------------------
# Flask App & Config
# ----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "solar-multisite-2026")

# API Configuration
API_URL = "https://openapi.growatt.com/v1/device/storage/storage_last_data"
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", 5))
DATA_FILE = "load_patterns.json"
HISTORY_FILE = "daily_history.json"
ML_MODEL_FILE = "appliance_models.pkl"
KPLC_BILLING_FILE = "kplc_billing_data.json"

for file in [DATA_FILE, HISTORY_FILE, ML_MODEL_FILE, KPLC_BILLING_FILE]:
    if not Path(file).exists():
        Path(file).touch()

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

RESEND_API_KEY = os.getenv('RESEND_API_KEY')
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')
WEATHERAPI_KEY = os.getenv('WEATHERAPI_KEY')

# ----------------------------
# KPLC Billing Integration
# ----------------------------
class KPLCBillingTracker:
    def __init__(self):
        self.billing_file = KPLC_BILLING_FILE
        self.account_number = os.getenv("KPLC_ACCOUNT_NUMBER", "2073344")
        self.billing_data = self.load_billing_data()
        self.data_lock = Lock()
        
        self.base_url = "https://selfservice.kplc.co.ke"
        self.auth_header = "Basic aVBXZkZTZTI2NkF2eVZHc2xpWk45Nl8yTzVzYTp3R3lRZEFFa3MzRm9lSkZHU0ZZUndFMERUdGNh"
        self.session = requests.Session()
        
        # HEADERS FIX for 403 Forbidden
        self.common_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Origin": "https://selfservice.kplc.co.ke",
            "Referer": "https://selfservice.kplc.co.ke/public/",
            "X-Self-Service-Channel": "WEB",
            "Accept": "application/json, text/plain, */*"
        }
        
    def load_billing_data(self):
        try:
            if os.path.getsize(self.billing_file) > 0:
                with open(self.billing_file, 'r') as f:
                    return json.load(f)
        except: pass
        return {"historical_bills": [], "current_forecast": None}
    
    def save_billing_data(self):
        try:
            with self.data_lock:
                with open(self.billing_file, 'w') as f:
                    json.dump(self.billing_data, f, indent=2)
        except Exception as e:
            print(f"Error saving billing data: {e}", flush=True)
    
    def get_api_token(self):
        url = f"{self.base_url}/api/token"
        headers = self.common_headers.copy()
        headers["Authorization"] = self.auth_header
        headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
        try:
            resp = self.session.post(url, headers=headers, data="grant_type=client_credentials", timeout=10)
            if resp.status_code == 200:
                return resp.json().get("access_token")
            else:
                print(f"KPLC Token Failed: {resp.status_code}", flush=True)
        except Exception as e:
            print(f"KPLC Token Error: {e}", flush=True)
        return None

    def scrape_kplc_bill(self):
        token = self.get_api_token()
        if not token: return None

        url = f"{self.base_url}/api/publicData/4/newContractList"
        headers = self.common_headers.copy()
        headers["Authorization"] = f"Bearer {token}"
        headers["Content-Type"] = "application/json"
        
        try:
            resp = self.session.get(url, headers=headers, params={"accountReference": self.account_number}, timeout=15)
            if resp.status_code == 200:
                return self.parse_kplc_response(resp.json())
            else:
                print(f"KPLC API Failed: {resp.status_code} - {resp.text}", flush=True)
        except Exception as e:
            print(f"Error fetching KPLC data: {e}", flush=True)
        return None
    
    def parse_kplc_response(self, data):
        try:
            if not data.get('data') or len(data['data']) == 0: return None
            main_record = data['data'][0]
            if not main_record.get('colBills') or len(main_record['colBills']) == 0: return None
            latest = main_record['colBills'][0]
            
            bill_ts = latest.get('billDate')
            if bill_ts:
                bill_date_obj = datetime.fromtimestamp(bill_ts / 1000.0)
                bill_date_str = bill_date_obj.strftime("%Y-%m-%d")
                month_str = bill_date_obj.strftime("%Y-%m")
            else:
                now = datetime.now()
                bill_date_str = now.strftime("%Y-%m-%d")
                month_str = now.strftime("%Y-%m")
            
            total_kwh = 0.0
            found_concepts = False
            for item in latest.get('concepts', []):
                name = item.get('conceptName', '')
                if 'Consumption' in name and item.get('unit') == 'kWh':
                    total_kwh += float(item.get('base', 0))
                    found_concepts = True
            
            return {
                'month': month_str,
                'kwh': total_kwh,
                'amount': float(latest.get('amount', 0)),
                'bill_date': bill_date_str,
            }
        except Exception as e:
            print(f"Error parsing KPLC: {e}", flush=True)
        return None
    
    def get_historical_kwh_from_growatt(self, site_config, start_date, end_date):
        try:
            token = site_config['api_token']
            if not site_config.get('serial_numbers'): return None
            serial_num = site_config['serial_numbers'][0]
            url = "https://openapi.growatt.com/v1/device/storage/storage_history_data"
            
            print(f"📊 Growatt History: {start_date.date()} to {end_date.date()}", flush=True)
            
            s_resp = requests.get(url, headers={"token": token}, params={"sn": serial_num, "date": start_date.strftime("%Y-%m-%d")}, timeout=10)
            e_resp = requests.get(url, headers={"token": token}, params={"sn": serial_num, "date": end_date.strftime("%Y-%m-%d")}, timeout=10)
            
            if s_resp.status_code == 200 and e_resp.status_code == 200:
                s_data = s_resp.json()
                e_data = e_resp.json()
                
                start_total = 0
                end_total = 0
                
                if s_data.get('success') and s_data.get('data', {}).get('datas'):
                    start_total = float(s_data['data']['datas'][0].get('eToUserTotal', 0))
                
                if e_data.get('success') and e_data.get('data', {}).get('datas'):
                    end_total = float(e_data['data']['datas'][0].get('eToUserTotal', 0))
                
                if end_total == 0:
                    print(f"⚠️ Growatt returned 0 for end date", flush=True)
                    return None

                period_kwh = max(0, end_total - start_total)
                return {'kwh_consumed': period_kwh}
                
        except Exception as e:
            print(f"Error Growatt Hist: {e}", flush=True)
        return None
    
    def update_billing_comparison(self, site_config):
        kplc_bill = self.scrape_kplc_bill()
        if kplc_bill and kplc_bill['month']:
            existing = [b for b in self.billing_data['historical_bills'] if b.get('month') == kplc_bill['month']]
            if not existing:
                bill_month = datetime.strptime(kplc_bill['month'], "%Y-%m")
                p_start = bill_month.replace(day=1)
                next_month = bill_month.replace(day=28) + timedelta(days=4)
                p_end = next_month - timedelta(days=next_month.day)
                
                g_data = self.get_historical_kwh_from_growatt(site_config, p_start, p_end)
                
                if g_data:
                    kwh_rate = kplc_bill['amount'] / kplc_bill['kwh'] if kplc_bill['kwh'] > 0 else 0
                    rec = {
                        'month': kplc_bill['month'],
                        'kplc_kwh': kplc_bill['kwh'],
                        'kplc_amount': kplc_bill['amount'],
                        'growatt_kwh': g_data['kwh_consumed'],
                        'kwh_rate': kwh_rate,
                        'bill_date': kplc_bill['bill_date'],
                        'match_accuracy': abs(kplc_bill['kwh'] - g_data['kwh_consumed']) / kplc_bill['kwh'] * 100 if kplc_bill['kwh'] > 0 else 0
                    }
                    self.billing_data['historical_bills'].append(rec)
                    self.save_billing_data()
                    print(f"✅ Added bill for {kplc_bill['month']}", flush=True)
        self.update_forecast(site_config)
    
    def update_forecast(self, site_config):
        try:
            now = datetime.now(EAT)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            g_data = self.get_historical_kwh_from_growatt(site_config, month_start, now)
            
            if g_data:
                rates = [b['kwh_rate'] for b in self.billing_data['historical_bills'] if b.get('kwh_rate', 0) > 0]
                avg_rate = sum(rates) / len(rates) if rates else 25.0
                
                self.billing_data['current_forecast'] = {
                    'month': now.strftime("%Y-%m"),
                    'accumulated_kwh': g_data['kwh_consumed'],
                    'estimated_amount': g_data['kwh_consumed'] * avg_rate,
                    'rate_used': avg_rate,
                    'last_updated': now.isoformat(),
                    'days_elapsed': (now - month_start).days
                }
                self.save_billing_data()
        except Exception as e:
            print(f"Forecast Error: {e}", flush=True)
    
    def get_billing_summary(self):
        summary = {
            'historical_count': len(self.billing_data.get('historical_bills', [])),
            'latest_bills': self.billing_data.get('historical_bills', [])[-3:],
            'current_forecast': self.billing_data.get('current_forecast'),
            'average_monthly_kwh': 0, 'average_monthly_cost': 0
        }
        if self.billing_data.get('historical_bills'):
            bills = self.billing_data['historical_bills']
            if len(bills) > 0:
                summary['average_monthly_kwh'] = sum(b.get('kplc_kwh', 0) for b in bills) / len(bills)
                summary['average_monthly_cost'] = sum(b.get('kplc_amount', 0) for b in bills) / len(bills)
        return summary

kplc_tracker = KPLCBillingTracker()

# ----------------------------
# Appliance Detection & Logic
# ----------------------------
class ApplianceDetector:
    def __init__(self, appliance_type="home"):
        self.appliance_type = appliance_type
        self.model_file = f"{ML_MODEL_FILE}_{appliance_type}"
        self.load_history = deque(maxlen=1000)
        self.appliance_classifier = None
        self.scaler = StandardScaler()
        self.model_lock = Lock()
        self.feature_window = 10
        self.training_data = []
        self.training_labels = []
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
                    else: self.init_default_models()
            except: self.init_default_models()
        else: self.init_default_models()
    
    def init_default_models(self):
        self.appliance_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        seed_data = [[50, 20, 100, 10, 45, 5, 10, 20, -15, 0, 0.4, 0.7], [3500, 600, 4200, 2800, 3400, 200, 400, 800, -600, 3, 0.17, 0.8]]
        seed_labels = ['idle', 'multiple_loads']
        self.training_data = seed_data
        self.training_labels = seed_labels
        X = np.array(seed_data)
        self.scaler.fit(X)
        self.appliance_classifier.fit(self.scaler.transform(X), seed_labels)
    
    def save_model(self):
        try:
            with self.model_lock:
                joblib.dump({'appliance_classifier': self.appliance_classifier, 'scaler': self.scaler, 'training_data': self.training_data, 'training_labels': self.training_labels}, self.model_file)
        except: pass
    
    def extract_features(self, load_data, time_data=None):
        features = []
        if len(load_data) == 0: return [0] * 12
        features.extend([np.mean(load_data), np.std(load_data), np.max(load_data), np.min(load_data), np.median(load_data)])
        if len(load_data) > 1:
            changes = np.diff(load_data)
            features.extend([np.mean(changes), np.std(changes), np.max(changes), np.min(changes), np.sum(np.abs(changes) > 500)])
            features.append(np.std(load_data) / np.mean(load_data) if np.mean(load_data) > 0 else 0)
        else: features.extend([0]*6)
        features.append(1.0 if time_data and 6 <= time_data.hour <= 22 else 0.5)
        return features[:12]
    
    def detect_appliances(self, current_load, previous_load=0):
        try:
            now = datetime.now(EAT)
            self.load_history.append({'timestamp': now, 'load': current_load})
            if len(self.load_history) < self.feature_window: return self._simple_fallback_detection(current_load)
            recent = [item['load'] for item in list(self.load_history)[-self.feature_window:]]
            features = self.extract_features(recent, now)
            
            try:
                f_scaled = self.scaler.transform([features])
                pred = self.appliance_classifier.predict(f_scaled)[0]
                conf = np.max(self.appliance_classifier.predict_proba(f_scaled))
                if conf > 0.4:
                    return [pred.replace('_', ' ').title()]
            except: pass
            return self._simple_fallback_detection(current_load)
        except: return ["System Error"]
    
    def _simple_fallback_detection(self, current_load):
        if current_load < 50: return ["Idle"]
        elif 50 <= current_load < 400: return ["Lights/Comp"]
        elif 400 <= current_load < 1500: return ["Office/Kitchen"]
        return ["High Load"]

    def train_from_feedback(self, feedback_data):
        try:
            load_p = feedback_data['load_pattern']
            ts = datetime.fromisoformat(feedback_data['timestamp'])
            feat = self.extract_features(load_p, ts)
            self.training_data.append(feat)
            self.training_labels.append(feedback_data['actual_appliance'])
            if len(self.training_data) > 500:
                self.training_data = self.training_data[-500:]
                self.training_labels = self.training_labels[-500:]
            X = np.array(self.training_data)
            self.scaler.fit(X)
            self.appliance_classifier.fit(self.scaler.transform(X), self.training_labels)
            self.save_model()
        except: pass

class PersistentLoadManager:
    def __init__(self, filename):
        self.filename = filename
        self.patterns = self.load_data()
    def load_data(self):
        try:
            with open(self.filename, 'r') as f: return json.load(f)
        except: return {"weekday": {str(h): [] for h in range(24)}, "weekend": {str(h): [] for h in range(24)}}
    def save_data(self):
        try:
            with open(self.filename, 'w') as f: json.dump(self.patterns, f)
        except: pass
    def update(self, load_watts):
        now = datetime.now(EAT)
        dt = "weekend" if now.weekday() >= 5 else "weekday"
        self.patterns[dt][str(now.hour)].append(load_watts)
        if len(self.patterns[dt][str(now.hour)]) > 100: self.patterns[dt][str(now.hour)] = self.patterns[dt][str(now.hour)][-100:]
    def get_forecast(self, hours_ahead=24):
        fc = []
        now = datetime.now(EAT)
        for i in range(hours_ahead):
            ft = now + timedelta(hours=i)
            dt = "weekend" if ft.weekday() >= 5 else "weekday"
            hist = self.patterns[dt][str(ft.hour)]
            est = sum(hist) / len(hist) if hist else 600
            fc.append({'time': ft, 'estimated_load': est})
        return fc

class DailyHistoryManager:
    def __init__(self, filename):
        self.filename = filename
        self.history = self.load_history()
        self.hourly_data = []
    def load_history(self):
        try:
            with open(self.filename, 'r') as f: return json.load(f)
        except: return {}
    def save_history(self):
        try:
            with open(self.filename, 'w') as f: json.dump(self.history, f)
        except: pass
    def add_hourly_datapoint(self, timestamp, load_w, battery_discharge_w, solar_w, grid_gen_w=0):
        self.hourly_data.append({'timestamp': timestamp.isoformat(), 'load': load_w, 'battery_discharge': battery_discharge_w, 'solar': solar_w, 'grid_gen': grid_gen_w})
        if len(self.hourly_data) > 288: self.hourly_data = self.hourly_data[-288:]
    def get_last_24h_data(self): return self.hourly_data
    def update_daily(self, date_str, total_consumption_wh, total_solar_wh, actual_irradiance_wh):
        if date_str not in self.history: self.history[date_str] = {'consumption': 0, 'solar': 0, 'potential': 0}
        self.history[date_str]['consumption'] = total_consumption_wh
        self.history[date_str]['solar'] = total_solar_wh
        self.history[date_str]['potential'] = actual_irradiance_wh
    def get_last_30_days(self):
        now = datetime.now(EAT)
        res = []
        for i in range(29, -1, -1):
            d = now - timedelta(days=i)
            ds = d.strftime('%Y-%m-%d')
            data = self.history.get(ds, {'consumption': 0, 'solar': 0, 'potential': 0})
            eff = min(100, (data['solar'] / data['potential']) * 100) if data['potential'] > 0 else 0
            res.append({'date': ds, 'day': d.day, 'month': d.strftime('%b'), 'weekday': d.strftime('%a'), 'consumption_kwh': round(data['consumption']/1000, 1), 'solar_kwh': round(data['solar']/1000, 1), 'efficiency': round(eff, 0)})
        return res

def identify_active_appliances(current, previous, gen_active, backup_volts, primary_pct, ml_detector_instance, site_id='kajiado'):
    detected = []
    if gen_active and site_id != 'nairobi':
        return ["Generator Load"] if primary_pct > 42 else ["System Charging"]
    apps = ml_detector_instance.detect_appliances(current, previous)
    return apps if apps else ([f"Load: {int(current)}W"] if current > 100 else ["Idle"])

APPLIANCE_PROFILES = [
    {"id": "pool", "name": "Pool Pump", "watts": 1200, "hours": 4, "icon": "🏊", "priority": "low"},
    {"id": "wash", "name": "Washer", "watts": 800, "hours": 1.5, "icon": "🧺", "priority": "medium"},
    {"id": "oven", "name": "Oven", "watts": 2500, "hours": 1.5, "icon": "🍳", "priority": "high"}
]

def get_energy_status(p_pct, b_volts, site_config):
    p_tot = site_config["primary_battery_wh"]
    b_tot = site_config["backup_battery_wh"] * site_config["backup_degradation"]
    curr_p = (p_pct / 100.0) * p_tot
    b_pct = max(0, min(100, (b_volts - 51.0) / 2.0 * 100)) if b_tot > 0 else 0
    curr_b = (b_pct / 100.0) * b_tot
    
    p_avail = max(0, curr_p - (p_tot * 0.40))
    b_avail = max(0, curr_b - (b_tot * 0.20)) if b_tot > 0 else 0
    e_avail = max(0, min(curr_p, p_tot * 0.40) - (p_tot * 0.20))
    
    total_cap = (p_tot * 0.60) + (b_tot * 0.80) + (p_tot * 0.20)
    total_avail = p_avail + b_avail + e_avail
    
    active = 'primary' if p_avail > 0 else ('backup' if b_avail > 0 else ('reserve' if e_avail > 0 else 'empty'))
    return {'total_pct': (total_avail/total_cap*100) if total_cap > 0 else 0, 'total_available_wh': total_avail, 'breakdown_wh': [p_avail, b_avail, e_avail], 'active_tier': active, 'b_pct': b_pct, 'curr_p_wh': curr_p, 'curr_b_wh': curr_b}

def generate_smart_schedule(status, solar_forecast_kw=0, load_forecast_kw=0, now_hour=None, heavy_loads_safe=False, gen_on=False, b_active=False, site_id='kajiado'):
    bat_pct = status['total_pct']
    advice = []
    
    for app in APPLIANCE_PROFILES:
        decision = {"msg": "Wait", "status": "unsafe", "color": "var(--warn)", "reason": ""}
        if gen_on and site_id != 'nairobi':
            decision = {"msg": "Gen On", "status": "unsafe", "color": "var(--crit)", "reason": "Generator"}
        elif site_id == 'nairobi' and not gen_on:
             decision = {"msg": "Grid Fail", "status": "unsafe", "color": "var(--crit)", "reason": "Battery Only"}
        elif bat_pct < 40:
            decision = {"msg": "Low Bat", "status": "unsafe", "color": "var(--crit)", "reason": "Low Battery"}
        elif heavy_loads_safe or bat_pct > 85:
            decision = {"msg": "Safe", "status": "safe", "color": "var(--success)", "reason": "Good Power"}
            
        advice.append({**app, "decision": decision["msg"], "status": decision["status"], "color": decision["color"], "reason": decision["reason"]})
    return advice

def calculate_battery_breakdown(p_pct, b_volts, site_config):
    s = get_energy_status(p_pct, b_volts, site_config)
    chart = [round(x/1000, 1) for x in s['breakdown_wh']]
    if site_config.get("backup_battery_wh", 0) == 0:
        chart = [chart[0], chart[2]]
        lbls = ['Primary', 'Reserve']
        cols = ['rgba(16, 185, 129, 0.9)', 'rgba(245, 158, 11, 0.8)']
    else:
        lbls = ['Primary', 'Backup', 'Reserve']
        cols = ['rgba(16, 185, 129, 0.9)', 'rgba(59, 130, 246, 0.8)', 'rgba(245, 158, 11, 0.8)']
    
    return {'chart_data': chart, 'tier_labels': lbls, 'tier_colors': cols, 'total_pct': round(s['total_pct'], 1), 'total_kwh': round(s['total_available_wh']/1000, 1), 'primary_pct': p_pct, 'backup_voltage': round(b_volts, 1), 'backup_pct': round(s['b_pct'], 1), 'status_obj': {**s, 'primary_battery_pct': p_pct}}

def calculate_battery_cascade(solar, load, p_pct, b_volts, site_config):
    if not solar or not load: return {'labels': [], 'data': [], 'tiers': []}
    st = get_energy_status(p_pct, b_volts, site_config)
    curr_p, curr_b = st['curr_p_wh'], st['curr_b_wh']
    p_tot, b_tot = site_config["primary_battery_wh"], site_config["backup_battery_wh"] * site_config["backup_degradation"]
    
    sim_data, sim_labels, tier_info = [st['total_pct']], ["Now"], [st['active_tier']]
    
    for i in range(min(len(solar), len(load))):
        net = solar[i]['estimated_generation'] - load[i]['estimated_load']
        if net > 0:
            space = p_tot - curr_p
            if net <= space: curr_p += net
            else:
                curr_p = p_tot
                if b_tot > 0: curr_b = min(b_tot, curr_b + (net - space))
        else:
            drain = abs(net)
            p_avail = max(0, curr_p - (p_tot * 0.40))
            if p_avail >= drain:
                curr_p -= drain
            else:
                curr_p = p_tot * 0.40
                drain -= p_avail
                if b_tot > 0:
                    b_avail = max(0, curr_b - (b_tot * 0.20))
                    if b_avail >= drain: curr_b -= drain
                    else:
                        curr_b = b_tot * 0.20
                        drain -= b_avail
                if drain > 0:
                    curr_p = max(p_tot * 0.20, curr_p - drain)
        
        p_av = max(0, curr_p - (p_tot * 0.40))
        b_av = max(0, curr_b - (b_tot * 0.20)) if b_tot > 0 else 0
        e_av = max(0, min(curr_p, p_tot * 0.40) - (p_tot * 0.20))
        tot_cap = (p_tot * 0.60) + (b_tot * 0.80) + (p_tot * 0.20)
        
        sim_data.append(((p_av + b_av + e_av)/tot_cap)*100 if tot_cap > 0 else 0)
        sim_labels.append(solar[i]['time'].strftime('%H:%M'))
        tier_info.append('primary' if p_av > 0 else ('backup' if b_av > 0 else 'reserve'))
        
    return {'labels': sim_labels, 'data': sim_data, 'tiers': tier_info}

last_alert_time = {}
alert_history = {}
site_latest_data = {}

def get_weather_forecast(lat, lon):
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=shortwave_radiation&timezone=Africa/Nairobi&forecast_days=2", timeout=5).json()
        return {'times': r['hourly']['time'], 'rad': r['hourly']['shortwave_radiation'], 'source': 'Open-Meteo'}
    except: return None

def generate_solar_forecast(wx, conf):
    fc = []
    if not wx: return fc
    now = datetime.now(EAT)
    w_map = {t: r for t, r in zip(wx['times'], wx['rad'])}
    for i in range(24):
        ft = now + timedelta(hours=i)
        rad = w_map.get(ft.strftime('%Y-%m-%dT%H:00'), 0)
        fc.append({'time': ft, 'estimated_generation': (rad/1000.0) * (conf["solar_capacity_kw"]*1000) * SOLAR_EFFICIENCY_FACTOR})
    return fc

def calculate_daily_irradiance_potential(wx, target, conf):
    if not wx: return 0
    total = 0
    w_map = {t: r for t, r in zip(wx['times'], wx['rad'])}
    for h in range(24):
        dt = target.replace(hour=h, minute=0)
        rad = w_map.get(dt.strftime('%Y-%m-%dT%H:00'), 0)
        total += (rad/1000.0) * (conf["solar_capacity_kw"]*1000) * SOLAR_EFFICIENCY_FACTOR
    return total

def send_email(sub, html, atype="general", email=True, site_id="kajiado"):
    if site_id not in last_alert_time: last_alert_time[site_id] = {}
    if site_id not in alert_history: alert_history[site_id] = []
    if atype in last_alert_time[site_id]:
        if (datetime.now(EAT) - last_alert_time[site_id][atype]) < timedelta(minutes=60): return
    
    recip = SITES.get(site_id, {}).get("recipient_email", RECIPIENT_EMAIL)
    if email and RESEND_API_KEY and recip:
        try: requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_API_KEY}"}, json={"from": SENDER_EMAIL, "to": [recip], "subject": f"[{SITES.get(site_id,{}).get('label')}] {sub}", "html": html})
        except: pass
    
    last_alert_time[site_id][atype] = datetime.now(EAT)
    alert_history[site_id].insert(0, {"timestamp": datetime.now(EAT), "type": atype, "subject": sub, "site_id": site_id})

def check_alerts(inv, sol, tot_sol, bat_dis, gen, site_id='kajiado'):
    for i in inv:
        if i.get('has_fault'): send_email(f"🚨 FAULT: {i['Label']}", "Fault", "fault", site_id=site_id)
    if site_id == 'nairobi' and not gen:
        send_email("🚨 CRITICAL: Grid Fail", "Battery Only", "critical", site_id=site_id)

polling_active = False
site_managers = {}

def poll_growatt():
    global site_latest_data, polling_active, site_managers
    
    # Track startup check for each site to avoid global variable issues
    site_startup_checks = {} 
    
    for sid, cfg in SITES.items():
        if sid not in site_managers:
            site_managers[sid] = {
                'load_manager': PersistentLoadManager(f"{sid}_{DATA_FILE}"),
                'history_manager': DailyHistoryManager(f"{sid}_{HISTORY_FILE}"),
                'ml_detector': ApplianceDetector(cfg["appliance_type"]),
                'daily_accumulator': {'consumption_wh': 0, 'solar_wh': 0, 'last_date': None},
                'pool_pump_start_time': None, 'pool_pump_last_alert': None,
                'last_save': datetime.now(EAT), 'last_ml_save': datetime.now(EAT), 'prev_watts': 0
            }
            site_startup_checks[sid] = False # Initialize check as not done
    
    polling_active = True
    while polling_active:
        try:
            for sid, cfg in SITES.items():
                mgr = site_managers[sid]
                tok = cfg["api_token"]
                if not tok: continue
                
                wx = get_weather_forecast(cfg["latitude"], cfg["longitude"])
                now = datetime.now(EAT)
                
                tot_out, tot_sol, tot_bat, tot_grid = 0, 0, 0, 0
                invs, p_caps = [], []
                b_dat, gen_on = None, False
                
                for sn in cfg["serial_numbers"]:
                    icfg = cfg["inverter_config"].get(sn, {"label": sn, "type": "unknown"})
                    try:
                        r = requests.post(API_URL, data={"storage_sn": sn}, headers={"token": tok, "Content-Type": "application/x-www-form-urlencoded"}, timeout=10).json()
                        if r.get("error_code") == 0:
                            d = r.get("data", {})
                            op = float(d.get("outPutPower") or 0)
                            cap = float(d.get("capacity") or 0)
                            sol = float(d.get("ppv") or 0) + float(d.get("ppv2") or 0)
                            pb = float(d.get("pBat") or 0)
                            gp = float(d.get("pAcInPut") or 0)
                            
                            tot_out += op; tot_sol += sol; tot_grid += gp
                            if pb > 0: tot_bat += pb
                            
                            invs.append({"SN": sn, "Label": icfg['label'], "OutputPower": op, "Capacity": cap, "has_fault": int(d.get("errorCode") or 0) != 0})
                            
                            if icfg['type'] == 'primary': 
                                p_caps.append(cap)
                                if sid == 'nairobi' and float(d.get("vGrid") or 0) >= 180: gen_on = True
                            elif icfg['type'] == 'backup':
                                b_dat = {"vBat": float(d.get("vBat") or 0), "OutputPower": op}
                    except: pass
                
                p_min = min(p_caps) if p_caps else 0
                b_v = b_dat['vBat'] if b_dat else 0
                b_act = b_dat['OutputPower'] > 50 if b_dat else False
                
                # Daily Logic
                cur_d = now.strftime('%Y-%m-%d')
                if mgr['daily_accumulator']['last_date'] != cur_d:
                    if mgr['daily_accumulator']['last_date']:
                        pot = calculate_daily_irradiance_potential(wx, now-timedelta(days=1), cfg)
                        mgr['history_manager'].update_daily(mgr['daily_accumulator']['last_date'], mgr['daily_accumulator']['consumption_wh'], mgr['daily_accumulator']['solar_wh'], pot)
                        mgr['history_manager'].save_history()
                    mgr['daily_accumulator'] = {'consumption_wh': 0, 'solar_wh': 0, 'last_date': cur_d}
                
                mgr['daily_accumulator']['consumption_wh'] += tot_out * (POLL_INTERVAL_MINUTES/60.0)
                mgr['daily_accumulator']['solar_wh'] += tot_sol * (POLL_INTERVAL_MINUTES/60.0)
                
                det = identify_active_appliances(tot_out, mgr['prev_watts'], gen_on, b_v, p_min, mgr['ml_detector'], sid)
                if not any("Generator" in x for x in det): mgr['load_manager'].update(tot_out)
                mgr['history_manager'].add_hourly_datapoint(now, tot_out, tot_bat, tot_sol, tot_grid)
                
                check_alerts(invs, None, tot_sol, tot_bat, gen_on, sid)
                
                if (now - mgr['last_save']) > timedelta(hours=1):
                    mgr['load_manager'].save_data()
                    mgr['last_save'] = now
                
                l_fc = mgr['load_manager'].get_forecast(24)
                s_fc = generate_solar_forecast(wx, cfg)
                bd = calculate_battery_breakdown(p_min, b_v, cfg)
                sim = calculate_battery_cascade(s_fc, l_fc, p_min, b_v, cfg)
                
                sched = generate_smart_schedule(bd['status_obj'], s_fc, l_fc, now.hour, False, gen_on, b_act, sid)
                del bd['status_obj']
                
                mgr['prev_watts'] = tot_out
                
                with Lock():
                    site_latest_data[sid] = {
                        "data": {
                            "timestamp": now.strftime("%H:%M:%S"),
                            "total_output_power": tot_out, "total_solar_input_W": tot_sol, "total_grid_input_W": tot_grid,
                            "primary_battery_min": p_min, "backup_battery_voltage": b_v, "generator_running": gen_on,
                            "detected_appliances": det, "load_forecast": l_fc[:12], "solar_forecast": s_fc[:12],
                            "battery_sim": sim, "energy_breakdown": bd, "scheduler": sched, "inverters": invs,
                            "heatmap_data": mgr['history_manager'].get_last_30_days(), "hourly_24h": mgr['history_manager'].get_last_24h_data()
                        },
                        "timestamp": now
                    }
                
                # KPLC Check (2AM or Startup)
                if sid == "nairobi" and (now.hour == 2 or site_startup_checks.get(sid) == False):
                    try: 
                        kplc_tracker.update_billing_comparison(cfg)
                        site_startup_checks[sid] = True
                    except: pass
                    
        except Exception as e: print(f"Loop Error: {e}", flush=True)
        time.sleep(POLL_INTERVAL_MINUTES * 60)

@app.route('/health')
def health(): return jsonify({"status": "healthy"})

@app.route('/start-polling')
def start_polling():
    global polling_active, polling_thread
    if not polling_active:
        polling_active = True
        polling_thread = Thread(target=poll_growatt, daemon=True)
        polling_thread.start()
    return jsonify({"status": "started"})

@app.route('/api/data')
def api_data():
    sid, _ = get_current_site()
    if not sid: return jsonify({"error": "Unauthorized"}), 401
    return jsonify(site_latest_data.get(sid, {}).get('data', {}))

@app.route('/update-kplc-billing')
def update_kplc_billing():
    """Deep Debug Endpoint"""
    site_id, site_config = get_current_site()
    if site_id != "nairobi": return jsonify({"error": "Nairobi only"}), 403
    
    logs = []
    def log(msg): logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")
    
    try:
        log("Checking KPLC...")
        tok = kplc_tracker.get_api_token()
        if not tok: return jsonify({"success": False, "logs": logs, "error": "Token Failed"})
        
        bill = kplc_tracker.scrape_kplc_bill()
        log(f"Bill Result: {bill}")
        
        if bill:
            kplc_tracker.update_billing_comparison(site_config)
            return jsonify({"success": True, "logs": logs, "data": kplc_tracker.get_billing_summary()})
        return jsonify({"success": False, "logs": logs, "error": "No Bill Found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "logs": logs})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        pwd = request.form.get('password', '')
        for sid, cfg in SITES.items():
            if pwd == cfg['password']:
                session['site_id'] = sid
                return redirect(url_for('home'))
        return render_template_string(LOGIN_TEMPLATE, error="Invalid password")
    return render_template_string(LOGIN_TEMPLATE, error=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/")
def home():
    sid, cfg = get_current_site()
    if not sid: return redirect(url_for('login'))
    
    d = site_latest_data.get(sid, {}).get('data', {})
    
    # Defaults
    if not d: d = {"timestamp": "Init...", "total_output_power": 0, "total_solar_input_W": 0, "energy_breakdown": {"chart_data": [1,0,1], "total_pct": 0, "total_kwh": 0}}
    
    # Billing Summary
    billing_summary = None
    if sid == "nairobi":
        try: billing_summary = kplc_tracker.get_billing_summary()
        except: pass

    return render_template_string(HTML_TEMPLATE, d=d, site_config=cfg, site_id=sid, billing_summary=billing_summary, 
                                  solar=d.get('total_solar_input_W',0), load=d.get('total_output_power',0), 
                                  p_pct=d.get('primary_battery_min',0), b_volt=d.get('backup_battery_voltage',0),
                                  breakdown=d.get('energy_breakdown', {}), sim=d.get('battery_sim', {}),
                                  heatmap=d.get('heatmap_data',[]), hourly_24h=d.get('hourly_24h',[]),
                                  alerts=alert_history.get(sid,[])[:8], detected=d.get('detected_appliances',[]),
                                  s_fc=d.get('solar_forecast',[]), l_fc=d.get('load_forecast',[]),
                                  schedule=d.get('scheduler',[]), recommendation_items=[], schedule_items=[],
                                  grid_watts=d.get('total_grid_input_W',0), is_importing=d.get('total_grid_input_W',0)>20,
                                  is_charging=d.get('total_solar_input_W',0) > d.get('total_output_power',0),
                                  is_discharging=d.get('total_battery_discharge_W',0) > 100,
                                  gen_on=d.get('generator_running', False),
                                  st_txt="NORMAL", st_col="var(--info)",
                                  tier_labels=['Primary','Backup'], primary_pct=0, backup_voltage=0, backup_pct=0, heavy_loads_safe=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
