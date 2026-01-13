import os
import time
import threading
import traceback
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, render_template_string
import requests

app = Flask(__name__)

# Configuration
TOKEN = os.getenv("API_TOKEN", "")
SERIAL_NUMBERS = os.getenv("SERIAL_NUMBERS", "RKG3B0400T,KAM4N5W0AG,JNK1CDR0KQ").split(",")
API_URL = "https://openapi.growatt.com/v1/device/storage/storage_last_data"
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", "5"))

print(f"🔧 Configuration: TOKEN={'SET' if TOKEN else 'NOT SET'}, SERIALS={len(SERIAL_NUMBERS)}")

# Global data storage
latest_data = {
    "timestamp": datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S"),
    "status": "Starting up...",
    "total_output_power": 0,
    "total_solar_input_W": 0,
    "primary_battery_min": 0,
    "backup_battery_voltage": 0,
    "backup_active": False,
    "generator_running": False,
    "inverters": [],
    "usable_energy": {
        "primary_kwh": 0,
        "backup_kwh": 0,
        "total_kwh": 0,
        "total_pct": 0
    }
}

# Track if polling is active
polling_active = False
polling_thread = None

# Health check endpoint
@app.route('/health')
def health():
    return "OK", 200

# Simple test endpoint
@app.route('/test')
def test():
    return jsonify({
        "app": "Solar Monitor",
        "status": "running",
        "timestamp": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "api_token_set": bool(TOKEN),
        "serial_numbers": SERIAL_NUMBERS,
        "port": os.getenv("PORT", "8080"),
        "polling_active": polling_active,
        "polling_thread_alive": polling_thread.is_alive() if polling_thread else False
    })

# Debug endpoint
@app.route('/debug')
def debug():
    threads = []
    for thread in threading.enumerate():
        threads.append({
            "name": thread.name,
            "alive": thread.is_alive(),
            "daemon": thread.daemon,
            "ident": thread.ident
        })
    
    return jsonify({
        "app": "Solar Monitor",
        "timestamp": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "environment": {
            "API_TOKEN_set": bool(TOKEN),
            "SERIAL_NUMBERS": SERIAL_NUMBERS,
            "PORT": os.getenv("PORT", "8080"),
            "POLL_INTERVAL_MINUTES": POLL_INTERVAL_MINUTES
        },
        "threads": threads,
        "thread_count": threading.active_count(),
        "latest_data": latest_data,
        "polling_active": polling_active,
        "polling_thread": {
            "exists": polling_thread is not None,
            "alive": polling_thread.is_alive() if polling_thread else False,
            "name": polling_thread.name if polling_thread else None
        }
    })

# API data endpoint
@app.route('/api/data')
def api_data():
    return jsonify(latest_data)

# Test inverter API call
@app.route('/test-fetch')
def test_fetch():
    """Test the Growatt API directly"""
    if not TOKEN:
        return jsonify({"error": "API_TOKEN not set"}), 400
    
    if len(SERIAL_NUMBERS) == 0:
        return jsonify({"error": "SERIAL_NUMBERS not set"}), 400
    
    try:
        sn = SERIAL_NUMBERS[0]
        headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"}
        
        response = requests.post(
            API_URL,
            data={"storage_sn": sn},
            headers=headers,
            timeout=10
        )
        
        return jsonify({
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else {"error": "Non-200 response"},
            "inverter": sn,
            "timestamp": datetime.now(timezone(timedelta(hours=3))).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "timestamp": datetime.now(timezone(timedelta(hours=3))).isoformat()
        }), 500

# Start/stop polling endpoints
@app.route('/start-polling')
def start_polling():
    """Manually start polling"""
    global polling_active, polling_thread
    
    if polling_active and polling_thread and polling_thread.is_alive():
        return jsonify({"status": "Polling already running"})
    
    try:
        polling_thread = threading.Thread(
            target=poll_growatt,
            name="polling_thread",
            daemon=True
        )
        polling_thread.start()
        polling_active = True
        
        return jsonify({
            "status": "Polling started",
            "thread_alive": polling_thread.is_alive(),
            "thread_name": polling_thread.name
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/stop-polling')
def stop_polling():
    """Stop polling"""
    global polling_active
    polling_active = False
    return jsonify({"status": "Polling stopped (will stop after current cycle)"})

# Polling function
def poll_growatt():
    """Main polling function"""
    global latest_data, polling_active
    
    print("🚀 POLL_GROWATT: Starting polling thread...")
    polling_active = True
    
    if not TOKEN:
        print("❌ POLL_GROWATT: No API_TOKEN, exiting")
        return
    
    print(f"✅ POLL_GROWATT: Starting with {len(SERIAL_NUMBERS)} inverters")
    
    try:
        while polling_active:
            try:
                now = datetime.now(timezone(timedelta(hours=3)))
                print(f"🔄 POLL_GROWATT: Polling at {now.strftime('%H:%M:%S')}")
                
                # Initialize totals
                tot_out, tot_sol, tot_bat = 0, 0, 0
                inv_data = []
                p_caps = []
                
                # Poll each inverter
                for sn in SERIAL_NUMBERS:
                    try:
                        print(f"  🔄 Polling {sn}...")
                        headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"}
                        
                        response = requests.post(
                            API_URL,
                            data={"storage_sn": sn},
                            headers=headers,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            if data.get("code") == 0:
                                d = data.get("data", {})
                                
                                # Extract values
                                op = float(d.get("outPutPower") or 0)
                                cap = float(d.get("capacity") or 0)
                                vb = float(d.get("vBat") or 0)
                                pb = float(d.get("pBat") or 0)
                                sol = float(d.get("ppv") or 0) + float(d.get("ppv2") or 0)
                                
                                # Update totals
                                tot_out += op
                                tot_sol += sol
                                if pb > 0:
                                    tot_bat += pb
                                
                                # Determine inverter type
                                inv_type = "backup" if "JNK" in sn else "primary"
                                label = f"Inverter {'Backup' if 'JNK' in sn else ('1' if 'RKG' in sn else '2')}"
                                
                                inv_data.append({
                                    "SN": sn,
                                    "Label": label,
                                    "Type": inv_type,
                                    "OutputPower": op,
                                    "Capacity": cap,
                                    "vBat": vb,
                                    "pBat": pb,
                                    "ppv": sol,
                                    "Status": d.get("statusText", "Unknown")
                                })
                                
                                if inv_type == "primary":
                                    p_caps.append(cap)
                                
                                print(f"  ✅ {label}: {op}W, {cap}%, {sol}W solar")
                            else:
                                print(f"  ❌ API error for {sn}: {data.get('msg', 'Unknown error')}")
                        else:
                            print(f"  ❌ HTTP error for {sn}: {response.status_code}")
                            
                    except Exception as e:
                        print(f"  ❌ Error polling {sn}: {str(e)[:100]}")
                
                # Calculate primary battery minimum
                p_min = min(p_caps) if p_caps else 0
                
                # Simple usable energy calculation
                primary_kwh = max(0, ((p_min - 40) / 100) * 30) if p_min > 40 else 0
                
                # Update latest data
                latest_data = {
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S EAT"),
                    "status": "Polling active",
                    "total_output_power": tot_out,
                    "total_solar_input_W": tot_sol,
                    "total_battery_discharge_W": tot_bat,
                    "primary_battery_min": p_min,
                    "backup_battery_voltage": next((inv["vBat"] for inv in inv_data if inv["Type"] == "backup"), 0),
                    "backup_active": any(inv["OutputPower"] > 50 and inv["Type"] == "backup" for inv in inv_data),
                    "generator_running": False,  # Simplified for now
                    "inverters": inv_data,
                    "usable_energy": {
                        "primary_kwh": round(primary_kwh, 1),
                        "backup_kwh": 0,  # Simplified for now
                        "total_kwh": round(primary_kwh, 1),
                        "total_pct": round((primary_kwh / 18) * 100, 1) if primary_kwh > 0 else 0
                    }
                }
                
                print(f"📊 {now.strftime('%H:%M:%S')} | Load: {tot_out:.0f}W | Solar: {tot_sol:.0f}W | Battery: {p_min}%")
                
            except Exception as e:
                print(f"❌ Error in polling cycle: {str(e)[:200]}")
                import traceback
                traceback.print_exc()
            
            # Wait for next poll
            print(f"⏳ Waiting {POLL_INTERVAL_MINUTES} minutes...")
            for i in range(POLL_INTERVAL_MINUTES * 60):
                if not polling_active:
                    print("🛑 Polling stopped by user")
                    return
                time.sleep(1)
                
    except Exception as e:
        print(f"🚨 POLL_GROWATT: Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        polling_active = False
        print("🛑 POLL_GROWATT: Thread exited")

# Main page
@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Solar Monitor</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: #1e293b; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .success { color: #10b981; }
            .error { color: #ef4444; }
            .warning { color: #f59e0b; }
            .info { color: #3b82f6; }
            a { color: #3b82f6; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .metric { font-size: 2em; font-weight: bold; margin: 10px 0; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
            button { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            button:hover { background: #2563eb; }
            button:disabled { background: #64748b; cursor: not-allowed; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌞 Solar Monitor Dashboard</h1>
            
            <div class="card">
                <h2>System Status</h2>
                <p>Status: <span id="status" class="info">Loading...</span></p>
                <p>Last Update: <span id="timestamp">--:--:--</span></p>
                <p>Polling: <span id="polling-status" class="warning">Checking...</span></p>
                <div>
                    <button onclick="startPolling()" id="start-btn">Start Polling</button>
                    <button onclick="stopPolling()" id="stop-btn" disabled>Stop Polling</button>
                    <button onclick="refreshData()">Refresh Data</button>
                </div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>⚡ Current Load</h3>
                    <div class="metric" id="load">0 W</div>
                    <div id="load-detail">--</div>
                </div>
                
                <div class="card">
                    <h3>☀️ Solar Output</h3>
                    <div class="metric success" id="solar">0 W</div>
                    <div id="solar-detail">--</div>
                </div>
                
                <div class="card">
                    <h3>🔋 Primary Battery</h3>
                    <div class="metric" id="battery">0 %</div>
                    <div id="battery-detail">--</div>
                </div>
                
                <div class="card">
                    <h3>🔄 Usable Energy</h3>
                    <div class="metric info" id="usable">0 %</div>
                    <div id="usable-detail">0 kWh</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Inverters</h2>
                <div id="inverters">Loading inverters...</div>
            </div>
            
            <div class="card">
                <h2>Quick Links</h2>
                <ul>
                    <li><a href="/test">Test Endpoint</a></li>
                    <li><a href="/debug">Debug Info</a></li>
                    <li><a href="/api/data">API Data</a></li>
                    <li><a href="/test-fetch">Test Growatt API</a></li>
                    <li><a href="/start-polling">Start Polling (API)</a></li>
                    <li><a href="/stop-polling">Stop Polling (API)</a></li>
                </ul>
            </div>
        </div>
        
        <script>
            let pollingInterval = null;
            
            function refreshData() {
                fetch('/api/data')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('status').innerText = data.status;
                        document.getElementById('timestamp').innerText = data.timestamp;
                        document.getElementById('load').innerText = Math.round(data.total_output_power) + ' W';
                        document.getElementById('solar').innerText = Math.round(data.total_solar_input_W) + ' W';
                        document.getElementById('battery').innerText = Math.round(data.primary_battery_min) + ' %';
                        document.getElementById('usable').innerText = Math.round(data.usable_energy.total_pct) + ' %';
                        document.getElementById('usable-detail').innerText = data.usable_energy.total_kwh + ' kWh usable';
                        
                        // Update inverters
                        let invertersHtml = '';
                        data.inverters.forEach(inv => {
                            invertersHtml += `
                                <div style="border: 1px solid #334155; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                    <strong>${inv.Label}</strong> (${inv.SN})<br>
                                    Output: ${Math.round(inv.OutputPower)}W | 
                                    Battery: ${Math.round(inv.Capacity)}% | 
                                    Solar: ${Math.round(inv.ppv)}W<br>
                                    Voltage: ${inv.vBat.toFixed(2)}V | 
                                    Status: ${inv.Status}
                                </div>
                            `;
                        });
                        document.getElementById('inverters').innerHTML = invertersHtml || 'No inverter data';
                    })
                    .catch(err => {
                        console.error('Error fetching data:', err);
                        document.getElementById('status').innerText = 'Error loading data';
                        document.getElementById('status').className = 'error';
                    });
            }
            
            function checkPollingStatus() {
                fetch('/test')
                    .then(r => r.json())
                    .then(data => {
                        const statusEl = document.getElementById('polling-status');
                        if (data.polling_thread_alive) {
                            statusEl.innerText = '✓ ACTIVE';
                            statusEl.className = 'success';
                            document.getElementById('start-btn').disabled = true;
                            document.getElementById('stop-btn').disabled = false;
                        } else {
                            statusEl.innerText = '✗ INACTIVE';
                            statusEl.className = 'error';
                            document.getElementById('start-btn').disabled = false;
                            document.getElementById('stop-btn').disabled = true;
                        }
                    });
            }
            
            function startPolling() {
                fetch('/start-polling')
                    .then(r => r.json())
                    .then(data => {
                        alert('Polling started: ' + (data.status || data.error));
                        checkPollingStatus();
                        setTimeout(refreshData, 2000); // Wait 2 seconds then refresh
                    });
            }
            
            function stopPolling() {
                fetch('/stop-polling')
                    .then(r => r.json())
                    .then(data => {
                        alert('Polling stopped');
                        checkPollingStatus();
                    });
            }
            
            // Initial load
            refreshData();
            checkPollingStatus();
            
            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);
            setInterval(checkPollingStatus, 10000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

# Startup
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    
    print(f"🚀 Starting Solar Monitor on port {port}")
    print(f"🔧 Environment: TOKEN={'SET' if TOKEN else 'NOT SET'}")
    print(f"🔧 Serial Numbers: {SERIAL_NUMBERS}")
    
    # Start Flask immediately
    print(f"🌐 Starting Flask on 0.0.0.0:{port}")
    
    # Note: We're NOT starting the polling thread automatically
    # User will start it manually via /start-polling
    print("ℹ️  Polling thread will be started manually via /start-polling")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )
