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
    "total_battery_discharge_W": 0,
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

# Diagnostic endpoint for polling
@app.route('/polling-debug')
def polling_debug():
    """Detailed debug info for polling"""
    # Test a direct API call
    test_result = None
    api_error = None
    
    if not TOKEN:
        api_error = "API_TOKEN not set"
    elif not SERIAL_NUMBERS:
        api_error = "SERIAL_NUMBERS not set"
    else:
        try:
            headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"}
            response = requests.post(
                API_URL,
                data={"storage_sn": SERIAL_NUMBERS[0]},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                test_result = {
                    "status_code": response.status_code,
                    "api_code": data.get("code"),
                    "api_message": data.get("msg", "No message"),
                    "has_data": bool(data.get("data")),
                    "data_keys": list(data.get("data", {}).keys()) if data.get("data") else []
                }
                
                # Extract some sample data
                if data.get("data"):
                    sample_data = data["data"]
                    test_result["sample_values"] = {
                        "outPutPower": sample_data.get("outPutPower"),
                        "capacity": sample_data.get("capacity"),
                        "vBat": sample_data.get("vBat"),
                        "ppv": sample_data.get("ppv"),
                        "statusText": sample_data.get("statusText")
                    }
            else:
                test_result = {
                    "status_code": response.status_code,
                    "error": f"HTTP Error: {response.status_code}",
                    "response_text": response.text[:200] if response.text else "No response body"
                }
                
        except requests.exceptions.Timeout:
            api_error = "API request timeout"
        except requests.exceptions.ConnectionError:
            api_error = "Connection error to Growatt API"
        except Exception as e:
            api_error = f"API Error: {str(e)}"
    
    # Check what the polling thread is doing
    polling_info = {}
    if polling_thread and polling_thread.is_alive():
        polling_info = {
            "is_alive": True,
            "name": polling_thread.name,
            "last_log": "Check Railway logs for polling output"
        }
    
    return jsonify({
        "timestamp": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "api_test": test_result if test_result else {"error": api_error},
        "polling_info": polling_info,
        "environment_check": {
            "API_TOKEN_length": len(TOKEN) if TOKEN else 0,
            "SERIAL_NUMBERS_count": len(SERIAL_NUMBERS),
            "POLL_INTERVAL_MINUTES": POLL_INTERVAL_MINUTES
        },
        "latest_data_snapshot": {
            "timestamp": latest_data.get("timestamp"),
            "inverter_count": len(latest_data.get("inverters", [])),
            "has_data": len(latest_data.get("inverters", [])) > 0
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
        
        result = {
            "status_code": response.status_code,
            "inverter": sn,
            "timestamp": datetime.now(timezone(timedelta(hours=3))).isoformat()
        }
        
        if response.status_code == 200:
            data = response.json()
            result["response"] = {
                "code": data.get("code"),
                "msg": data.get("msg"),
                "has_data": bool(data.get("data"))
            }
            
            if data.get("data"):
                # Extract key fields
                d = data["data"]
                result["extracted_data"] = {
                    "outPutPower": d.get("outPutPower"),
                    "capacity": d.get("capacity"),
                    "vBat": d.get("vBat"),
                    "pBat": d.get("pBat"),
                    "ppv": d.get("ppv"),
                    "ppv2": d.get("ppv2"),
                    "statusText": d.get("statusText"),
                    "errorCode": d.get("errorCode")
                }
        else:
            result["error"] = f"HTTP {response.status_code}"
            result["response_text"] = response.text[:500] if response.text else "No response body"
            
        return jsonify(result)
        
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
        # Stop any existing thread
        if polling_thread and polling_thread.is_alive():
            polling_active = False
            polling_thread.join(timeout=2)
        
        # Start new thread
        polling_thread = threading.Thread(
            target=poll_growatt,
            name="polling_thread",
            daemon=True
        )
        polling_thread.start()
        polling_active = True
        
        # Give it a moment to start
        time.sleep(1)
        
        return jsonify({
            "status": "Polling started",
            "thread_alive": polling_thread.is_alive(),
            "thread_name": polling_thread.name,
            "timestamp": datetime.now(timezone(timedelta(hours=3))).isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/stop-polling')
def stop_polling():
    """Stop polling"""
    global polling_active
    polling_active = False
    return jsonify({
        "status": "Polling stop signal sent",
        "timestamp": datetime.now(timezone(timedelta(hours=3))).isoformat()
    })

# Fixed polling function with better error handling
def poll_growatt():
    """Main polling function - FIXED VERSION"""
    global latest_data, polling_active
    
    print("🚀 POLL_GROWATT: Starting polling thread...")
    polling_active = True
    
    if not TOKEN:
        print("❌ POLL_GROWATT: No API_TOKEN, exiting")
        return
    
    if not SERIAL_NUMBERS:
        print("❌ POLL_GROWATT: No SERIAL_NUMBERS, exiting")
        return
    
    print(f"✅ POLL_GROWATT: Starting with {len(SERIAL_NUMBERS)} inverters")
    print(f"✅ POLL_GROWATT: Token present: {'Yes' if TOKEN else 'No'}")
    
    poll_count = 0
    
    try:
        while polling_active:
            try:
                poll_count += 1
                now = datetime.now(timezone(timedelta(hours=3)))
                print(f"\n🔄 POLL #{poll_count} at {now.strftime('%H:%M:%S')}")
                
                # Initialize totals
                tot_out, tot_sol, tot_bat = 0, 0, 0
                inv_data = []
                p_caps = []
                successful_polls = 0
                
                # Poll each inverter
                for sn in SERIAL_NUMBERS:
                    try:
                        print(f"  📡 Polling {sn}...")
                        headers = {"token": TOKEN, "Content-Type": "application/x-www-form-urlencoded"}
                        
                        # Make the API request
                        response = requests.post(
                            API_URL,
                            data={"storage_sn": sn},
                            headers=headers,
                            timeout=15
                        )
                        
                        print(f"  📊 Response status: {response.status_code}")
                        
                        if response.status_code == 200:
                            data = response.json()
                            api_code = data.get("code", -1)
                            api_msg = data.get("msg", "No message")
                            
                            print(f"  📋 API Code: {api_code}, Message: {api_msg}")
                            
                            if api_code == 0:
                                d = data.get("data", {})
                                
                                # Debug: print available keys
                                if poll_count == 1:  # Only on first poll
                                    print(f"  🔑 Data keys: {list(d.keys())[:10]}...")
                                
                                # Extract values with safe defaults
                                op = float(d.get("outPutPower") or 0)
                                cap = float(d.get("capacity") or 0)
                                vb = float(d.get("vBat") or 0)
                                pb = float(d.get("pBat") or 0)
                                sol = float(d.get("ppv") or 0) + float(d.get("ppv2") or 0)
                                
                                print(f"  📈 Extracted: {op}W, {cap}%, {vb}V, {sol}W solar")
                                
                                # Update totals
                                tot_out += op
                                tot_sol += sol
                                if pb > 0:
                                    tot_bat += pb
                                
                                # Determine inverter type and label
                                if "JNK" in sn:
                                    inv_type = "backup"
                                    label = "Inverter 3 (Backup)"
                                elif "RKG" in sn:
                                    inv_type = "primary"
                                    label = "Inverter 1"
                                else:
                                    inv_type = "primary"
                                    label = "Inverter 2"
                                
                                inv_data.append({
                                    "SN": sn,
                                    "Label": label,
                                    "Type": inv_type,
                                    "OutputPower": op,
                                    "Capacity": cap,
                                    "vBat": vb,
                                    "pBat": pb,
                                    "ppv": sol,
                                    "Status": d.get("statusText", "Unknown"),
                                    "has_fault": int(d.get("errorCode") or 0) != 0
                                })
                                
                                if inv_type == "primary":
                                    p_caps.append(cap)
                                
                                successful_polls += 1
                                print(f"  ✅ {label}: Success")
                            else:
                                print(f"  ❌ API error for {sn}: Code {api_code} - {api_msg}")
                        else:
                            print(f"  ❌ HTTP error for {sn}: {response.status_code}")
                            print(f"  📄 Response: {response.text[:200] if response.text else 'No body'}")
                            
                    except requests.exceptions.Timeout:
                        print(f"  ⏱️  Timeout for {sn}")
                    except requests.exceptions.ConnectionError:
                        print(f"  🔌 Connection error for {sn}")
                    except Exception as e:
                        print(f"  ❌ Unexpected error for {sn}: {str(e)[:100]}")
                
                print(f"  📊 Polling complete: {successful_polls}/{len(SERIAL_NUMBERS)} successful")
                
                # Calculate primary battery minimum
                p_min = min(p_caps) if p_caps else 0
                
                # Get backup battery voltage
                backup_voltage = 0
                backup_active = False
                for inv in inv_data:
                    if inv["Type"] == "backup":
                        backup_voltage = inv["vBat"]
                        backup_active = inv["OutputPower"] > 50
                        break
                
                # Simple usable energy calculation
                primary_kwh = max(0, ((p_min - 40) / 100) * 30) if p_min > 40 else 0
                backup_kwh = max(0, ((backup_voltage - 51.0) / 2.0 * 100 - 20) / 100 * 14.7) if backup_voltage > 51.0 else 0
                total_kwh = primary_kwh + backup_kwh
                total_pct = min(100, (total_kwh / 29.76) * 100) if total_kwh > 0 else 0
                
                # Update latest data
                latest_data.update({
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S EAT"),
                    "status": "Polling active",
                    "total_output_power": tot_out,
                    "total_solar_input_W": tot_sol,
                    "total_battery_discharge_W": tot_bat,
                    "primary_battery_min": p_min,
                    "backup_battery_voltage": backup_voltage,
                    "backup_active": backup_active,
                    "generator_running": False,  # Simplified for now
                    "inverters": inv_data,
                    "usable_energy": {
                        "primary_kwh": round(primary_kwh, 1),
                        "backup_kwh": round(backup_kwh, 1),
                        "total_kwh": round(total_kwh, 1),
                        "total_pct": round(total_pct, 1)
                    }
                })
                
                print(f"📊 Summary: Load: {tot_out:.0f}W | Solar: {tot_sol:.0f}W | Battery: {p_min}% | Usable: {total_pct:.0f}%")
                
            except Exception as e:
                print(f"❌ Error in polling cycle: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Wait for next poll with active checking
            if polling_active:
                print(f"⏳ Waiting {POLL_INTERVAL_MINUTES} minutes until next poll...")
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

# Main page with auto-start
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
            button { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #2563eb; }
            button:disabled { background: #64748b; cursor: not-allowed; }
            .inverter { border: 1px solid #334155; padding: 10px; margin: 5px 0; border-radius: 5px; }
            .primary { border-left: 4px solid #3b82f6; }
            .backup { border-left: 4px solid #f59e0b; }
            .status-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-left: 10px; }
            .status-active { background: #10b981; color: white; }
            .status-inactive { background: #64748b; color: white; }
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
                    <button onclick="refreshData()">Refresh Now</button>
                    <button onclick="location.reload()">Reload Page</button>
                </div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>⚡ Current Load</h3>
                    <div class="metric" id="load">0 W</div>
                    <div id="load-detail">Total power consumption</div>
                </div>
                
                <div class="card">
                    <h3>☀️ Solar Output</h3>
                    <div class="metric success" id="solar">0 W</div>
                    <div id="solar-detail">Total solar generation</div>
                </div>
                
                <div class="card">
                    <h3>🔋 Primary Battery</h3>
                    <div class="metric" id="battery">0 %</div>
                    <div id="battery-detail">Minimum capacity</div>
                </div>
                
                <div class="card">
                    <h3>🔄 Usable Energy</h3>
                    <div class="metric info" id="usable">0 %</div>
                    <div id="usable-detail"><span id="usable-kwh">0</span> kWh available</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Inverters <span id="inverter-count">(0)</span></h2>
                <div id="inverters">Loading inverter data...</div>
            </div>
            
            <div class="card">
                <h2>System Info</h2>
                <div class="grid">
                    <div>
                        <h3>Backup System</h3>
                        <p>Voltage: <span id="backup-voltage">0</span> V</p>
                        <p>Status: <span id="backup-status">Inactive</span></p>
                    </div>
                    <div>
                        <h3>Polling</h3>
                        <p>Interval: <span id="poll-interval">5</span> minutes</p>
                        <p>Last Success: <span id="last-success">--:--:--</span></p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Quick Links & Diagnostics</h2>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    <a href="/test"><button>Test Endpoint</button></a>
                    <a href="/debug"><button>Debug Info</button></a>
                    <a href="/api/data"><button>API Data</button></a>
                    <a href="/test-fetch"><button>Test API Call</button></a>
                    <a href="/polling-debug"><button>Polling Debug</button></a>
                    <a href="/start-polling"><button>Start Polling (API)</button></a>
                    <a href="/stop-polling"><button>Stop Polling (API)</button></a>
                </div>
            </div>
        </div>
        
        <script>
            let pollingInterval = null;
            let lastUpdateTime = null;
            
            function updateUI(data) {
                // Update basic metrics
                document.getElementById('status').innerText = data.status;
                document.getElementById('timestamp').innerText = data.timestamp;
                document.getElementById('load').innerText = Math.round(data.total_output_power) + ' W';
                document.getElementById('solar').innerText = Math.round(data.total_solar_input_W) + ' W';
                document.getElementById('battery').innerText = Math.round(data.primary_battery_min) + ' %';
                document.getElementById('usable').innerText = Math.round(data.usable_energy.total_pct) + ' %';
                document.getElementById('usable-kwh').innerText = data.usable_energy.total_kwh;
                
                // Update backup info
                document.getElementById('backup-voltage').innerText = data.backup_battery_voltage.toFixed(1);
                const backupStatusEl = document.getElementById('backup-status');
                if (data.backup_active) {
                    backupStatusEl.innerText = 'ACTIVE';
                    backupStatusEl.className = 'status-active status-badge';
                } else {
                    backupStatusEl.innerText = 'INACTIVE';
                    backupStatusEl.className = 'status-inactive status-badge';
                }
                
                // Update inverters
                let invertersHtml = '';
                data.inverters.forEach(inv => {
                    const invClass = inv.Type === 'backup' ? 'backup' : 'primary';
                    invertersHtml += `
                        <div class="inverter ${invClass}">
                            <strong>${inv.Label}</strong> 
                            <span class="status-badge ${inv.has_fault ? 'error' : 'success'}">
                                ${inv.has_fault ? 'FAULT' : 'OK'}
                            </span><br>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 5px;">
                                <div>Output: <strong>${Math.round(inv.OutputPower)}W</strong></div>
                                <div>Battery: <strong>${Math.round(inv.Capacity)}%</strong></div>
                                <div>Solar: <strong>${Math.round(inv.ppv)}W</strong></div>
                            </div>
                            <div style="font-size: 0.9em; color: #94a3b8; margin-top: 5px;">
                                Voltage: ${inv.vBat.toFixed(2)}V | 
                                Status: ${inv.Status}
                            </div>
                        </div>
                    `;
                });
                
                document.getElementById('inverters').innerHTML = invertersHtml || '<div class="inverter">No inverter data available</div>';
                document.getElementById('inverter-count').innerText = `(${data.inverters.length})`;
                
                // Update last success time
                if (data.inverters.length > 0) {
                    lastUpdateTime = new Date();
                    document.getElementById('last-success').innerText = lastUpdateTime.toLocaleTimeString();
                }
            }
            
            function refreshData() {
                fetch('/api/data')
                    .then(r => {
                        if (!r.ok) throw new Error(`HTTP ${r.status}`);
                        return r.json();
                    })
                    .then(data => {
                        updateUI(data);
                        document.getElementById('status').className = 'success';
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
                    })
                    .catch(err => {
                        console.error('Error checking polling status:', err);
                    });
            }
            
            function startPolling() {
                fetch('/start-polling')
                    .then(r => r.json())
                    .then(data => {
                        alert('Polling started: ' + (data.status || data.error));
                        checkPollingStatus();
                        setTimeout(refreshData, 3000); // Wait 3 seconds then refresh
                    })
                    .catch(err => {
                        alert('Error starting polling: ' + err.message);
                    });
            }
            
            function stopPolling() {
                fetch('/stop-polling')
                    .then(r => r.json())
                    .then(data => {
                        alert('Polling stop signal sent');
                        checkPollingStatus();
                    })
                    .catch(err => {
                        alert('Error stopping polling: ' + err.message);
                    });
            }
            
            // Auto-start polling on page load
            function autoStartPolling() {
                fetch('/test')
                    .then(r => r.json())
                    .then(data => {
                        if (!data.polling_thread_alive) {
                            console.log('Auto-starting polling...');
                            startPolling();
                        } else {
                            console.log('Polling already active');
                            refreshData();
                        }
                    });
            }
            
            // Initial load
            document.addEventListener('DOMContentLoaded', function() {
                refreshData();
                checkPollingStatus();
                autoStartPolling();
                
                // Auto-refresh every 30 seconds
                setInterval(refreshData, 30000);
                setInterval(checkPollingStatus, 10000);
                
                // Update polling interval display
                document.getElementById('poll-interval').innerText = 5;
            });
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
    print(f"🔧 Polling Interval: {POLL_INTERVAL_MINUTES} minutes")
    
    # Don't auto-start polling - let the web interface control it
    print("ℹ️  Polling will be started from the web interface")
    
    # Start Flask
    print(f"🌐 Starting Flask on 0.0.0.0:{port}")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )
