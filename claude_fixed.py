import os
import time
import threading
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, render_template_string
import requests

app = Flask(__name__)

# Configuration
TOKEN = os.getenv("API_TOKEN", "")
SERIAL_NUMBERS = os.getenv("SERIAL_NUMBERS", "RKG3B0400T,KAM4N5W0AG,JNK1CDR0KQ").split(",")
API_URL = "https://openapi.growatt.com/v1/device/storage/storage_last_data"

print(f"🔧 Configuration: TOKEN={'SET' if TOKEN else 'NOT SET'}, SERIALS={len(SERIAL_NUMBERS)}")

# Global data storage
latest_data = {
    "timestamp": datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S"),
    "status": "Starting up...",
    "inverters": []
}

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
        "port": os.getenv("PORT", "10000")
    })

# Debug endpoint
@app.route('/debug')
def debug():
    threads = []
    for thread in threading.enumerate():
        threads.append({
            "name": thread.name,
            "alive": thread.is_alive(),
            "daemon": thread.daemon
        })
    
    return jsonify({
        "app": "Solar Monitor",
        "timestamp": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "environment": {
            "API_TOKEN_set": bool(TOKEN),
            "SERIAL_NUMBERS": SERIAL_NUMBERS,
            "PORT": os.getenv("PORT", "10000")
        },
        "threads": threads,
        "thread_count": threading.active_count(),
        "latest_data": latest_data
    })

# Simple API data endpoint
@app.route('/api/data')
def api_data():
    return jsonify(latest_data)

# Test inverter API call
@app.route('/test-fetch')
def test_fetch():
    """Test the Growatt API directly"""
    if not TOKEN:
        return jsonify({"error": "API_TOKEN not set in environment variables"}), 400
    
    if len(SERIAL_NUMBERS) == 0:
        return jsonify({"error": "SERIAL_NUMBERS not set"}), 400
    
    try:
        # Test with first inverter
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
        <style>
            body { font-family: Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .card { background: #1e293b; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .success { color: #10b981; }
            .error { color: #ef4444; }
            .warning { color: #f59e0b; }
            a { color: #3b82f6; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌞 Solar Monitor</h1>
            <div class="card">
                <h2>Status</h2>
                <p id="status">Loading...</p>
                <p>Timestamp: <span id="timestamp">--:--:--</span></p>
            </div>
            
            <div class="card">
                <h2>Quick Links</h2>
                <ul>
                    <li><a href="/test">Test Endpoint</a></li>
                    <li><a href="/debug">Debug Info</a></li>
                    <li><a href="/api/data">API Data</a></li>
                    <li><a href="/test-fetch">Test Growatt API</a></li>
                    <li><a href="/health">Health Check</a></li>
                </ul>
            </div>
            
            <div class="card">
                <h2>Environment Check</h2>
                <p>API Token: <span id="token-status" class="warning">Checking...</span></p>
                <p>Serial Numbers: <span id="serial-count">0</span></p>
            </div>
        </div>
        
        <script>
            // Load initial data
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').innerText = data.status;
                    document.getElementById('timestamp').innerText = data.timestamp;
                });
            
            // Check environment
            fetch('/test')
                .then(r => r.json())
                .then(data => {
                    const tokenEl = document.getElementById('token-status');
                    tokenEl.innerText = data.api_token_set ? '✓ SET' : '✗ NOT SET';
                    tokenEl.className = data.api_token_set ? 'success' : 'error';
                    
                    document.getElementById('serial-count').innerText = data.serial_numbers.length;
                })
                .catch(err => {
                    console.error('Error:', err);
                });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

# Simple polling function (will add later)
def poll_growatt_simple():
    """Simple polling function for testing"""
    print("🔄 Simple polling function started")
    
    if not TOKEN:
        print("❌ No API_TOKEN, polling disabled")
        return
    
    while True:
        try:
            now = datetime.now(timezone(timedelta(hours=3)))
            print(f"🔄 Polling at {now.strftime('%H:%M:%S')}")
            
            # Update latest_data
            latest_data["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
            latest_data["status"] = f"Last polled at {now.strftime('%H:%M:%S')}"
            
        except Exception as e:
            print(f"❌ Polling error: {e}")
        
        time.sleep(60)  # Poll every minute

# Startup
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    
    print(f"🚀 Starting Solar Monitor on port {port}")
    
    # Start simple polling thread
    try:
        poll_thread = threading.Thread(
            target=poll_growatt_simple,
            name="polling_thread",
            daemon=True
        )
        poll_thread.start()
        print(f"✅ Started polling thread: {poll_thread.name}")
    except Exception as e:
        print(f"❌ Failed to start polling thread: {e}")
    
    # Start Flask
    print(f"🌐 Starting Flask on 0.0.0.0:{port}")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )
