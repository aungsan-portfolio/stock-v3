"""
dashboard.py — Stock Engine Pro V3 GUI Dashboard
Run: python dashboard.py
Open: http://localhost:5050
"""
import subprocess, sys, os, json, time
from pathlib import Path

try:
    from flask import Flask, request, jsonify, send_file, Response
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "--break-system-packages"])
    from flask import Flask, request, jsonify, send_file, Response

BASE_DIR = Path(__file__).parent
app = Flask(__name__)

# Read HTML ONCE at startup
HTML_PATH = BASE_DIR / "dashboard.html"
HTML_CONTENT = HTML_PATH.read_text(encoding="utf-8") if HTML_PATH.exists() else "<h1>dashboard.html not found</h1>"

@app.route("/")
def index():
    return Response(HTML_CONTENT, content_type="text/html")

@app.route("/api/portfolio")
def api_portfolio():
    try:
        gp = BASE_DIR / "get_portfolio.py"
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(gp)],
            cwd=str(BASE_DIR), capture_output=True, text=True, timeout=30
        )
        return jsonify(json.loads(proc.stdout))
    except Exception as e:
        return jsonify({"cash": 0, "net_liquidation": 0, "positions": [], "orders": [], "error": str(e)})

@app.route("/api/hot")
def api_hot():
    csv_path = BASE_DIR / "reports" / "hot_candidates.csv"
    if not csv_path.exists():
        return jsonify({"headers": [], "rows": []})
    lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return jsonify({"headers": [], "rows": []})
    hdrs = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        cols = []
        cur = ""
        inq = False
        for c in line:
            if c == '"': inq = not inq
            elif c == "," and not inq: cols.append(cur); cur = ""
            else: cur += c
        cols.append(cur)
        rows.append(cols)
    return jsonify({"headers": hdrs, "rows": rows[:30]})

@app.route("/api/models")
def api_models():
    rf, lstm, thresh = 0, 0, "?"
    try:
        import joblib
        rf_file = BASE_DIR / "models" / "rf_models.joblib"
        if rf_file.exists():
            d = joblib.load(rf_file)
            rf = len(d) if d else 0
    except: pass
    try:
        import torch
        lstm_file = BASE_DIR / "models" / "lstm_checkpoint.pt"
        if lstm_file.exists():
            ckpt = torch.load(str(lstm_file), map_location="cpu")
            lstm = len(ckpt.get("state_dicts", {}))
    except: pass
    try:
        sys.path.insert(0, str(BASE_DIR))
        import config
        thresh = str(config.BUY_THRESHOLD)
    except: pass
    return jsonify({"rf": rf, "lstm": lstm, "threshold": thresh})

@app.route("/api/reports")
def api_reports():
    reports_dir = BASE_DIR / "reports"
    files = []
    if reports_dir.exists():
        for f in sorted(reports_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            if f.is_file() and f.suffix in (".csv",".jsonl",".json",".txt",".md",".html",".log"):
                sz = f.stat().st_size
                sz_str = f"{sz/1024:.1f}KB" if sz > 1024 else f"{sz}B"
                mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
                files.append({"name": f.name, "size": sz_str, "modified": mt})
    return jsonify(files)

@app.route("/report/<path:name>")
def view_report(name):
    safe = (BASE_DIR / "reports" / name).resolve()
    if safe.exists() and str(safe).startswith(str((BASE_DIR / "reports").resolve())):
        return send_file(str(safe))
    return "Not found", 404

@app.route("/api/cancel")
def api_cancel():
    cp = BASE_DIR / "cancel_open_orders.py"
    if cp.exists():
        proc = subprocess.run([sys.executable, "-X", "utf8", str(cp)], cwd=str(BASE_DIR), capture_output=True, text=True, timeout=30)
        return jsonify({"output": (proc.stdout + proc.stderr).strip(), "exit": proc.returncode})
    return jsonify({"output": "cancel_open_orders.py not found", "exit": -1})

@app.route("/api/run/<cmd>")
@app.route("/api/run/<cmd>/<path:args>")
def api_run(cmd, args=""):
    args_list = args.split("/") if args else []
    full_args = [sys.executable, "-X", "utf8", str(BASE_DIR / "main.py"), cmd] + args_list
    try:
        proc = subprocess.run(full_args, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=600)
        output = (proc.stdout + "\n" + proc.stderr).strip()
        return jsonify({"output": output, "exit": proc.returncode})
    except subprocess.TimeoutExpired:
        return jsonify({"output": "Command timed out", "exit": -1})
    except Exception as e:
        return jsonify({"output": f"Error: {e}", "exit": -1})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Dashboard: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
