#!/usr/bin/env python3
"""
Sales Forecast API  -  single-file edition (v2)
===============================================
Short-term monthly revenue forecast (1 to 3 months ahead) served from a
scikit-learn Simple Linear Regression model (feature: previous month's revenue).

HOW TO RUN
  1. Put this file and sales_forecast_model.pkl in the same folder.
  2. Run:      python app.py
  3. Browser opens at   http://127.0.0.1:8000        (API docs: /docs)

Why only 3 months?
  The model predicts each month from the previous one, so beyond ~3 steps its
  output flattens to a constant level and carries no real information.
  Requests further out are rejected with a clear message. To forecast a later
  period, send your latest ACTUAL revenue (base_month + base_revenue).

Optional environment variables
  MODEL_PATH   path to a .pkl file (default: sales_forecast_model.pkl next to this file)
  HOST / PORT  default 127.0.0.1 / 8000
  API_KEY      if set, uploading a new model requires the header  X-API-Key: <value>
  NO_BROWSER   set to 1 to stop the browser from opening automatically
"""
# ----------------------------------------------------------------------------
# 0. Auto-install missing dependencies
# ----------------------------------------------------------------------------
import importlib.util
import subprocess
import sys

_REQUIRED = {"fastapi": "fastapi", "uvicorn": "uvicorn", "joblib": "joblib",
             "sklearn": "scikit-learn", "pandas": "pandas", "numpy": "numpy"}
_missing = [pip for mod, pip in _REQUIRED.items() if importlib.util.find_spec(mod) is None]
if _missing:
    print(f"[setup] Installing missing packages: {', '.join(_missing)} ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_missing],
                              stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("\n[setup] Automatic install was blocked by your Python setup.\n"
              "        Please run this once, then start the app again:\n\n"
              f"            {sys.executable} -m pip install {' '.join(_missing)}\n")
        sys.exit(1)

# ----------------------------------------------------------------------------
# 1. Imports & configuration
# ----------------------------------------------------------------------------
import io
import os
import re
import socket
import tempfile
import threading
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "sales_forecast_model.pkl"
CURRENCY = "BRL"
MAX_HORIZON = 3                 # hard limit: months ahead of the last actual month
STALE_AFTER_MONTHS = 6          # warn when the last actual month is older than this
MAX_UPLOAD_MB = 50
Z95 = 1.96
HOST = os.getenv("HOST", "127.0.0.1")
API_KEY = os.getenv("API_KEY")
IS_LOCAL = HOST in ("127.0.0.1", "localhost", "::1")
UPLOAD_ENABLED = IS_LOCAL or bool(API_KEY)   # never accept pickles from a network without a key


# ----------------------------------------------------------------------------
# 2. Model store: finds, validates and (hot-)reloads the .pkl
# ----------------------------------------------------------------------------
class ModelStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.artifact: Optional[dict] = None
        self.path: Optional[Path] = None
        self.loaded_at: Optional[str] = None
        self.sigma: Optional[float] = None       # residual std used for the 95% interval

    @staticmethod
    def find_pkl() -> Optional[Path]:
        env = os.getenv("MODEL_PATH")
        if env and Path(env).is_file():
            return Path(env)
        p = APP_DIR / DEFAULT_MODEL_NAME
        if p.is_file():
            return p
        pkls = sorted(APP_DIR.glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)
        return pkls[0] if pkls else None

    @staticmethod
    def validate(art) -> dict:
        if not isinstance(art, dict):
            raise ValueError("The file is not a model package created by the notebook "
                             "(expected a dictionary with 'model', 'first_month', ...).")
        for key in ("model", "first_month", "last_training_month", "last_actual"):
            if art.get(key) is None:
                raise ValueError(f"Missing required key '{key}' in the .pkl file.")
        model = art["model"]
        if not hasattr(model, "predict") or not hasattr(model, "coef_"):
            raise ValueError("The 'model' inside the file must be a fitted linear model.")
        if list(art.get("features", ["lag_1"])) != ["lag_1"]:
            raise ValueError(f"Unsupported features {art.get('features')}. "
                             "This API serves the lag_1 Simple Linear Regression model.")
        art["first_month"] = pd.Timestamp(art["first_month"]).to_period("M").to_timestamp()
        art["last_training_month"] = pd.Timestamp(art["last_training_month"]).to_period("M").to_timestamp()
        art["last_actual"] = float(art["last_actual"])
        return art

    @staticmethod
    def derive_sigma(art) -> Optional[float]:
        if art.get("residual_std") is not None:
            return float(art["residual_std"])
        f = art.get("forecast_next_month") or {}
        try:
            return max((float(f["upper_95"]) - float(f["lower_95"])) / (2 * Z95), 0.0)
        except (KeyError, TypeError, ValueError):
            return None

    def load(self, path: Path):
        try:
            raw = joblib.load(path)
        except Exception:
            raise ValueError("This is not a valid .pkl model file (it could not be read). "
                             "Use the file produced by the notebook: sales_forecast_model.pkl.")
        art = self.validate(raw)
        with self.lock:
            self.artifact, self.path = art, Path(path)
            self.sigma = self.derive_sigma(art)
            self.loaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def load_default(self) -> bool:
        p = self.find_pkl()
        if not p:
            return False
        try:
            self.load(p)
            return True
        except Exception as e:                      # keep the server alive, show the message in the UI
            print(f"[model] Could not load {p.name}: {e}")
            return False


store = ModelStore()


def require_model() -> dict:
    if store.artifact is None:
        raise HTTPException(503, f"No model loaded. Put '{DEFAULT_MODEL_NAME}' next to app.py "
                                 "or upload it from the dashboard (POST /api/model/upload).")
    return store.artifact


# ----------------------------------------------------------------------------
# 3. Forecast logic
# ----------------------------------------------------------------------------
def parse_month(s: str) -> pd.Timestamp:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", s or ""):
        raise HTTPException(422, f"Invalid month '{s}'. Use the format YYYY-MM, e.g. 2018-09.")
    year = int(s[:4])
    if not 1990 <= year <= 2100:
        raise HTTPException(422, "Year must be between 1990 and 2100.")
    return pd.Timestamp(year=year, month=int(s[5:]), day=1)


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def get_anchor(art: dict, base_month: Optional[str], base_revenue: Optional[float]):
    """Last KNOWN actual month and its revenue: the one saved in the model, or one supplied by the caller."""
    if (base_month is None) != (base_revenue is None):
        raise HTTPException(422, "Provide both base_month and base_revenue, or neither.")
    if base_month is None:
        return art["last_training_month"], art["last_actual"]
    ts = parse_month(base_month)
    if ts < art["first_month"]:
        raise HTTPException(422, f"base_month is before the first training month ({art['first_month']:%Y-%m}).")
    if base_revenue < 0:
        raise HTTPException(422, "base_revenue must be zero or positive.")
    return ts, float(base_revenue)


def check_ahead(base_ts: pd.Timestamp, ts: pd.Timestamp) -> int:
    """Return how many months `ts` is after the anchor, or raise a clear error."""
    ahead = months_between(base_ts, ts)
    first, last = base_ts + pd.DateOffset(months=1), base_ts + pd.DateOffset(months=MAX_HORIZON)
    if ahead < 1:
        raise HTTPException(422, f"This model forecasts months AFTER the last actual month "
                                 f"({base_ts:%Y-%m}). Try {first:%Y-%m} or later.")
    if ahead > MAX_HORIZON:
        raise HTTPException(422, f"Forecasts are limited to {MAX_HORIZON} months after the last actual month "
                                 f"({base_ts:%Y-%m}), i.e. {first:%Y-%m} to {last:%Y-%m}. For a later period, "
                                 "send base_month and base_revenue with your latest actual revenue.")
    return ahead


def forecast_path(art: dict, base_value: float, steps: int) -> List[float]:
    """Recursive multi-step forecast: each month is predicted from the previous one."""
    out, v = [], float(base_value)
    for _ in range(steps):
        v = float(art["model"].predict(pd.DataFrame({"lag_1": [v]}))[0])
        out.append(v)
    return out


def make_row(art: dict, ts: pd.Timestamp, ahead: int, value: float) -> dict:
    coef = float(art["model"].coef_[0])
    tm = art.get("test_metrics") or {}
    rec = art.get("test_metrics_recursive") or tm
    mae = (tm if ahead == 1 else rec).get("MAE")
    sigma_k = store.sigma * sum(coef ** (2 * i) for i in range(ahead)) ** 0.5 if store.sigma else None
    row = {"month": f"{ts:%Y-%m}", "forecast": round(max(value, 0.0), 2), "months_ahead": ahead,
           "error_range_low": None, "error_range_high": None, "ci95_low": None, "ci95_high": None}
    if mae is not None:        # typical error: +/- hold-out MAE
        row["error_range_low"] = round(max(value - mae, 0.0), 2)
        row["error_range_high"] = round(value + mae, 2)
    if sigma_k:                # statistical 95% interval from training residuals (wider, conservative)
        row["ci95_low"] = round(max(value - Z95 * sigma_k, 0.0), 2)
        row["ci95_high"] = round(value + Z95 * sigma_k, 2)
    return row


def forecast_range(art: dict, start: Optional[str], months: int, anchor) -> List[dict]:
    base_ts, base_val = anchor
    first = parse_month(start) if start else base_ts + pd.DateOffset(months=1)
    ahead0 = check_ahead(base_ts, first)
    check_ahead(base_ts, first + pd.DateOffset(months=months - 1))          # last month must be in range too
    path = forecast_path(art, base_val, ahead0 + months - 1)[ahead0 - 1:]
    return [make_row(art, first + pd.DateOffset(months=i), ahead0 + i, path[i]) for i in range(months)]


def anchor_json(anchor) -> dict:
    return {"month": f"{anchor[0]:%Y-%m}", "revenue": round(anchor[1], 2)}


def months_old(anchor) -> int:
    now = pd.Timestamp.now()
    return months_between(anchor[0], pd.Timestamp(year=now.year, month=now.month, day=1))


def warnings_for(art: dict, anchor) -> List[str]:
    out = []
    if months_old(anchor) > STALE_AFTER_MONTHS:
        out.append(f"The last actual month ({anchor[0]:%Y-%m}) is more than {STALE_AFTER_MONTHS} months old. "
                   "These forecasts describe the months right after it, not the current period. "
                   "Send base_month and base_revenue with your latest actual revenue.")
    return out


def response_for(art: dict, anchor, rows: List[dict]) -> dict:
    return {"currency": CURRENCY, "count": len(rows), "anchor": anchor_json(anchor), "forecasts": rows,
            "warnings": warnings_for(art, anchor)}


# ----------------------------------------------------------------------------
# 4. API
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    ok = store.load_default()
    print("[model] " + (f"Loaded {store.path.name}" if ok else
                        f"No model yet - put {DEFAULT_MODEL_NAME} next to app.py or upload it in the dashboard."))
    yield


app = FastAPI(
    title="Sales Forecast API",
    version="2.0.0",
    description=f"Short-term (up to {MAX_HORIZON} months ahead) monthly revenue forecasting, "
                "backed by a scikit-learn Simple Linear Regression model.",
    lifespan=lifespan,
)


class BatchRequest(BaseModel):
    months: List[str] = Field(..., min_length=1, max_length=MAX_HORIZON,
                              examples=[["2018-09", "2018-10", "2018-11"]])
    base_month: Optional[str] = Field(None, examples=["2018-08"])
    base_revenue: Optional[float] = Field(None, examples=[848860])


BASE_M = Query(None, description="Latest month whose ACTUAL revenue you know (YYYY-MM). Optional.", examples=["2018-08"])
BASE_R = Query(None, description=f"Actual revenue of base_month in {CURRENCY}. Send together with base_month.", examples=[848860])


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/health", tags=["System"], summary="Service health check")
def health():
    return {"status": "ok", "model_loaded": store.artifact is not None}


@app.get("/api/model", tags=["Model"], summary="Model details and metrics")
def model_info():
    upload = {"enabled": UPLOAD_ENABLED, "needs_key": bool(API_KEY), "max_mb": MAX_UPLOAD_MB}
    art = store.artifact
    if art is None:
        return {"loaded": False, "expected_file": DEFAULT_MODEL_NAME, "folder": str(APP_DIR), "upload": upload}
    m = art["model"]
    b, a = float(m.coef_[0]), float(getattr(m, "intercept_", 0.0))
    anchor = get_anchor(art, None, None)
    return {
        "loaded": True,
        "file": store.path.name,
        "file_size_kb": round(store.path.stat().st_size / 1024, 1),
        "loaded_at": store.loaded_at,
        "algorithm": type(m).__name__,
        "target": art.get("target", "revenue"),
        "currency": CURRENCY,
        "equation": {"intercept": a, "lag_coefficient": b,
                     "long_run_level": a / (1 - b) if b < 1 else None},
        "anchor": anchor_json(anchor),
        "data_age_months": months_old(anchor),
        "training": {"first_month": f"{art['first_month']:%Y-%m}",
                     "last_month": f"{art['last_training_month']:%Y-%m}",
                     "n_months": months_between(art["first_month"], art["last_training_month"]) + 1},
        "holdout_metrics": art.get("test_metrics"),
        "holdout_metrics_recursive": art.get("test_metrics_recursive"),
        "limits": {"max_horizon_months": MAX_HORIZON, "stale_after_months": STALE_AFTER_MONTHS},
        "upload": upload,
    }


@app.get("/api/forecast", tags=["Forecast"], summary=f"Forecast for one month (up to {MAX_HORIZON} months ahead)")
def forecast_month(month: str = Query(..., description="Month in YYYY-MM format", examples=["2018-09"]),
                   base_month: Optional[str] = BASE_M, base_revenue: Optional[float] = BASE_R):
    art = require_model()
    anchor = get_anchor(art, base_month, base_revenue)
    rows = forecast_range(art, month, 1, anchor)
    r = response_for(art, anchor, rows)
    return {**rows[0], "currency": CURRENCY, "anchor": r["anchor"], "warnings": r["warnings"]}


@app.get("/api/forecast/range", tags=["Forecast"],
         summary=f"Forecast for the next N months (N up to {MAX_HORIZON}), starting right after the last actual month")
def forecast_months(start: Optional[str] = Query(None, description="First month, YYYY-MM (optional)", examples=["2018-09"]),
                    months: int = Query(MAX_HORIZON, ge=1, le=MAX_HORIZON),
                    base_month: Optional[str] = BASE_M, base_revenue: Optional[float] = BASE_R):
    art = require_model()
    anchor = get_anchor(art, base_month, base_revenue)
    return response_for(art, anchor, forecast_range(art, start, months, anchor))


@app.get("/api/forecast/range.csv", tags=["Forecast"], summary="Download the range forecast as CSV (opens in Excel)")
def forecast_csv(start: Optional[str] = None, months: int = Query(MAX_HORIZON, ge=1, le=MAX_HORIZON),
                 base_month: Optional[str] = BASE_M, base_revenue: Optional[float] = BASE_R):
    art = require_model()
    df = pd.DataFrame(forecast_range(art, start, months, get_anchor(art, base_month, base_revenue)))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="sales_forecast_{df.month.iloc[0]}_{months}m.csv"'})


@app.post("/api/forecast/batch", tags=["Forecast"], summary=f"Forecast for a list of up to {MAX_HORIZON} months")
def forecast_batch(body: BatchRequest):
    art = require_model()
    anchor = get_anchor(art, body.base_month, body.base_revenue)
    rows = [forecast_range(art, m, 1, anchor)[0] for m in body.months]
    return response_for(art, anchor, rows)


@app.post("/api/model/upload", tags=["Model"], summary="Replace the model: send the raw .pkl bytes as the request body")
async def upload_model(request: Request, x_api_key: Optional[str] = Header(None)):
    if not UPLOAD_ENABLED:
        raise HTTPException(403, "Model upload is disabled when the server is exposed on a network "
                                 "without an API_KEY (pickle files can run code).")
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Missing or wrong X-API-Key header.")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty file.")
    if len(body) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB).")

    fd, tmp = tempfile.mkstemp(suffix=".pkl", dir=APP_DIR)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        ModelStore().load(Path(tmp))                            # validate BEFORE replacing anything
        os.replace(tmp, APP_DIR / DEFAULT_MODEL_NAME)
        store.load(APP_DIR / DEFAULT_MODEL_NAME)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise HTTPException(400, f"Rejected - the previous model is still active. Reason: {e}")
    return {"ok": True, "message": "Model uploaded and activated.", "info": model_info()}


@app.exception_handler(500)
async def _server_error(_, exc):
    return JSONResponse({"detail": f"Unexpected server error: {exc}"}, status_code=500)


# ----------------------------------------------------------------------------
# 5. Dashboard (served from this same file - no extra files needed)
# ----------------------------------------------------------------------------
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sales Forecast</title>
<meta name="color-scheme" content="light dark">
<style>
:root{--bg:#f5f7fb;--card:#fff;--ink:#0f172a;--mute:#64748b;--line:#e2e8f0;--brand:#4f46e5;--brand2:#7c3aed;
--ok:#16a34a;--warn:#d97706;--bad:#dc2626;--okbg:#dcfce7;--warnbg:#fef3c7;--badbg:#fee2e2;--shadow:0 1px 3px rgba(15,23,42,.08),0 8px 24px rgba(15,23,42,.05)}
@media(prefers-color-scheme:dark){:root{--bg:#0b1020;--card:#131a2e;--ink:#e6eaf5;--mute:#93a0bd;--line:#232c47;--okbg:#10331f;--warnbg:#3a2a0b;--badbg:#3b1416;--shadow:none}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Segoe UI",system-ui,-apple-system,Tahoma,Arial,sans-serif}
header{background:linear-gradient(120deg,var(--brand),var(--brand2));color:#fff;padding:22px 0}
.wrap{max-width:1100px;margin:0 auto;padding:0 20px}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
h1{margin:0;font-size:22px;letter-spacing:.2px}.sub{opacity:.85;font-size:13px}
.tools{display:flex;gap:8px;align-items:center}
.pill{padding:5px 12px;border-radius:99px;font-size:12px;font-weight:600;background:rgba(255,255,255,.18);display:inline-flex;gap:6px;align-items:center}
.dot{width:8px;height:8px;border-radius:50%;background:#fca5a5}.dot.on{background:#86efac}
button,.btn{font:inherit;cursor:pointer;border:0;border-radius:10px;padding:9px 16px;font-weight:600;background:var(--brand);color:#fff;text-decoration:none;display:inline-block}
button:hover{filter:brightness(1.08)}button.ghost{background:transparent;color:inherit;border:1px solid var(--line)}
header button.ghost{border-color:rgba(255,255,255,.4);padding:5px 12px;font-size:12px}
main{padding:22px 0 50px}.grid{display:grid;gap:16px}.kpis{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:var(--shadow)}
.kpi .l{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.6px}.kpi .v{font-size:24px;white-space:nowrap;font-weight:700;margin-top:4px}.kpi .s{color:var(--mute);font-size:12.5px;margin-top:2px}
h2{font-size:16px;margin:0 0 12px}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end}
label{display:block;font-size:12px;color:var(--mute);margin-bottom:4px}
select,input[type=month],input[type=number],input[type=password]{font:inherit;padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--ink)}
.two{grid-template-columns:2fr 1fr}@media(max-width:860px){.two{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px 10px;text-align:start;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.5px}.tw{overflow-x:auto}td.n{font-variant-numeric:tabular-nums}
.badge{padding:3px 10px;border-radius:99px;font-size:12px;font-weight:700}.high{background:var(--okbg);color:var(--ok)}.medium{background:var(--warnbg);color:var(--warn)}.low{background:var(--badbg);color:var(--bad)}.unknown{background:var(--line);color:var(--mute)}
.alert{padding:12px 14px;border-radius:12px;font-size:14px;margin-bottom:16px;border:1px solid transparent}
.alert.warn{background:var(--warnbg);color:var(--warn);border-color:var(--warn)}.alert.bad{background:var(--badbg);color:var(--bad);border-color:var(--bad)}.alert.note{background:var(--card);color:var(--mute);border-color:var(--line)}.alert.good{background:var(--okbg);color:var(--ok);border-color:var(--ok)}
.drop{border:2px dashed var(--line);border-radius:14px;padding:22px;text-align:center;color:var(--mute);cursor:pointer;transition:.15s}
.drop.over,.drop:hover{border-color:var(--brand);color:var(--brand);background:rgba(79,70,229,.06)}
dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:13.5px}dt{color:var(--mute)}dd{margin:0;text-align:end;font-variant-numeric:tabular-nums}
pre{background:#0f172a;color:#e2e8f0;padding:12px 14px;border-radius:12px;overflow:auto;font-size:12.5px;margin:0 0 10px;direction:ltr;text-align:left;position:relative}
.copy{position:absolute;top:6px;right:6px;padding:3px 9px;font-size:11px;background:#334155}
.anchor{margin-top:14px;padding:12px 14px;border:1px dashed var(--line);border-radius:12px}.anchor .s{color:var(--mute);font-size:12.5px;margin:2px 0 10px}
.hidden{display:none!important}footer{text-align:center;color:var(--mute);font-size:12px;padding:10px}
canvas{max-height:340px}.lookup{margin-top:14px;padding-top:14px;border-top:1px solid var(--line)}
</style>
</head>
<body>
<header><div class="wrap top">
  <div><h1 data-i="title">Sales Forecast</h1><div class="sub" data-i="subtitle"></div></div>
  <div class="tools">
    <span class="pill"><span class="dot" id="dot"></span><span id="status"></span></span>
    <button class="ghost" id="langBtn">العربية</button>
    <a class="btn ghost" style="padding:5px 12px;font-size:12px;border:1px solid rgba(255,255,255,.4)" href="/docs" target="_blank">API docs</a>
  </div>
</div></header>

<main class="wrap">
  <div id="msg"></div>

  <section id="onboard" class="card hidden" style="margin-bottom:16px">
    <h2 data-i="ob_title"></h2><p data-i="ob_text" style="color:var(--mute);margin-top:0"></p>
  </section>

  <section id="app" class="hidden">
    <div class="grid kpis">
      <div class="card kpi"><div class="l" id="k1l"></div><div class="v" id="k1v">-</div><div class="s" id="k1s"></div></div>
      <div class="card kpi"><div class="l" data-i="k_range"></div><div class="v" id="k2v">-</div><div class="s" data-i="k_range_s"></div></div>
      <div class="card kpi"><div class="l" id="k3l"></div><div class="v" id="k3v">-</div><div class="s" id="k3s"></div></div>
    </div>

    <div class="grid two" style="margin-top:0">
      <div class="card">
        <div class="row" style="justify-content:space-between">
          <h2 data-i="fc_title" style="margin:0"></h2>
          <div class="row">
            <div><label data-i="horizon"></label>
              <select id="horizon"><option value="1">1</option><option value="2">2</option><option value="3" selected>3</option></select></div>
            <button id="runBtn" data-i="run"></button>
            <a class="btn ghost" id="csvBtn" data-i="csv" href="#"></a>
          </div>
        </div>
        <div id="anchorBox" class="anchor hidden">
          <b data-i="an_title"></b><div class="s" data-i="an_text"></div>
          <div class="row"><div><label data-i="an_month"></label><input type="month" id="anMonth"></div>
            <div><label data-i="an_rev"></label><input type="number" id="anRev" min="0" step="1000" style="width:170px"></div>
            <button id="anBtn" data-i="an_btn"></button><button id="anReset" class="ghost hidden" data-i="an_reset"></button></div>
          <div id="anState"></div>
        </div>
        <div style="margin-top:14px"><canvas id="chart"></canvas></div>
        <div class="tw" style="margin-top:14px"><table>
          <thead><tr><th data-i="t_month"></th><th data-i="t_fc"></th><th data-i="t_err"></th><th data-i="t_ci"></th><th data-i="t_ahead"></th></tr></thead>
          <tbody id="tbody"></tbody></table></div>
        <div class="lookup">
          <h2 data-i="lk_title"></h2>
          <div class="row"><div><label data-i="lk_label"></label><input type="month" id="lkMonth"></div>
          <button id="lkBtn" class="ghost" data-i="lk_btn"></button></div>
          <div id="lkHint" style="font-size:12.5px;color:var(--mute);margin-top:6px"></div>
          <div id="lkOut" style="margin-top:10px"></div>
        </div>
      </div>

      <div class="grid" style="align-content:start">
        <div class="card"><h2 data-i="mi_title"></h2><dl id="mi"></dl></div>
        <div class="card"><h2 data-i="up_title"></h2>
          <div id="upBox">
            <div class="drop" id="drop"><div style="font-size:26px">📦</div><div data-i="up_drop"></div><input type="file" id="file" accept=".pkl" class="hidden"></div>
            <div id="keyRow" class="hidden" style="margin-top:10px"><label data-i="up_key"></label><input type="password" id="apiKey" style="width:100%"></div>
            <p style="font-size:12px;color:var(--mute);margin:10px 0 0" data-i="up_warn"></p>
          </div>
          <p id="upOff" class="hidden" style="font-size:13px;color:var(--mute)" data-i="up_off"></p>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h2 data-i="api_title"></h2><p style="color:var(--mute);margin-top:0" data-i="api_text"></p>
      <pre id="s1"></pre><pre id="s2"></pre><pre id="s3"></pre><pre id="s4" class="hidden"></pre>
    </div>
  </section>

  <section id="uploadOnly" class="card hidden" style="margin-bottom:16px">
    <div class="drop" id="drop2"><div style="font-size:30px">📦</div><div data-i="up_drop"></div></div>
  </section>
</main>
<footer data-i="footer"></footer>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const $=s=>document.querySelector(s);
const MAX=3;
const TX={
en:{title:"Sales Forecast",subtitle:"Next 1–3 months revenue forecast · Simple Linear Regression",
 ready:"Model ready",missing:"No model loaded",
 ob_title:"Add your model to get started",ob_text:"Drop your .pkl file below, or copy sales_forecast_model.pkl next to app.py and refresh this page.",
 k_next:"Next month forecast",k_range:"Typical error range",k_range_s:"± average hold-out error",
 fc_title:"Forecast",horizon:"Months ahead (max 3)",run:"Update",csv:"Download CSV",
 t_month:"Month",t_fc:"Forecast (BRL)",t_err:"Typical range",t_ci:"95% interval",t_ahead:"Months ahead",
 lk_title:"Look up a specific month",lk_label:"Month",lk_btn:"Get forecast",lk_hint:"Available: {a} to {b}",
 mi_title:"Model details",up_title:"Replace model",up_drop:"Drop a .pkl file here or click to browse",
 up_warn:"Only upload files you created yourself. A .pkl file can contain code.",up_key:"API key",up_off:"Upload is disabled while the server is exposed on a network without an API_KEY.",
 api_title:"Use it from your own code",api_text:"The same forecasts are available over HTTP. Full interactive docs at /docs.",
 up_ok:"Model uploaded and activated.",footer:"Sales Forecast API · forecasts are statistical estimates, not guarantees",
 f_file:"File",f_algo:"Algorithm",f_train:"Training period",f_n:"Months used",f_r2:"Hold-out R²",f_mae:"Hold-out MAE",f_rmse:"Hold-out RMSE",f_mape:"Hold-out MAPE",f_loaded:"Loaded at",
 hist:"Last actual",fcl:"Forecast",rng:"Typical range",none:"Not available",
 nobeat:"On the test months the model did not beat a simple average, so read it as a stable baseline level, not a growth or seasonality signal.",
 stale:"The last actual month ({m}) is {n} months old. These numbers describe the months right after it, not the current period. Enter your latest actual revenue below to get current forecasts.",
 k_level:"Long-run level",k_level_s:"forecasts drift toward this value",f_eq:"Equation",
 lag_note:"Forecasts are limited to 3 months ahead. The model assumes sales stay near their recent level; it cannot predict growth or seasonal spikes (for example Black Friday in November).",
 an_title:"Got newer sales numbers?",an_text:"Enter the actual revenue of the latest month to refresh the forecast.",an_month:"Latest actual month",an_rev:"Actual revenue (BRL)",
 an_btn:"Recalculate",an_reset:"Back to saved data",an_using:"Forecasting from:",an_saved:"(saved with the model)",an_bad:"Enter both a month and a non-negative revenue."},
ar:{title:"توقّع المبيعات",subtitle:"توقّع إيرادات 1–3 شهور قادمة · الانحدار الخطي البسيط",
 ready:"الموديل جاهز",missing:"لا يوجد موديل",
 ob_title:"أضف الموديل للبدء",ob_text:"اسحب ملف الـ pkl هنا، أو انسخ sales_forecast_model.pkl بجوار app.py ثم حدّث الصفحة.",
 k_next:"توقّع الشهر القادم",k_range:"نطاق الخطأ المعتاد",k_range_s:"± متوسط الخطأ على بيانات الاختبار",
 fc_title:"التوقّعات",horizon:"عدد الشهور القادمة (حد أقصى 3)",run:"تحديث",csv:"تحميل CSV",
 t_month:"الشهر",t_fc:"التوقّع (BRL)",t_err:"النطاق المعتاد",t_ci:"فترة ثقة 95%",t_ahead:"شهور قدّام",
 lk_title:"ابحث عن شهر معيّن",lk_label:"الشهر",lk_btn:"اعرض التوقّع",lk_hint:"المتاح: من {a} إلى {b}",
 mi_title:"تفاصيل الموديل",up_title:"استبدال الموديل",up_drop:"اسحب ملف pkl هنا أو اضغط للاختيار",
 up_warn:"ارفع ملفات أنت أنشأتها بنفسك فقط. ملف pkl ممكن يحتوي على كود.",up_key:"مفتاح الـ API",up_off:"الرفع متوقف لأن السيرفر مفتوح على الشبكة بدون API_KEY.",
 api_title:"استخدمه من الكود الخاص بك",api_text:"نفس التوقّعات متاحة عبر HTTP. التوثيق التفاعلي الكامل على /docs.",
 up_ok:"تم رفع الموديل وتفعيله.",footer:"Sales Forecast API · التوقّعات تقديرات إحصائية وليست ضمانات",
 f_file:"الملف",f_algo:"الخوارزمية",f_train:"فترة التدريب",f_n:"عدد الشهور",f_r2:"R² على الاختبار",f_mae:"MAE على الاختبار",f_rmse:"RMSE على الاختبار",f_mape:"MAPE على الاختبار",f_loaded:"وقت التحميل",
 hist:"آخر رقم فعلي",fcl:"التوقّع",rng:"النطاق المعتاد",none:"غير متاح",
 nobeat:"في شهور الاختبار الموديل ماكانش أحسن من المتوسط البسيط، فاعتبره مستوى أساسي ثابت مش مؤشر نمو أو مواسم.",
 stale:"آخر شهر فعلي ({m}) عدّى عليه {n} شهر. الأرقام دي بتوصف الشهور اللي بعده مباشرة مش الفترة الحالية. اكتب آخر إيراد فعلي عندك تحت علشان تاخد توقّع حالي.",
 k_level:"المستوى طويل المدى",k_level_s:"التوقّعات بتقترب من الرقم ده",f_eq:"المعادلة",
 lag_note:"التوقّع محدود بـ 3 شهور قدّام. الموديل بيفترض إن المبيعات هتفضل قريبة من مستواها الأخير، ومش بيتوقع نمو ولا مواسم الذروة (زي الجمعة السوداء في نوفمبر).",
 an_title:"عندك أرقام مبيعات أحدث؟",an_text:"اكتب الإيراد الفعلي لآخر شهر علشان يتحدّث التوقّع.",an_month:"آخر شهر فعلي",an_rev:"الإيراد الفعلي (BRL)",
 an_btn:"أعد الحساب",an_reset:"الرجوع للبيانات المحفوظة",an_using:"التوقّع مبني على:",an_saved:"(محفوظ مع الموديل)",an_bad:"اكتب الشهر والإيراد (رقم موجب)."}
};
let lang=localStorage.getItem('lang')||((navigator.language||'').startsWith('ar')?'ar':'en'), info=null, chart=null, anchor=null;
const t=k=>TX[lang][k];
const fill=(s,o)=>Object.keys(o).reduce((a,k)=>a.replace('{'+k+'}',o[k]),s);
const fmt=n=>n==null?'-':Number(n).toLocaleString('en-US',{maximumFractionDigits:0});
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const addMonths=(ym,n)=>{const [y,m]=ym.split('-').map(Number);const d=new Date(y,m-1+n,1);return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0');};
const base=()=>anchor?{m:anchor.month,v:anchor.rev}:{m:info.anchor.month,v:info.anchor.revenue};
const aq=()=>anchor?`&base_month=${anchor.month}&base_revenue=${anchor.rev}`:'';
function applyLang(){
  document.documentElement.lang=lang; document.documentElement.dir=lang==='ar'?'rtl':'ltr';
  document.querySelectorAll('[data-i]').forEach(e=>e.textContent=t(e.dataset.i)||e.textContent);
  $('#langBtn').textContent=lang==='ar'?'English':'العربية';
  $('#status').textContent=info&&info.loaded?t('ready'):t('missing');
  $('#k1l').textContent=t('k_next');
  if(info&&info.loaded){renderInfo();runForecast();}
}
async function api(path,opt){
  const r=await fetch(path,opt); const j=await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(typeof j.detail==='string'?j.detail:(j.detail&&j.detail[0]&&j.detail[0].msg)||r.statusText); return j;
}
function showMsg(text,kind){$('#msg').innerHTML=text?`<div class="alert ${kind}">${esc(text)}</div>`:'';}

async function loadInfo(){
  info=await api('/api/model');
  $('#dot').classList.toggle('on',info.loaded);
  $('#status').textContent=info.loaded?t('ready'):t('missing');
  $('#onboard').classList.toggle('hidden',info.loaded);
  $('#uploadOnly').classList.toggle('hidden',info.loaded||!info.upload.enabled);
  $('#app').classList.toggle('hidden',!info.loaded);
  $('#keyRow').classList.toggle('hidden',!info.upload.needs_key);
  $('#upBox').classList.toggle('hidden',!info.upload.enabled);
  $('#upOff').classList.toggle('hidden',info.upload.enabled);
  if(info.loaded){renderInfo();await runForecast();}
}
function renderInfo(){
  const m=info.holdout_metrics||{}, tr=info.training, eq=info.equation;
  const rows=[[t('f_file'),info.file],[t('f_algo'),info.algorithm],[t('f_train'),`${tr.first_month} → ${tr.last_month}`],[t('f_n'),tr.n_months],
    [t('f_eq'),`${fmt(eq.intercept)} + ${eq.lag_coefficient.toFixed(2)} × prev`],
    [t('f_r2'),m.R2!=null?m.R2.toFixed(2):'-'],[t('f_mae'),fmt(m.MAE)],[t('f_rmse'),fmt(m.RMSE)],[t('f_mape'),m['MAPE %']!=null?m['MAPE %'].toFixed(1)+'%':'-'],[t('f_loaded'),info.loaded_at]];
  $('#mi').innerHTML=rows.map(r=>`<dt>${esc(r[0])}</dt><dd>${esc(r[1])}</dd>`).join('');
  $('#k3l').textContent=t('k_level'); $('#k3s').textContent=t('k_level_s'); $('#k3v').textContent=fmt(eq.long_run_level);
  const o=location.origin;
  $('#s1').innerHTML=`<button class="copy" onclick="cp(this)">copy</button>curl "${o}/api/forecast?month=${addMonths(info.anchor.month,1)}"`;
  $('#s2').innerHTML=`<button class="copy" onclick="cp(this)">copy</button>curl "${o}/api/forecast/range?months=${MAX}"`;
  $('#s3').innerHTML=`<button class="copy" onclick="cp(this)">copy</button>import requests\nr = requests.get("${o}/api/forecast", params={"month": "${addMonths(info.anchor.month,1)}"})\nprint(r.json()["forecast"])`;
  $('#s4').classList.remove('hidden');
  $('#s4').innerHTML=`<button class="copy" onclick="cp(this)">copy</button># with the latest actual revenue you know:\ncurl "${o}/api/forecast/range?months=${MAX}&base_month=2026-09&base_revenue=900000"`;
}
function cp(b){navigator.clipboard.writeText(b.parentElement.innerText.replace(/^copy\n?/,'')); b.textContent='✓'; setTimeout(()=>b.textContent='copy',1200);}

function renderAnchor(){
  if(!$('#anMonth').value)$('#anMonth').value=info.anchor.month;
  $('#anReset').classList.toggle('hidden',!anchor);
  const b=base();
  $('#anState').innerHTML=`<div class="s" style="margin:8px 0 0">${esc(t('an_using'))} <b>${b.m}</b> = <bdi dir="ltr"><b>${fmt(b.v)} BRL</b></bdi> ${anchor?'':esc(t('an_saved'))}</div>`;
  const a=addMonths(b.m,1), z=addMonths(b.m,MAX);
  $('#lkMonth').min=a; $('#lkMonth').max=z;
  $('#lkHint').textContent=fill(t('lk_hint'),{a,b:z});
  if(!$('#lkMonth').value||$('#lkMonth').value<a||$('#lkMonth').value>z)$('#lkMonth').value=a;
}
async function runForecast(){
  try{
    const h=+$('#horizon').value;
    const r=await api(`/api/forecast/range?months=${h}${aq()}`);
    const fc=r.forecasts, pre=[{month:r.anchor.month,forecast:r.anchor.revenue}];
    showMsg('','');
    renderKpis(fc); renderTable(fc); renderChart(pre,fc); renderAnchor();
    $('#csvBtn').href=`/api/forecast/range.csv?months=${h}${aq()}`;
  }catch(e){showMsg(e.message,'bad');}
}
function rangeText(r){return r.error_range_low!=null?`<bdi dir="ltr">${fmt(r.error_range_low)} – ${fmt(r.error_range_high)}</bdi>`:'-';}
function renderKpis(fc){
  const r=fc[0]; $('#k1v').innerHTML=`<bdi dir="ltr">${fmt(r.forecast)} BRL</bdi>`; $('#k1s').textContent=r.month;
  $('#k2v').innerHTML=r.error_range_low!=null?rangeText(r):t('none');
}
function renderTable(fc){
  $('#tbody').innerHTML=fc.map(r=>`<tr><td>${r.month}</td><td class="n"><b>${fmt(r.forecast)}</b></td><td class="n">${rangeText(r)}</td>
   <td class="n">${r.ci95_low!=null?`<bdi dir="ltr">${fmt(r.ci95_low)} – ${fmt(r.ci95_high)}</bdi>`:'-'}</td><td class="n">${r.months_ahead}</td></tr>`).join('');
}
function renderChart(pre,fc){
  if(typeof Chart==='undefined'){$('#chart').classList.add('hidden');return;}
  const nP=pre.length, labels=[...pre,...fc].map(r=>r.month), nulls=n=>Array(n).fill(null);
  const fcLine=[...nulls(nP-1),pre[nP-1].forecast,...fc.map(r=>r.forecast)];
  const lo=[...nulls(nP),...fc.map(r=>r.error_range_low)], hi=[...nulls(nP),...fc.map(r=>r.error_range_high)];
  if(chart) chart.destroy();
  chart=new Chart($('#chart'),{type:'line',data:{labels,datasets:[
    {label:t('hist'),data:[...pre.map(r=>r.forecast),...nulls(fc.length)],borderColor:'#0f172a',backgroundColor:'#0f172a',pointRadius:6,tension:0},
    {label:t('rng'),data:hi,borderColor:'transparent',backgroundColor:'rgba(79,70,229,.15)',pointRadius:0,fill:'+1'},
    {label:'lo',data:lo,borderColor:'transparent',pointRadius:0,fill:false},
    {label:t('fcl'),data:fcLine,borderColor:'#4f46e5',backgroundColor:'#4f46e5',borderWidth:3,pointRadius:3,tension:0}
  ]},options:{responsive:true,interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{filter:i=>i.text!=='lo'}}},
   scales:{y:{ticks:{callback:v=>(v/1000)+'k'}}}}});
}
async function lookup(){
  const m=$('#lkMonth').value; if(!m)return;
  try{
    const r=await api('/api/forecast?month='+m+aq());
    $('#lkOut').innerHTML=`<div class="alert good"><b>${r.month}: ${fmt(r.forecast)} BRL</b><br>${t('t_err')}: ${rangeText(r)}</div>`;
  }catch(e){$('#lkOut').innerHTML=`<div class="alert bad">${esc(e.message)}</div>`;}
}
async function upload(file){
  if(!file)return;
  const h={'Content-Type':'application/octet-stream'}; const k=$('#apiKey').value; if(k)h['X-API-Key']=k;
  try{await api('/api/model/upload',{method:'POST',headers:h,body:file});anchor=null;showMsg(t('up_ok'),'good');await loadInfo();}
  catch(e){showMsg(e.message,'bad');}
}
function wireDrop(box,input){
  box.onclick=()=>input.click();
  ['dragover','dragenter'].forEach(ev=>box.addEventListener(ev,e=>{e.preventDefault();box.classList.add('over');}));
  ['dragleave','drop'].forEach(ev=>box.addEventListener(ev,e=>{e.preventDefault();box.classList.remove('over');}));
  box.addEventListener('drop',e=>upload(e.dataTransfer.files[0])); input.onchange=()=>upload(input.files[0]);
}
wireDrop($('#drop'),$('#file'));
(()=>{const i=document.createElement('input');i.type='file';i.accept='.pkl';i.className='hidden';document.body.appendChild(i);wireDrop($('#drop2'),i);})();
$('#anBtn').onclick=()=>{const m=$('#anMonth').value,v=parseFloat($('#anRev').value);
  if(!m||isNaN(v)||v<0){showMsg(t('an_bad'),'bad');return;} anchor={month:m,rev:v}; $('#lkMonth').value=''; runForecast();};
$('#anReset').onclick=()=>{anchor=null;$('#anRev').value='';$('#anMonth').value=info.anchor.month;$('#lkMonth').value='';runForecast();};
$('#runBtn').onclick=runForecast; $('#horizon').onchange=runForecast; $('#lkBtn').onclick=lookup;
$('#langBtn').onclick=()=>{lang=lang==='en'?'ar':'en';localStorage.setItem('lang',lang);applyLang();};
applyLang(); loadInfo().catch(e=>showMsg(e.message,'bad'));
</script>
</body></html>
"""


# ----------------------------------------------------------------------------
# 6. Entry point: python app.py
# ----------------------------------------------------------------------------
def _free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


if __name__ == "__main__":
    port = _free_port(int(os.getenv("PORT", "8000")))
    url = f"http://{'127.0.0.1' if IS_LOCAL else HOST}:{port}"
    print(f"\n  Sales Forecast API  ->  {url}\n  API docs            ->  {url}/docs\n  (press Ctrl+C to stop)\n")
    if IS_LOCAL and not os.getenv("NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
