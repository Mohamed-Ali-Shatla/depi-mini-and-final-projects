#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review Prediction  -  API + Web Dashboard  (single file)
=============================================================
Predicts whether a DELIVERED order is likely to get a BAD review (1-3 stars),
using the model bundle saved by the notebook (bad_review_pipeline.pkl).

Quick start
-----------
    python app.py

  1. Put  bad_review_pipeline.pkl  next to this file   (or drag & drop it in the browser)
  2. The dashboard opens at  http://127.0.0.1:8000
  3. Interactive API docs at  http://127.0.0.1:8000/docs

Options
-------
    python app.py --model path/to/model.pkl --port 9000 --host 0.0.0.0 --api-key SECRET --no-browser
"""
import argparse
import importlib.util
import io
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "1.0.0"
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # pragma: no cover
    BASE_DIR = Path.cwd()


def _ensure_deps():
    """Offer to pip-install anything missing, so a normal user just runs `python app.py`."""
    need = {"numpy": "numpy", "pandas": "pandas", "joblib": "joblib", "lightgbm": "lightgbm",
            "fastapi": "fastapi", "uvicorn": "uvicorn"}
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if not (importlib.util.find_spec("multipart") or importlib.util.find_spec("python_multipart")):
        missing.append("python-multipart")
    if not missing:
        return
    print("Missing packages: " + ", ".join(missing))
    ans = "y"
    if sys.stdin and sys.stdin.isatty():
        ans = input("Install them now? [Y/n] ").strip().lower()
    if ans in ("", "y", "yes"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        sys.exit("Please run:  pip install " + " ".join(missing))


if __name__ == "__main__":
    _ensure_deps()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# =====================================================================================
#  CORE  -  feature engineering + prediction (mirrors the notebook exactly)
# =====================================================================================
REQUIRED_KEYS = ["models", "features", "threshold", "cat_cols", "cat_levels",
                 "te_maps", "te_prior", "cat_freq_map", "start_date"]
DATE_COLS = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
REQUIRED_INPUT = DATE_COLS + ["price", "freight_value"]
OPTIONAL_NUM = ["payment_installments", "payment_value", "product_weight_g", "product_length_cm",
                "product_height_cm", "product_width_cm", "customer_zip_code_prefix", "seller_zip_code_prefix"]
NUM_INPUT = ["price", "freight_value", "order_item_id", "product_name_lenght",
             "product_description_lenght", "product_photos_qty"] + OPTIONAL_NUM
TEXT_INPUT = ["product_category", "customer_state", "seller_state", "payment_type", "seller_id", "product_id"]
MAX_ROWS = 200_000

EXAMPLE_ORDER = {
    "order_id": "demo-0001",
    "order_purchase_timestamp": "2018-03-10 14:30:00",
    "order_estimated_delivery_date": "2018-03-30 00:00:00",
    "order_delivered_customer_date": "2018-04-12 16:45:00",
    "price": 89.9, "freight_value": 21.5, "order_item_id": 1,
    "product_category": "health_beauty", "product_name_lenght": 52,
    "product_description_lenght": 640, "product_photos_qty": 2,
    "product_weight_g": 700, "product_length_cm": 24, "product_height_cm": 12, "product_width_cm": 18,
    "customer_state": "SP", "seller_state": "RJ", "payment_type": "credit_card",
    "payment_installments": 3, "payment_value": 111.4,
}

# name, kind, required, group, label EN, label AR, features that make the field useful (None = always shown)
FIELDS = [
    ("order_purchase_timestamp", "datetime", True, "dates", "Purchase date & time", "تاريخ ووقت الشراء", None),
    ("order_estimated_delivery_date", "datetime", True, "dates", "Estimated delivery date", "تاريخ التسليم المتوقع", None),
    ("order_delivered_customer_date", "datetime", True, "dates", "Actual delivery date", "تاريخ التسليم الفعلي", None),
    ("price", "number", True, "order", "Item price", "سعر المنتج", None),
    ("freight_value", "number", True, "order", "Shipping cost", "تكلفة الشحن", None),
    ("order_item_id", "number", False, "order", "Item number in order", "رقم الصنف في الأوردر", ["order_item_id"]),
    ("product_category", "combo", False, "product", "Product category", "فئة المنتج", ["product_category", "category_freq"]),
    ("product_name_lenght", "number", False, "product", "Product name length (chars)", "طول اسم المنتج (حروف)", ["product_name_lenght"]),
    ("product_description_lenght", "number", False, "product", "Description length (chars)", "طول الوصف (حروف)", ["product_description_lenght"]),
    ("product_photos_qty", "number", False, "product", "Number of photos", "عدد الصور", ["product_photos_qty"]),
    ("product_weight_g", "number", False, "product", "Weight (g)", "الوزن (جرام)", ["product_weight_g", "freight_per_kg", "product_density"]),
    ("product_length_cm", "number", False, "product", "Length (cm)", "الطول (سم)", ["product_length_cm", "product_volume_cm3", "product_density"]),
    ("product_height_cm", "number", False, "product", "Height (cm)", "الارتفاع (سم)", ["product_height_cm", "product_volume_cm3", "product_density"]),
    ("product_width_cm", "number", False, "product", "Width (cm)", "العرض (سم)", ["product_width_cm", "product_volume_cm3", "product_density"]),
    ("customer_state", "combo", False, "geo", "Customer state", "ولاية العميل", ["customer_state", "same_state"]),
    ("seller_state", "combo", False, "geo", "Seller state", "ولاية البائع", ["seller_state", "same_state"]),
    ("customer_zip_code_prefix", "number", False, "geo", "Customer ZIP prefix", "الرمز البريدي للعميل", ["customer_zip_code_prefix"]),
    ("seller_zip_code_prefix", "number", False, "geo", "Seller ZIP prefix", "الرمز البريدي للبائع", ["seller_zip_code_prefix"]),
    ("payment_type", "combo", False, "pay", "Payment type", "طريقة الدفع", ["payment_type"]),
    ("payment_installments", "number", False, "pay", "Installments", "عدد الأقساط", ["payment_installments"]),
    ("payment_value", "number", False, "pay", "Payment value", "قيمة الدفع", ["payment_value"]),
    ("seller_id", "text", False, "ids", "Seller ID", "رقم البائع", ["te_seller_id", "order_n_sellers"]),
    ("product_id", "text", False, "ids", "Product ID", "رقم المنتج", ["te_product_id", "order_n_products"]),
    ("order_id", "text", False, "ids", "Order ID", "رقم الأوردر", None),
]

_L = {
    "delivery_days": ("Delivery time (days)", "مدة التوصيل (أيام)"),
    "delivery_delay_days": ("Delay vs. estimate (days)", "التأخير عن الموعد المتوقع (أيام)"),
    "estimated_days": ("Promised delivery time (days)", "مدة التوصيل الموعودة (أيام)"),
    "delay_ratio": ("Delay ratio", "نسبة التأخير"),
    "is_delayed": ("Delivered late", "تم التسليم متأخرًا"),
    "delay_severity": ("Delay severity", "شدة التأخير"),
    "early_days": ("Days delivered early", "أيام التسليم المبكر"),
    "delivery_to_estimate_ratio": ("Actual vs. promised time", "الفعلي مقابل الموعود"),
    "price": ("Item price", "سعر المنتج"),
    "freight_value": ("Shipping cost", "تكلفة الشحن"),
    "freight_to_price_ratio": ("Shipping / price ratio", "نسبة الشحن للسعر"),
    "total_value": ("Item total value", "إجمالي قيمة الصنف"),
    "price_per_photo": ("Price per photo", "السعر لكل صورة"),
    "product_name_lenght": ("Product name length", "طول اسم المنتج"),
    "product_description_lenght": ("Description length", "طول الوصف"),
    "product_photos_qty": ("Number of photos", "عدد الصور"),
    "description_to_name_ratio": ("Description / name length", "نسبة الوصف للاسم"),
    "product_info_missing": ("Missing product info", "بيانات منتج ناقصة"),
    "order_item_id": ("Item number in order", "رقم الصنف في الأوردر"),
    "order_n_items": ("Items in order", "عدد الأصناف في الأوردر"),
    "order_total_price": ("Order total price", "إجمالي سعر الأوردر"),
    "order_total_freight": ("Order total shipping", "إجمالي شحن الأوردر"),
    "order_total_value": ("Order total value", "إجمالي قيمة الأوردر"),
    "price_share": ("Item share of order price", "نسبة الصنف من سعر الأوردر"),
    "order_n_products": ("Distinct products in order", "عدد المنتجات المختلفة"),
    "order_n_sellers": ("Sellers in order", "عدد البائعين في الأوردر"),
    "same_state": ("Seller & customer in same state", "البائع والعميل في نفس الولاية"),
    "purchase_month": ("Purchase month", "شهر الشراء"),
    "purchase_day_of_week": ("Purchase weekday", "يوم الشراء"),
    "purchase_hour": ("Purchase hour", "ساعة الشراء"),
    "is_weekend_purchase": ("Weekend purchase", "شراء في الويك إند"),
    "delivered_day_of_week": ("Delivery weekday", "يوم التسليم"),
    "days_since_start": ("Purchase date (trend)", "تاريخ الشراء (الاتجاه العام)"),
    "product_category": ("Product category", "فئة المنتج"),
    "category_freq": ("Category popularity", "شيوع الفئة"),
    "customer_state": ("Customer state", "ولاية العميل"),
    "seller_state": ("Seller state", "ولاية البائع"),
    "payment_type": ("Payment type", "طريقة الدفع"),
    "payment_installments": ("Installments", "عدد الأقساط"),
    "payment_value": ("Payment value", "قيمة الدفع"),
    "product_weight_g": ("Weight (g)", "الوزن (جرام)"),
    "product_length_cm": ("Length (cm)", "الطول (سم)"),
    "product_height_cm": ("Height (cm)", "الارتفاع (سم)"),
    "product_width_cm": ("Width (cm)", "العرض (سم)"),
    "product_volume_cm3": ("Package volume", "حجم العبوة"),
    "freight_per_kg": ("Shipping per kg", "الشحن لكل كيلو"),
    "product_density": ("Product density", "كثافة المنتج"),
    "customer_zip_code_prefix": ("Customer ZIP prefix", "الرمز البريدي للعميل"),
    "seller_zip_code_prefix": ("Seller ZIP prefix", "الرمز البريدي للبائع"),
    "te_seller_id": ("Seller's past bad-review rate", "نسبة التقييمات السيئة السابقة للبائع"),
    "te_product_id": ("Product's past bad-review rate", "نسبة التقييمات السيئة السابقة للمنتج"),
}


def label(name):
    en, ar = _L.get(name, (name.replace("_", " ").capitalize(), name.replace("_", " ")))
    return {"en": en, "ar": ar}


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _isna(v):
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _py(v):
    if _isna(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    return str(v)


def _clean_json(obj):
    return json.loads(json.dumps(obj, default=str))


def _dt(s):
    try:
        x = pd.to_datetime(s, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        x = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(x.dt, "tz", None) is not None:
            x = x.dt.tz_localize(None)
    except Exception:
        pass
    return x


def _blank(df, idx=None):
    return pd.Series(None, index=df.index, dtype=object)


def create_features(data, start_date):
    """Same feature engineering as the notebook (Cell 4)."""
    f = data.copy()
    p, d, e = f[DATE_COLS[0]], f[DATE_COLS[1]], f[DATE_COLS[2]]

    f["purchase_month"] = p.dt.month
    f["purchase_day_of_week"] = p.dt.dayofweek
    f["purchase_hour"] = p.dt.hour
    f["is_weekend_purchase"] = f["purchase_day_of_week"].isin([5, 6]).astype(int)
    f["days_since_start"] = (p - start_date).dt.total_seconds() / 86400
    f["delivered_day_of_week"] = d.dt.dayofweek

    f["delivery_days"] = (d - p).dt.total_seconds() / 86400
    f["delivery_delay_days"] = (d - e).dt.total_seconds() / 86400
    f["estimated_days"] = (e - p).dt.total_seconds() / 86400
    f["delay_ratio"] = f["delivery_delay_days"] / (f["estimated_days"] + 1)
    f["is_delayed"] = (f["delivery_delay_days"] > 0).astype(int)
    f["early_days"] = (-f["delivery_delay_days"]).clip(lower=0)
    f["delivery_to_estimate_ratio"] = f["delivery_days"] / (f["estimated_days"] + 1)
    bins = [-np.inf, 0, 3, 7, 15, np.inf]
    f["delay_severity"] = pd.cut(f["delivery_delay_days"], bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)

    f["price_per_photo"] = f["price"] / (f["product_photos_qty"] + 1)
    f["description_to_name_ratio"] = f["product_description_lenght"] / (f["product_name_lenght"] + 1)
    f["freight_to_price_ratio"] = f["freight_value"] / (f["price"] + 1)
    f["total_value"] = f["price"] + f["freight_value"]
    f["product_info_missing"] = f[["product_name_lenght", "product_description_lenght",
                                   "product_photos_qty"]].isna().any(axis=1).astype(int)
    f["product_volume_cm3"] = f["product_length_cm"] * f["product_height_cm"] * f["product_width_cm"]
    f["freight_per_kg"] = f["freight_value"] / (f["product_weight_g"] / 1000 + 0.1)
    f["product_density"] = f["product_weight_g"] / (f["product_volume_cm3"] + 1)

    g = f.groupby("order_id")
    f["order_n_items"] = g["order_id"].transform("size")
    f["order_total_price"] = g["price"].transform("sum")
    f["order_total_freight"] = g["freight_value"].transform("sum")
    f["order_total_value"] = f["order_total_price"] + f["order_total_freight"]
    f["price_share"] = f["price"] / (f["order_total_price"] + 1)
    f["order_n_products"] = g["product_id"].transform("nunique").clip(lower=1)
    f["order_n_sellers"] = g["seller_id"].transform("nunique").clip(lower=1)
    f["same_state"] = ((f["customer_state"] == f["seller_state"]) & f["customer_state"].notna()
                       & f["seller_state"].notna()).astype(int)
    return f


def prepare(b, raw):
    """Raw records -> (X, F, order_ids, errors{row_pos: msg}, valid_positions)."""
    df = raw.copy().reset_index(drop=True)
    n = len(df)
    for c in DATE_COLS + NUM_INPUT + TEXT_INPUT + ["order_id"]:
        if c not in df.columns:
            df[c] = np.nan
    df["order_id"] = [f"row_{i + 1}" if (_isna(v) or str(v).strip() == "") else str(v).strip()
                      for i, v in enumerate(df["order_id"])]
    for c in DATE_COLS:
        df[c] = _dt(df[c])
    for c in NUM_INPUT:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["order_item_id"] = df["order_item_id"].fillna(1)
    for c in TEXT_INPUT:
        df[c] = df[c].astype(object).map(lambda v: None if (_isna(v) or str(v).strip() == "") else str(v).strip())

    miss = df[REQUIRED_INPUT].isna()
    errors = {}
    for i in np.flatnonzero(miss.any(axis=1).values):
        errors[int(i)] = "Missing or invalid value for: " + ", ".join(c for c in REQUIRED_INPUT if miss.iat[i, REQUIRED_INPUT.index(c)])
    valid_pos = [i for i in range(n) if i not in errors]
    ids = df["order_id"].tolist()
    if not valid_pos:
        return None, None, ids, errors, valid_pos

    f = create_features(df.iloc[valid_pos].copy(), b["start_date"])

    f["category_freq"] = f["product_category"].map(b["cat_freq_map"]).astype(float).fillna(1)
    for col, mp in b["te_maps"].items():
        src = f[col] if col in f.columns else _blank(f)
        f["te_" + col] = src.map(mp).astype(float).fillna(b["te_prior"])
    for c in b["cat_cols"]:
        src = f[c] if c in f.columns else _blank(f)
        f[c] = pd.Categorical(src.astype(object).map(lambda v: None if _isna(v) else str(v)),
                              categories=b["cat_levels"][c])

    X = f.reindex(columns=b["features"])
    for c in b["features"]:
        if c not in b["cat_cols"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.reset_index(drop=True), f.reset_index(drop=True), ids, errors, valid_pos


def predict_matrix(b, X, explain):
    models = b["models"]
    probs = np.mean([m.predict(X, num_iteration=(getattr(m, "best_iteration", 0) or None)) for m in models], axis=0)
    contrib = None
    if explain:
        contrib = np.mean([m.predict(X, num_iteration=(getattr(m, "best_iteration", 0) or None), pred_contrib=True)
                           for m in models], axis=0)
    return probs, contrib


def risk_level(p, thr):
    if p >= thr:
        return "high"
    if p >= thr * 0.6:
        return "medium"
    return "low"


def top_factors(b, crow, X, j, k):
    c = np.asarray(crow)[:-1]
    out = []
    for i in np.argsort(-np.abs(c))[:k]:
        if abs(c[i]) < 1e-9:
            continue
        name = b["features"][int(i)]
        out.append({"feature": name, "label": label(name), "value": _py(X.iat[j, int(i)]),
                    "impact": round(float(c[i]), 4),
                    "effect": "increases_risk" if c[i] > 0 else "reduces_risk"})
    return out


def run_predictions(b, raw, explain=False, detail=False, k=6):
    n = len(raw)
    X, F, ids, errors, valid_pos = prepare(b, raw)
    results = [None] * n
    for pos, msg in errors.items():
        results[pos] = {"row": pos + 1, "order_id": ids[pos], "probability": None, "risk_level": None,
                        "predicted_bad_review": None, "error": msg}
    if valid_pos:
        probs, contrib = predict_matrix(b, X, explain)
        thr = b["threshold"]
        for j, pos in enumerate(valid_pos):
            p = float(probs[j])
            r = {"row": pos + 1, "order_id": ids[pos], "probability": round(p, 4),
                 "risk_level": risk_level(p, thr), "predicted_bad_review": bool(p >= thr)}
            if contrib is not None:
                r["top_factors"] = top_factors(b, contrib[j], X, j, k)
            if detail:
                dd, ed, dl = (float(F[c].iat[j]) for c in ("delivery_days", "estimated_days", "delivery_delay_days"))
                r["delivery"] = {"delivery_days": round(dd, 2), "promised_days": round(ed, 2), "delay_days": round(dl, 2)}
                w = []
                if dd < 0:
                    w.append("Delivery date is earlier than the purchase date - please check the dates.")
                if dd > 150:
                    w.append("Delivery took more than 150 days - please check the dates.")
                if float(F["price"].iat[j]) <= 0:
                    w.append("Price is zero or negative.")
                r["warnings"] = w
            results[pos] = r
    return results


def summarize(results, thr):
    ok = [r for r in results if r.get("probability") is not None]
    probs = np.array([r["probability"] for r in ok], dtype=float)
    return {
        "rows": len(results), "scored": len(ok), "invalid": len(results) - len(ok),
        "high": sum(r["risk_level"] == "high" for r in ok),
        "medium": sum(r["risk_level"] == "medium" for r in ok),
        "low": sum(r["risk_level"] == "low" for r in ok),
        "predicted_bad": sum(bool(r["predicted_bad_review"]) for r in ok),
        "mean_probability": round(float(probs.mean()), 4) if len(probs) else None,
        "histogram": (np.histogram(probs, bins=10, range=(0, 1))[0].tolist() if len(probs) else [0] * 10),
        "threshold": round(thr, 4),
    }


def drivers_text(r, n=3):
    xs = [x for x in r.get("top_factors", []) if x["impact"] > 0][:n]
    return "; ".join("%s = %s" % (x["label"]["en"], x["value"]) for x in xs)


def results_to_csv(raw, results):
    out = raw.reset_index(drop=True).copy()
    out["bad_review_probability"] = [r["probability"] for r in results]
    out["risk_level"] = [r["risk_level"] for r in results]
    out["predicted_bad_review"] = pd.array([None if r["predicted_bad_review"] is None else int(r["predicted_bad_review"])
                                            for r in results], dtype="Int64")
    if any("top_factors" in r for r in results):
        out["top_risk_drivers"] = [drivers_text(r) for r in results]
    if any(r.get("error") for r in results):
        out["error"] = [r.get("error") for r in results]
    return out.to_csv(index=False).encode("utf-8-sig")


def read_table(data, filename):
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(io.BytesIO(data))
            except ImportError:
                raise ApiError(400, "excel_unsupported",
                               "Reading Excel files needs 'openpyxl' (pip install openpyxl). Or save the file as CSV.")
        else:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
            first = text.splitlines()[0] if text.strip() else ""
            sep = max([",", ";", "\t"], key=first.count)
            df = pd.read_csv(io.StringIO(text), sep=sep)
    except ApiError:
        raise
    except Exception as ex:
        raise ApiError(400, "unreadable_file", "Could not read the file as a table: %s" % ex)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ApiError(400, "empty_file", "The file has no rows.")
    if len(df) > MAX_ROWS:
        raise ApiError(413, "too_many_rows", "Maximum %s rows per file (got %s)." % (f"{MAX_ROWS:,}", f"{len(df):,}"))
    return df


# ------------------------------------------------------------------ model store
class ModelStore:
    def __init__(self):
        self.bundle = None
        self.path = None
        self.loaded_at = None
        self.error = None
        self.importance = []
        self.lock = threading.Lock()

    def require(self):
        b = self.bundle
        if b is None:
            raise ApiError(503, "model_not_loaded",
                           "No model loaded. Drop your .pkl file in the dashboard, put it next to app.py, "
                           "or POST it to /api/model/upload.")
        return b

    def load(self, path):
        path = Path(path)
        try:
            obj = joblib.load(path)
        except Exception as ex:
            raise ValueError("Could not open the file (%s: %s). Make sure it was saved with joblib.dump "
                             "and that lightgbm is installed." % (type(ex).__name__, ex))
        if not isinstance(obj, dict):
            raise ValueError("This file is not the model bundle saved by the notebook (expected a dict "
                             "with keys: %s)." % ", ".join(REQUIRED_KEYS))
        missing = [k for k in REQUIRED_KEYS if k not in obj]
        if missing:
            raise ValueError("Model file is missing keys: " + ", ".join(missing))
        models = obj["models"]
        models = list(models) if isinstance(models, (list, tuple)) else [models]
        if not models or not all(hasattr(m, "predict") for m in models):
            raise ValueError("The 'models' entry does not contain trained models.")
        b = dict(obj)
        b["models"] = models
        b["features"] = list(obj["features"])
        b["cat_cols"] = list(obj["cat_cols"])
        b["threshold"] = float(obj["threshold"])
        b["te_prior"] = float(obj["te_prior"])
        sd = pd.Timestamp(obj["start_date"])
        b["start_date"] = sd.tz_localize(None) if sd.tzinfo is not None else sd
        # dry run: reject incompatible files immediately with a clear message
        try:
            probe = run_predictions(b, pd.DataFrame([EXAMPLE_ORDER]), explain=True, detail=True)[0]
            if probe.get("probability") is None:
                raise ValueError(probe.get("error"))
        except Exception as ex:
            raise ValueError("Model loaded but a test prediction failed: %s: %s" % (type(ex).__name__, ex))
        imp = self._importance(b)
        with self.lock:
            self.bundle, self.path, self.loaded_at, self.error, self.importance = b, path, datetime.now(), None, imp

    @staticmethod
    def _importance(b):
        acc = {}
        for m in b["models"]:
            try:
                names = m.feature_name()
                g = np.asarray(m.feature_importance(importance_type="gain"), dtype=float)
            except Exception:
                continue
            s = g.sum() or 1.0
            for nm, v in zip(names, g):
                acc[nm] = acc.get(nm, 0.0) + v / s
        k = max(len(b["models"]), 1)
        items = sorted(((n, v / k) for n, v in acc.items()), key=lambda kv: -kv[1])[:20]
        return [{"feature": n, "label": label(n), "importance": round(v, 4)} for n, v in items]


store = ModelStore()


def find_model(explicit=None):
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            cands = sorted(list(p.glob("*.pkl")) + list(p.glob("*.joblib")), key=lambda x: -x.stat().st_mtime)
            return cands[0] if cands else None
        return p if p.exists() else None
    for d in (BASE_DIR, Path.cwd()):
        p = d / "bad_review_pipeline.pkl"
        if p.exists():
            return p
    cands = []
    for d in {BASE_DIR, Path.cwd()}:
        cands += list(d.glob("*.pkl")) + list(d.glob("*.joblib"))
    cands.sort(key=lambda x: -x.stat().st_mtime)
    return cands[0] if cands else None


def example_record(b):
    ex = dict(EXAMPLE_ORDER)
    for c, pref in (("product_category", "health_beauty"), ("customer_state", "SP"),
                    ("seller_state", "RJ"), ("payment_type", "credit_card")):
        lv = b["cat_levels"].get(c)
        if lv and pref not in lv:
            ex[c] = lv[0]
    return ex


def schema_fields(b):
    feats = set(b["features"])
    out = []
    for name, kind, req, group, en, ar, deps in FIELDS:
        used = req or deps is None or any(d in feats for d in deps)
        item = {"name": name, "kind": kind, "required": req, "group": group,
                "label": {"en": en, "ar": ar}, "used": bool(used)}
        if kind == "combo":
            item["options"] = list(b["cat_levels"].get(name, []))
        out.append(item)
    return out


def model_info():
    if store.bundle is None:
        return {"loaded": False, "error": store.error}
    b = store.bundle
    size = store.path.stat().st_size / 1e6 if store.path and store.path.exists() else None
    return {
        "loaded": True, "file": store.path.name if store.path else None,
        "size_mb": round(size, 2) if size is not None else None,
        "loaded_at": store.loaded_at.isoformat(timespec="seconds"),
        "n_models": len(b["models"]), "n_features": len(b["features"]),
        "threshold": round(b["threshold"], 4), "start_date": str(b["start_date"].date()),
        "best_iterations": [int(getattr(m, "best_iteration", 0) or 0) for m in b["models"]],
        "categorical": {c: len(b["cat_levels"][c]) for c in b["cat_cols"]},
        "target_encoded": list(b["te_maps"].keys()),
        "importance": store.importance,
        "best_params": _clean_json(b.get("best_params", {})),
    }


# =====================================================================================
#  WEB LAYER  -  FastAPI
# =====================================================================================
from fastapi import FastAPI, File, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, Response  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

CFG = {"model_path": None, "api_key": None}
START_TIME = time.time()


@asynccontextmanager
async def lifespan(_app):
    p = find_model(CFG["model_path"])
    if p is None:
        print("  [model] none found - open the dashboard and drop your .pkl file there.")
    else:
        try:
            store.load(p)
            print("  [model] loaded %s  (%d models, %d features, threshold %.3f)" % (
                p.name, len(store.bundle["models"]), len(store.bundle["features"]), store.bundle["threshold"]))
        except Exception as ex:
            store.error = str(ex)
            print("  [model] FAILED to load %s: %s" % (p.name, ex))
    yield


DESCRIPTION = """
Predicts the probability that a **delivered order** receives a **bad review (1-3 stars)**.

**Quick start**
1. `POST /api/predict` with one order (JSON) -> probability, risk level and the top reasons.
2. `POST /api/predict/file` with a CSV/Excel file -> scores every order (add `?format=csv` to download results).
3. `GET /api/model` shows what the loaded model expects. Upload a new `.pkl` with `POST /api/model/upload`.

Risk levels: **high** = probability >= model threshold, **medium** = >= 60% of the threshold, **low** = below.
If the server was started with `--api-key`, send the key in the `X-API-Key` header.
"""

app = FastAPI(title="Review Prediction API", version=VERSION, description=DESCRIPTION, lifespan=lifespan,
              openapi_tags=[{"name": "Prediction"}, {"name": "Model"}, {"name": "System"}])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def _auth(request: Request, call_next):
    key, path = CFG["api_key"], request.url.path
    if key and path.startswith("/api/") and path != "/api/health" and request.method != "OPTIONS":
        if request.headers.get("x-api-key") != key:
            return JSONResponse({"error": {"code": "unauthorized", "message": "Missing or invalid X-API-Key header."}},
                                status_code=401)
    return await call_next(request)


@app.exception_handler(ApiError)
async def _api_error(_request: Request, exc: ApiError):
    return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)


@app.exception_handler(Exception)
async def _any_error(_request: Request, exc: Exception):
    return JSONResponse({"error": {"code": "internal_error", "message": "%s: %s" % (type(exc).__name__, exc)}},
                        status_code=500)


class OrderIn(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_ORDER})
    order_purchase_timestamp: str
    order_estimated_delivery_date: str
    order_delivered_customer_date: str
    price: float
    freight_value: float
    order_id: Optional[str] = None
    order_item_id: Optional[float] = None
    product_category: Optional[str] = None
    product_name_lenght: Optional[float] = None
    product_description_lenght: Optional[float] = None
    product_photos_qty: Optional[float] = None
    product_weight_g: Optional[float] = None
    product_length_cm: Optional[float] = None
    product_height_cm: Optional[float] = None
    product_width_cm: Optional[float] = None
    customer_state: Optional[str] = None
    seller_state: Optional[str] = None
    customer_zip_code_prefix: Optional[float] = None
    seller_zip_code_prefix: Optional[float] = None
    payment_type: Optional[str] = None
    payment_installments: Optional[float] = None
    payment_value: Optional[float] = None
    seller_id: Optional[str] = None
    product_id: Optional[str] = None


class BatchIn(BaseModel):
    records: List[OrderIn]


def _dump(o):
    return o.model_dump() if hasattr(o, "model_dump") else o.dict()


# ------------------------------------------------------------------ System
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    return HTMLResponse(INDEX_HTML.replace("__VERSION__", VERSION))


@app.get("/api/health", tags=["System"], summary="Service status")
def health():
    return {"status": "ok", "version": VERSION, "model_loaded": store.bundle is not None,
            "model_error": store.error, "auth_required": bool(CFG["api_key"]),
            "uptime_seconds": int(time.time() - START_TIME)}


# ------------------------------------------------------------------ Model
@app.get("/api/model", tags=["Model"], summary="Loaded model details (features, threshold, importance)")
def get_model():
    store.require()
    return model_info()


@app.get("/api/schema", tags=["Model"], summary="Input fields the model understands + an example order")
def get_schema():
    b = store.require()
    return {"fields": schema_fields(b), "example": example_record(b)}


@app.post("/api/model/upload", tags=["Model"], summary="Upload / replace the model (.pkl)")
def upload_model(file: UploadFile = File(...)):
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "model.pkl").name)
    if not name.lower().endswith((".pkl", ".pickle", ".joblib")):
        raise ApiError(400, "bad_extension", "Please upload a .pkl (or .joblib) file.")
    data = file.file.read()
    if len(data) < 100:
        raise ApiError(400, "empty_file", "The uploaded file is empty or too small.")
    tmp = BASE_DIR / (name + ".uploading")
    try:
        tmp.write_bytes(data)
        store.load(tmp)
        dest = BASE_DIR / name
        os.replace(tmp, dest)
        store.path = dest
    except ValueError as ex:
        raise ApiError(422, "invalid_model", str(ex))
    except OSError as ex:
        raise ApiError(500, "write_failed", "Could not save the model next to app.py: %s" % ex)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return model_info()


# ------------------------------------------------------------------ Prediction
@app.post("/api/predict", tags=["Prediction"], summary="Score one order")
def predict_one(order: OrderIn, explain: bool = True):
    """Returns the bad-review probability, risk level, delivery summary and the top reasons."""
    b = store.require()
    r = run_predictions(b, pd.DataFrame([_dump(order)]), explain=explain, detail=True)[0]
    if r.get("error"):
        raise ApiError(422, "invalid_input", r["error"])
    r["threshold"] = round(b["threshold"], 4)
    return r


@app.post("/api/predict/batch", tags=["Prediction"], summary="Score many orders (JSON)")
def predict_batch(body: BatchIn, explain: bool = False):
    """Rows sharing the same `order_id` are treated as items of one order (order-level features are computed)."""
    b = store.require()
    if not body.records:
        raise ApiError(422, "empty", "'records' is empty.")
    if len(body.records) > 20000:
        raise ApiError(413, "too_many_rows", "Maximum 20,000 records per JSON call - use /api/predict/file for more.")
    raw = pd.DataFrame([_dump(o) for o in body.records])
    res = run_predictions(b, raw, explain=explain, k=3)
    return {"summary": summarize(res, b["threshold"]), "results": res}


@app.post("/api/predict/file", tags=["Prediction"], summary="Score a CSV / Excel file")
def predict_file(file: UploadFile = File(...), explain: bool = True, format: str = "json"):
    """Upload a table with the same columns as your training data. `format=csv` returns your file + predictions."""
    b = store.require()
    raw = read_table(file.file.read(), file.filename)
    missing = [c for c in REQUIRED_INPUT if c not in raw.columns]
    if missing:
        raise ApiError(422, "missing_columns", "The file is missing required column(s): " + ", ".join(missing))
    skipped = bool(explain and len(raw) > 5000)
    res = run_predictions(b, raw, explain=explain and not skipped, k=3)
    if format.lower() == "csv":
        stem = Path(file.filename or "orders").stem
        return Response(results_to_csv(raw, res), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="%s_predictions.csv"' % stem})
    return {"summary": summarize(res, b["threshold"]), "explain_skipped": skipped, "results": res}


# =====================================================================================
#  DASHBOARD (HTML / CSS / JS, served from "/")
# =====================================================================================
INDEX_HTML = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Prediction</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%234f46e5'/%3E%3Cpath d='M8 17l5 5 11-12' stroke='white' stroke-width='3' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<style>
:root{--bg:#f3f5fb;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e3e8f2;--brand:#4f46e5;--brand2:#7c3aed;--soft:#eef0ff;--ok:#16a34a;--warn:#d97706;--bad:#dc2626;--okbg:#dcfce7;--warnbg:#fef3c7;--badbg:#fee2e2;--shadow:0 1px 2px rgba(15,23,42,.05),0 10px 30px rgba(15,23,42,.06);--r:16px}
:root[data-theme=dark]{--bg:#0a0f1f;--card:#121a31;--ink:#e8ecf7;--muted:#95a2c0;--line:#25315a;--soft:#1a2347;--okbg:#0f3a24;--warnbg:#3f2f0b;--badbg:#431a1f;--shadow:0 1px 2px rgba(0,0,0,.3)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Segoe UI",Tahoma,"Noto Sans Arabic",system-ui,-apple-system,Roboto,Arial,sans-serif}
button,input,select{font:inherit;color:inherit}
.top{position:sticky;top:0;z-index:30;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 28px;background:var(--card);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:44px;height:44px;border-radius:13px;background:linear-gradient(135deg,var(--brand),var(--brand2));display:grid;place-items:center;color:#fff;box-shadow:0 6px 16px rgba(79,70,229,.35)}
h1{font-size:18px;margin:0;line-height:1.25}.brand p{margin:0;color:var(--muted);font-size:12.5px}
.actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:99px;background:var(--soft);font-size:13px;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;background:var(--bad)}.pill.on .dot{background:var(--ok);box-shadow:0 0 0 4px rgba(22,163,74,.18)}
.ghost{border:1px solid var(--line);background:var(--card);border-radius:10px;padding:6px 12px;cursor:pointer;font-weight:600}
.ghost:hover{border-color:var(--brand);color:var(--brand)}
nav.tabs{display:flex;gap:6px;padding:14px 28px 0;max-width:1280px;margin:0 auto;overflow-x:auto}
nav.tabs button{border:0;background:transparent;padding:10px 16px;border-radius:10px 10px 0 0;cursor:pointer;font-weight:700;color:var(--muted);border-bottom:3px solid transparent;white-space:nowrap}
nav.tabs button.on{color:var(--brand);border-bottom-color:var(--brand)}
main{max-width:1280px;margin:0 auto;padding:18px 28px 60px}
.grid2{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:20px;align-items:start}
@media(max-width:980px){.grid2{grid-template-columns:1fr}.top,main,nav.tabs{padding-inline:16px}}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);padding:20px;margin-bottom:20px}
.card-h{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
h2{font-size:16px;margin:0}h3{font-size:13px;margin:18px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.chip{border:1px solid var(--line);background:var(--card);border-radius:99px;padding:4px 12px;font-size:13px;cursor:pointer;font-weight:600}
.chip:hover,.chip.on{background:var(--soft);border-color:var(--brand);color:var(--brand)}
.fields{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
label.f{display:flex;flex-direction:column;gap:4px;font-size:13px;font-weight:600;color:var(--muted)}
label.f i{color:var(--bad);font-style:normal}
input,select{width:100%;padding:9px 11px;border:1px solid var(--line);border-radius:10px;background:var(--bg);outline:none;font-weight:500}
input:focus,select:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(79,70,229,.15);background:var(--card)}
.primary{border:0;border-radius:12px;padding:12px 26px;cursor:pointer;font-weight:800;color:#fff;background:linear-gradient(135deg,var(--brand),var(--brand2));box-shadow:0 8px 20px rgba(79,70,229,.3)}
.primary:disabled{opacity:.6;cursor:wait}.foot{margin-top:18px;display:flex;gap:10px;align-items:center}
.empty{text-align:center;color:var(--muted);padding:46px 10px}.empty svg{opacity:.5;margin-bottom:8px}
.gauge{width:100%;max-width:320px;display:block;margin:0 auto}.g-track{fill:none;stroke:var(--line);stroke-width:14;stroke-linecap:round}
.g-val{fill:none;stroke:var(--c);stroke-width:14;stroke-linecap:round}.g-thr{stroke:var(--ink);stroke-width:2.5;opacity:.7}
.g-num{font-size:30px;font-weight:800;text-anchor:middle;fill:var(--ink)}.g-sub{font-size:9.5px;text-anchor:middle;fill:var(--muted)}
.gauge.low{--c:var(--ok)}.gauge.medium{--c:var(--warn)}.gauge.high{--c:var(--bad)}
.badge{display:inline-block;padding:3px 12px;border-radius:99px;font-weight:800;font-size:13px}
.badge.low{background:var(--okbg);color:var(--ok)}.badge.medium{background:var(--warnbg);color:var(--warn)}.badge.high{background:var(--badbg);color:var(--bad)}.badge.err{background:var(--line);color:var(--muted)}
.center{text-align:center}.verdict{font-size:17px;font-weight:800;margin:6px 0 2px}.muted{color:var(--muted)}.small{font-size:13px}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0}.kv div{background:var(--bg);border-radius:12px;padding:10px;text-align:center}.kv b{display:block;font-size:17px}.kv span{font-size:12px;color:var(--muted)}
.advice{border-radius:12px;padding:10px 14px;font-size:14px;margin:8px 0}.advice.high{background:var(--badbg)}.advice.medium{background:var(--warnbg)}.advice.low{background:var(--okbg)}
.warnbox{background:var(--warnbg);border-radius:12px;padding:8px 12px;font-size:13px;margin:8px 0}
.fx{margin:10px 0}.fx-l{display:flex;justify-content:space-between;gap:10px;font-size:13.5px}.fx-l span{color:var(--muted);font-variant-numeric:tabular-nums}
.fx-bar{height:8px;background:var(--bg);border-radius:9px;overflow:hidden;margin-top:4px}.fx-bar i{display:block;height:100%;border-radius:9px}.fx-bar i.up{background:var(--bad)}.fx-bar i.dn{background:var(--ok)}
.tag{font-size:11px;font-weight:700;margin-inline-start:6px}.tag.up{color:var(--bad)}.tag.dn{color:var(--ok)}
details{margin-top:14px}summary{cursor:pointer;font-weight:700;color:var(--brand);font-size:13.5px}
pre{background:#0b1226;color:#d6def5;border-radius:12px;padding:14px;overflow:auto;font-size:12.5px;line-height:1.55;direction:ltr;text-align:left;margin:8px 0}
.drop{border:2px dashed var(--line);border-radius:var(--r);padding:34px 16px;text-align:center;cursor:pointer;transition:.15s;background:var(--bg)}
.drop:hover,.drop.over{border-color:var(--brand);background:var(--soft)}.drop b{display:block;font-size:16px;margin:8px 0 2px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;box-shadow:var(--shadow)}.stat b{font-size:24px;display:block}.stat span{color:var(--muted);font-size:13px}
.hist{display:flex;align-items:flex-end;gap:5px;height:90px}.hist div{flex:1;border-radius:6px 6px 0 0;background:var(--brand);opacity:.85;min-height:2px}.hist div.hot{background:var(--bad)}
.hist-x{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:4px;direction:ltr}
.tw{overflow:auto;max-height:560px;border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:9px 12px;text-align:start;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;background:var(--soft);font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}td.wrap{white-space:normal;min-width:220px}
.pbar{display:inline-block;width:90px;height:7px;background:var(--line);border-radius:9px;vertical-align:middle;margin-inline-end:8px;overflow:hidden}.pbar i{display:block;height:100%;background:var(--brand)}
.ep{display:grid;grid-template-columns:78px 1fr;gap:4px 12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--line)}.ep code{direction:ltr;text-align:left;font-weight:700}.ep span.d{grid-column:2;color:var(--muted);font-size:13px}
.m{display:inline-block;text-align:center;padding:2px 0;border-radius:7px;font-size:11.5px;font-weight:800;color:#fff;width:62px}.m.get{background:#0ea5e9}.m.post{background:#16a34a}
.steps{counter-reset:s;list-style:none;padding:0;margin:0}.steps li{counter-increment:s;display:flex;gap:12px;margin:10px 0}.steps li:before{content:counter(s);flex:0 0 28px;height:28px;border-radius:50%;background:var(--soft);color:var(--brand);font-weight:800;display:grid;place-items:center}
.spin{width:18px;height:18px;border:3px solid var(--line);border-top-color:var(--brand);border-radius:50%;animation:sp .7s linear infinite;display:inline-block;vertical-align:middle}@keyframes sp{to{transform:rotate(360deg)}}
#toast{position:fixed;bottom:22px;inset-inline:0;margin:auto;width:fit-content;max-width:92vw;background:#111827;color:#fff;padding:11px 18px;border-radius:12px;box-shadow:0 12px 30px rgba(0,0,0,.35);z-index:99;display:none;font-size:14px}#toast.err{background:#b91c1c}
[hidden]{display:none!important}.ltr{direction:ltr;text-align:left}
</style>
</head>
<body>
<header class="top">
  <div class="brand">
    <div class="logo"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l2.9 6.3 6.9.7-5.2 4.6 1.5 6.8L12 17l-6.1 3.4 1.5-6.8L2.2 9l6.9-.7z"/></svg></div>
    <div><h1 data-i="title"></h1></div>
  </div>
  <div class="actions">
    <span class="pill" id="pill"><i class="dot"></i><span id="pillText"></span></span>
    <button class="ghost" id="btnLang">EN</button>
    <button class="ghost" id="btnTheme" title="Theme">◐</button>
  </div>
</header>

<nav class="tabs" id="tabs" hidden>
  <button data-tab="predict" class="on" data-i="tab_predict"></button>
  <button data-tab="batch" data-i="tab_batch"></button>
  <button data-tab="model" data-i="tab_model"></button>
  <button data-tab="api" data-i="tab_api"></button>
</nav>

<main>
  <section id="onboard" hidden>
    <div class="card" style="max-width:720px;margin:30px auto">
      <h2 data-i="ob_title" style="font-size:22px"></h2>
      <p class="muted" data-i="ob_text"></p>
      <div class="drop" id="dropOnboard"></div>
      <p id="onboardErr" class="warnbox" hidden></p>
      <ol class="steps" style="margin-top:18px">
        <li><span data-i="ob_s1"></span></li><li><span data-i="ob_s2"></span></li><li><span data-i="ob_s3"></span></li>
      </ol>
    </div>
  </section>

  <section id="tab-predict" class="tab">
    <div class="grid2">
      <div class="card">
        <div class="card-h"><h2 data-i="enter_order"></h2>
          <div class="row"><button type="button" class="chip" data-ex="ok" data-i="ex_ok"></button><button type="button" class="chip" data-ex="late" data-i="ex_late"></button><button type="button" class="chip" id="btnClear" data-i="clear"></button></div></div>
        <form id="form" autocomplete="off"></form>
      </div>
      <div class="card" id="resultCard" style="position:sticky;top:88px"></div>
    </div>
  </section>

  <section id="tab-batch" class="tab" hidden>
    <div class="card">
      <div class="card-h"><h2 data-i="batch_title"></h2>
        <div class="row"><label class="row small"><input type="checkbox" id="chkExplain" checked style="width:auto"> <span data-i="batch_explain"></span></label>
        <button class="ghost" id="btnTemplate" data-i="template"></button></div></div>
      <p class="muted small" data-i="batch_help"></p>
      <div class="drop" id="dropBatch"></div>
    </div>
    <div id="batchOut"></div>
  </section>

  <section id="tab-model" class="tab" hidden><div id="modelBody"></div></section>
  <section id="tab-api" class="tab" hidden><div id="apiBody"></div></section>
</main>
<div id="toast"></div>

<script>
const I={
title:["Review Prediction","Review Prediction"],
tab_predict:["Single order","أوردر واحد"],tab_batch:["Batch file","ملف كامل"],tab_model:["Model","الموديل"],tab_api:["API","الـ API"],
no_model:["No model loaded","مفيش موديل متحمّل"],model_ready:["Model ready","الموديل جاهز"],
ob_title:["Add your model to get started","ضيف الموديل علشان تبدأ"],
ob_text:["Drop the .pkl file your notebook saved (bad_review_pipeline.pkl). Nothing else is needed.","اسحب ملف الـ .pkl اللي النوت بوك حفظه (bad_review_pipeline.pkl) وسيبه هنا. مش محتاج حاجة تانية."],
ob_s1:["Drag the .pkl file into the box above (or click it to browse).","اسحب ملف الـ .pkl جوه المربع اللي فوق (أو اضغط عليه واختار الملف)."],
ob_s2:["The model is checked automatically and saved next to app.py.","الموديل بيتفحص تلقائيًا وبيتحفظ جنب app.py."],
ob_s3:["Next time you start the app it loads by itself.","المرة الجاية لما تشغّل البرنامج هيتحمّل لوحده."],
drop_t:["Drop the model file here","اسحب ملف الموديل هنا"],drop_s:["or click to browse (.pkl)","أو اضغط للاختيار (.pkl)"],
drop_batch_t:["Drop your orders file here","اسحب ملف الأوردرات هنا"],drop_batch_s:["CSV or Excel — or click to browse","CSV أو Excel — أو اضغط للاختيار"],
uploading:["Checking the file…","بنفحص الملف…"],model_loaded_ok:["Model loaded successfully","الموديل اتحمّل بنجاح"],
enter_order:["Order details","بيانات الأوردر"],ex_ok:["Example: on-time","مثال: تسليم في الميعاد"],ex_late:["Example: late","مثال: تسليم متأخر"],clear:["Clear","مسح"],
g_dates:["Dates","التواريخ"],g_order:["Order & pricing","الأوردر والسعر"],g_product:["Product","المنتج"],g_geo:["Location","الموقع"],g_pay:["Payment","الدفع"],g_ids:["Identifiers","المعرّفات"],
predict:["Predict","توقّع"],predicting:["Predicting…","بنحسب…"],
fill_hint:["Fill the required fields (*) and press Predict. More details = more accurate.","املا الحقول المطلوبة (*) واضغط توقّع. كل ما زادت التفاصيل زادت الدقة."],
bad_prob:["probability of a bad review","احتمال التقييم السيئ"],
lvl_high:["High risk","خطر عالي"],lvl_medium:["Medium risk","خطر متوسط"],lvl_low:["Low risk","خطر منخفض"],
v_bad:["Likely to leave a BAD review","غالبًا هيسيب تقييم سيئ"],v_good:["Likely to leave a good review","غالبًا هيسيب تقييم كويس"],
act_high:["Act now: contact the customer, apologise or offer a voucher before they review.","اتصرف دلوقتي: كلّم العميل واعتذر أو اعرض عليه كوبون قبل ما يقيّم."],
act_medium:["Keep an eye on it: a short follow-up message could help.","خليك واخد بالك: رسالة متابعة قصيرة ممكن تفرق."],
act_low:["No action needed.","مفيش داعي لأي إجراء."],
delivered_in:["Delivered in","اتسلّم في"],promised:["Promised","الموعود"],delay:["Delay","التأخير"],days:["days","يوم"],
why:["Why this score?","ليه النتيجة دي؟"],raises:["raises risk","بيزوّد الخطر"],lowers:["lowers risk","بيقلّل الخطر"],
threshold:["Decision threshold","حد القرار"],raw_json:["Request / response (JSON)","الطلب والرد (JSON)"],copy:["Copy","نسخ"],copied:["Copied","تم النسخ"],
batch_title:["Score a whole file","قيّم ملف كامل"],batch_explain:["include reasons","اعرض الأسباب"],template:["Download template CSV","نزّل ملف CSV نموذجي"],
batch_help:["Columns must match your training data. Rows with the same order_id are treated as one order. Required: purchase, estimated & delivered dates, price, freight_value.","الأعمدة لازم تطابق بيانات التدريب. الصفوف بنفس order_id بتتعامل كأوردر واحد. المطلوب: تواريخ الشراء والمتوقع والتسليم + price + freight_value."],
scoring:["Scoring…","بنقيّم…"],rows:["Orders scored","أوردرات تم تقييمها"],high_risk:["High risk","خطر عالي"],avg_prob:["Average probability","متوسط الاحتمال"],invalid:["Rows skipped","صفوف اتجاهلت"],
dist:["Probability distribution","توزيع الاحتمالات"],all:["All","الكل"],download:["Download results (CSV)","نزّل النتائج (CSV)"],
c_row:["#","#"],c_order:["Order","الأوردر"],c_prob:["Probability","الاحتمال"],c_risk:["Risk","الخطر"],c_drivers:["Main reasons","أهم الأسباب"],
showing:["Showing the first {n} rows (sorted by risk). The CSV download has everything.","معروض أول {n} صف (مرتبين بالخطر). ملف الـ CSV فيه كل حاجة."],
m_models:["Models in ensemble","موديلات في المجموعة"],m_features:["Features","عدد الـ features"],m_thr:["Decision threshold","حد القرار"],m_start:["Training data starts","بداية بيانات التدريب"],m_file:["File","الملف"],m_loaded:["Loaded at","وقت التحميل"],
m_imp:["What drives predictions most","إيه أكتر حاجة بتأثر على التوقع"],m_replace:["Replace the model","استبدل الموديل"],m_thr_note:["Risk is HIGH when the probability reaches the threshold, MEDIUM from 60% of it.","الخطر يبقى عالي لما الاحتمال يوصل لحد القرار، ومتوسط من 60% منه."],
m_params:["Training parameters","إعدادات التدريب"],m_cats:["Known categories","القيم المعروفة"],
api_title:["Use it from your own code","استخدمه من الكود بتاعك"],api_base:["Base URL","العنوان الأساسي"],api_docs:["Interactive docs (Swagger)","التوثيق التفاعلي (Swagger)"],api_endpoints:["Endpoints","الـ Endpoints"],api_examples:["Examples","أمثلة"],
d_predict:["Score one order — probability, risk level and reasons","قيّم أوردر واحد — الاحتمال والخطر والأسباب"],d_batch:["Score many orders sent as JSON","قيّم أوردرات كتير مبعوتة JSON"],d_file:["Score a CSV/Excel file (add ?format=csv to get a file back)","قيّم ملف CSV/Excel (زوّد ?format=csv علشان ترجّع ملف)"],d_model:["Model details & feature importance","تفاصيل الموديل وأهمية الـ features"],d_schema:["Fields the model accepts + example","الحقول اللي الموديل بيقبلها + مثال"],d_upload:["Upload / replace the model file","ارفع أو استبدل ملف الموديل"],d_health:["Service status","حالة الخدمة"],
ask_key:["This server needs an API key:","السيرفر محتاج API key:"],unauth:["Unauthorized","غير مصرّح"],err:["Error","خطأ"]
};
const store={get(k){try{return localStorage.getItem(k)}catch(e){return null}},set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};
let lang=store.get('lang')||'ar';const t=k=>(I[k]||[k,k])[lang==='ar'?1:0];
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const S={health:null,model:null,schema:null,batch:null,file:null,filter:'all',last:null};
const fmt=v=>v==null?'—':(typeof v==='number'?(Math.abs(v)>=1000?v.toLocaleString('en',{maximumFractionDigits:0}):String(Math.round(v*100)/100)):esc(v));
let tmr;function toast(m,err){const e=$('#toast');e.textContent=m;e.className=err?'err':'';e.style.display='block';clearTimeout(tmr);tmr=setTimeout(()=>e.style.display='none',err?6000:2600)}
function copy(txt){const done=()=>toast(t('copied'));if(navigator.clipboard&&window.isSecureContext)navigator.clipboard.writeText(txt).then(done);else{const a=document.createElement('textarea');a.value=txt;document.body.appendChild(a);a.select();document.execCommand('copy');a.remove();done()}}

async function api(path,opt={}){
  const h=Object.assign({},opt.headers||{}),k=store.get('apikey');if(k)h['X-API-Key']=k;
  const r=await fetch(path,Object.assign({},opt,{headers:h}));
  if(r.status===401){const nk=prompt(t('ask_key'));if(nk){store.set('apikey',nk);return api(path,opt)}throw new Error(t('unauth'))}
  if(!r.ok){let m=r.status+' '+r.statusText;try{const j=await r.json();m=j.error?j.error.message:(Array.isArray(j.detail)?j.detail.map(d=>(d.loc||[]).slice(1).join('.')+': '+d.msg).join(' | '):(j.detail||m))}catch(e){}throw new Error(m)}
  return (r.headers.get('content-type')||'').includes('json')?r.json():r;
}

/* ---------- theme / language ---------- */
function setTheme(x){document.documentElement.dataset.theme=x;store.set('theme',x)}
setTheme(store.get('theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light'));
$('#btnTheme').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
function setLang(l){lang=l;store.set('lang',l);document.documentElement.lang=l;document.documentElement.dir=l==='ar'?'rtl':'ltr';$('#btnLang').textContent=l==='ar'?'EN':'ع';$$('[data-i]').forEach(e=>e.textContent=t(e.dataset.i));renderAll()}
$('#btnLang').onclick=()=>setLang(lang==='ar'?'en':'ar');

/* ---------- tabs ---------- */
function showTab(n){$$('#tabs button').forEach(b=>b.classList.toggle('on',b.dataset.tab===n));$$('.tab').forEach(s=>s.hidden=s.id!=='tab-'+n)}
$$('#tabs button').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));

/* ---------- drop zones ---------- */
function dropzone(el,title,sub,accept,onFile){
  el.innerHTML=`<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--brand)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4m0 0L7 9m5-5l5 5"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/></svg><b>${esc(title)}</b><span class="muted small">${esc(sub)}</span><input type="file" accept="${accept}" hidden>`;
  const inp=$('input',el);el.onclick=()=>inp.click();inp.onchange=()=>{if(inp.files[0])onFile(inp.files[0]);inp.value=''};
  el.ondragenter=el.ondragover=e=>{e.preventDefault();el.classList.add('over')};
  el.ondragleave=e=>{e.preventDefault();el.classList.remove('over')};
  el.ondrop=e=>{e.preventDefault();el.classList.remove('over');if(e.dataTransfer.files[0])onFile(e.dataTransfer.files[0])};
}
async function uploadModel(file){
  const box=$('#onboardErr');box.hidden=true;toast(t('uploading'));
  try{const fd=new FormData();fd.append('file',file);await api('/api/model/upload',{method:'POST',body:fd});toast(t('model_loaded_ok'));await loadState();showTab('predict')}
  catch(e){box.textContent=e.message;box.hidden=false;toast(e.message,true)}
}

/* ---------- state ---------- */
async function loadState(){
  S.health=await api('/api/health');S.model=S.schema=null;
  if(S.health.model_loaded){[S.model,S.schema]=await Promise.all([api('/api/model'),api('/api/schema')])}
  renderAll();
}
function renderAll(){
  if(!S.health){$('#onboard').hidden=true;$('#tabs').hidden=true;return}
  const ok=S.health&&S.health.model_loaded;
  $('#pill').className='pill'+(ok?' on':'');
  $('#pillText').textContent=ok?`${t('model_ready')} · ${S.model.n_models} ${lang==='ar'?'موديلات':'models'}`:t('no_model');
  $('#onboard').hidden=!!ok;$('#tabs').hidden=!ok;$$('.tab').forEach(s=>{if(!ok)s.hidden=true});
  if(!ok){dropzone($('#dropOnboard'),t('drop_t'),t('drop_s'),'.pkl,.joblib,.pickle',uploadModel);
    if(S.health&&S.health.model_error){const b=$('#onboardErr');b.textContent=S.health.model_error;b.hidden=false}return}
  if($$('.tab').every(s=>s.hidden))showTab('predict');
  buildForm();renderResult();renderBatch();renderModel();renderApi();
  dropzone($('#dropBatch'),t('drop_batch_t'),t('drop_batch_s'),'.csv,.tsv,.txt,.xlsx,.xls',runBatch);
}

/* ---------- single prediction ---------- */
const GROUPS=['dates','order','product','geo','pay','ids'];
function buildForm(){
  const keep={};$$('#form [name]').forEach(e=>keep[e.name]=e.value);
  const fs=S.schema.fields.filter(f=>f.used);let h='';
  GROUPS.forEach(g=>{const list=fs.filter(f=>f.group===g);if(!list.length)return;
    h+=`<h3>${t('g_'+g)}</h3><div class="fields">`+list.map(f=>{
      const lb=f.label[lang]+(f.required?' <i>*</i>':'');const nm=`name="${f.name}"`+(f.required?' required':'');
      let inp;if(f.kind==='datetime')inp=`<input type="datetime-local" ${nm}>`;
      else if(f.kind==='number')inp=`<input type="number" step="any" ${nm}>`;
      else if(f.kind==='combo')inp=`<input list="dl_${f.name}" ${nm}><datalist id="dl_${f.name}">${(f.options||[]).map(o=>`<option value="${esc(o)}">`).join('')}</datalist>`;
      else inp=`<input type="text" ${nm}>`;
      return `<label class="f">${lb}${inp}</label>`}).join('')+`</div>`});
  h+=`<div class="foot"><button class="primary" id="btnPredict" type="submit">${t('predict')}</button><span class="muted small">${t('fill_hint')}</span></div>`;
  $('#form').innerHTML=h;$$('#form [name]').forEach(e=>{if(keep[e.name]!=null)e.value=keep[e.name]});
}
$('#form').addEventListener('submit',predict);
$('#btnClear').onclick=()=>{$('#form').reset();S.last=null;renderResult()};
const p2=n=>String(n).padStart(2,'0');
const toLocal=d=>`${d.getFullYear()}-${p2(d.getMonth()+1)}-${p2(d.getDate())}T${p2(d.getHours())}:${p2(d.getMinutes())}`;
$$('[data-ex]').forEach(b=>b.onclick=()=>{
  const ex=Object.assign({},S.schema.example),base=new Date('2018-03-10T14:30'),add=d=>toLocal(new Date(base.getTime()+d*864e5)),late=b.dataset.ex==='late';
  ex.order_purchase_timestamp=add(0);ex.order_estimated_delivery_date=add(20);ex.order_delivered_customer_date=add(late?34:9);
  $$('#form [name]').forEach(e=>{e.value=ex[e.name]!=null?ex[e.name]:''});
});
function collect(){const r={};$$('#form [name]').forEach(e=>{const v=e.value.trim();if(v==='')return;r[e.name]=e.type==='number'?Number(v):v});return r}
async function predict(e){
  e.preventDefault();const btn=$('#btnPredict');btn.disabled=true;btn.textContent=t('predicting');const body=collect();
  try{const res=await api('/api/predict?explain=true',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});S.last={res,body};renderResult()}
  catch(err){toast(err.message,true)}finally{btn.disabled=false;btn.textContent=t('predict')}
}
function gauge(p,thr,lvl){
  const L=251.33,a=Math.PI*thr,x1=100-70*Math.cos(a),y1=100-70*Math.sin(a),x2=100-93*Math.cos(a),y2=100-93*Math.sin(a);
  return `<svg viewBox="0 0 200 118" class="gauge ${lvl}"><path d="M20 100A80 80 0 0 1 180 100" class="g-track"/><path d="M20 100A80 80 0 0 1 180 100" class="g-val" stroke-dasharray="${L}" stroke-dashoffset="${L*(1-p)}"/><line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="g-thr"/><text x="100" y="92" class="g-num">${(p*100).toFixed(1)}%</text><text x="100" y="112" class="g-sub">${t('bad_prob')}</text></svg>`;
}
function renderResult(){
  const c=$('#resultCard');if(!S.last){c.innerHTML=`<div class="empty"><svg width="54" height="54" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/></svg><p>${t('fill_hint')}</p></div>`;return}
  const r=S.last.res,thr=r.threshold,lvl=r.risk_level,d=r.delivery||{},mx=Math.max(1e-6,...(r.top_factors||[]).map(f=>Math.abs(f.impact)));
  c.innerHTML=`<div class="center">${gauge(r.probability,thr,lvl)}<span class="badge ${lvl}">${t('lvl_'+lvl)}</span><div class="verdict">${t(r.predicted_bad_review?'v_bad':'v_good')}</div><div class="muted small">${t('threshold')}: ${(thr*100).toFixed(1)}%</div></div>
  <div class="kv"><div><b>${fmt(d.delivery_days)}</b><span>${t('delivered_in')} (${t('days')})</span></div><div><b>${fmt(d.promised_days)}</b><span>${t('promised')} (${t('days')})</span></div><div><b>${d.delay_days>0?'+':''}${fmt(d.delay_days)}</b><span>${t('delay')} (${t('days')})</span></div></div>
  <div class="advice ${lvl}">${t('act_'+lvl)}</div>${(r.warnings||[]).map(w=>`<div class="warnbox">⚠ ${esc(w)}</div>`).join('')}
  <h3>${t('why')}</h3>${(r.top_factors||[]).map(f=>`<div class="fx"><div class="fx-l"><span style="color:var(--ink);font-weight:600">${esc(f.label[lang])}<em class="tag ${f.impact>0?'up':'dn'}">${f.impact>0?'▲ '+t('raises'):'▼ '+t('lowers')}</em></span><span>${fmt(f.value)}</span></div><div class="fx-bar"><i class="${f.impact>0?'up':'dn'}" style="width:${Math.max(4,Math.abs(f.impact)/mx*100)}%"></i></div></div>`).join('')}
  <details><summary>${t('raw_json')}</summary><pre id="rawJson">${esc(JSON.stringify({request:S.last.body,response:r},null,2))}</pre><button class="ghost" onclick="copy($('#rawJson').textContent)">${t('copy')}</button></details>`;
}

/* ---------- batch ---------- */
async function runBatch(file){
  S.file=file;S.filter='all';$('#batchOut').innerHTML=`<div class="card center"><span class="spin"></span> ${t('scoring')}</div>`;
  try{const fd=new FormData();fd.append('file',file);S.batch=await api('/api/predict/file?explain='+$('#chkExplain').checked,{method:'POST',body:fd});renderBatch()}
  catch(e){$('#batchOut').innerHTML='';toast(e.message,true)}
}
function renderBatch(){
  const o=$('#batchOut');if(!S.batch){o.innerHTML='';return}
  const s=S.batch.summary,res=S.batch.results,thr=s.threshold,mxh=Math.max(1,...s.histogram);
  const rows=res.filter(r=>S.filter==='all'?true:(S.filter==='err'?r.error:r.risk_level===S.filter)).sort((a,b)=>(b.probability??-1)-(a.probability??-1)).slice(0,300);
  const drv=r=>(r.top_factors||[]).filter(f=>f.impact>0).slice(0,2).map(f=>`${esc(f.label[lang])} (${fmt(f.value)})`).join('، ');
  o.innerHTML=`<div class="stats"><div class="stat"><b>${s.scored.toLocaleString()}</b><span>${t('rows')}</span></div><div class="stat"><b style="color:var(--bad)">${s.high.toLocaleString()}</b><span>${t('high_risk')} (${s.scored?Math.round(s.high/s.scored*100):0}%)</span></div><div class="stat"><b>${s.mean_probability==null?'—':(s.mean_probability*100).toFixed(1)+'%'}</b><span>${t('avg_prob')}</span></div><div class="stat"><b>${s.invalid}</b><span>${t('invalid')}</span></div></div>
  <div class="card"><div class="card-h"><h2>${t('dist')}</h2><button class="primary" id="btnDl">${t('download')}</button></div>
  <div class="hist">${s.histogram.map((v,i)=>`<div class="${i/10>=thr-1e-9?'hot':''}" style="height:${v/mxh*100}%" title="${i*10}–${i*10+10}%: ${v}"></div>`).join('')}</div><div class="hist-x"><span>0%</span><span>50%</span><span>100%</span></div></div>
  <div class="card"><div class="row" style="margin-bottom:12px">${['all','high','medium','low'].concat(s.invalid?['err']:[]).map(f=>`<button class="chip ${S.filter===f?'on':''}" data-f="${f}">${f==='all'?t('all'):(f==='err'?t('invalid'):t('lvl_'+f))}</button>`).join('')}</div>
  <div class="tw"><table><thead><tr><th>${t('c_row')}</th><th>${t('c_order')}</th><th>${t('c_prob')}</th><th>${t('c_risk')}</th><th>${t('c_drivers')}</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${r.row}</td><td class="ltr">${esc(r.order_id)}</td><td>${r.probability==null?'—':`<span class="pbar"><i style="width:${r.probability*100}%"></i></span>${(r.probability*100).toFixed(1)}%`}</td><td><span class="badge ${r.error?'err':r.risk_level}">${r.error?esc(t('err')):t('lvl_'+r.risk_level)}</span></td><td class="wrap muted small">${r.error?esc(r.error):drv(r)}</td></tr>`).join('')}</tbody></table></div>
  ${res.length>300?`<p class="muted small">${t('showing').replace('{n}',300)}</p>`:''}</div>`;
  $$('[data-f]',o).forEach(b=>b.onclick=()=>{S.filter=b.dataset.f;renderBatch()});
  $('#btnDl').onclick=async()=>{try{const fd=new FormData();fd.append('file',S.file);const r=await api('/api/predict/file?format=csv&explain='+$('#chkExplain').checked,{method:'POST',body:fd});
    const u=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=u;a.download=S.file.name.replace(/\.[^.]+$/,'')+'_predictions.csv';a.click();URL.revokeObjectURL(u)}catch(e){toast(e.message,true)}};
}
$('#btnTemplate').onclick=()=>{
  const fs=S.schema.fields.filter(f=>f.used),ex=S.schema.example,q=v=>/[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;
  const csv=fs.map(f=>f.name).join(',')+'\n'+fs.map(f=>q(String(ex[f.name]!=null?ex[f.name]:''))).join(',')+'\n';
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv'}));a.download='orders_template.csv';a.click();
};

/* ---------- model tab ---------- */
function renderModel(){
  const m=S.model;if(!m)return;const mx=Math.max(1e-6,...m.importance.map(x=>x.importance));
  $('#modelBody').innerHTML=`<div class="stats">
  <div class="stat"><b>${m.n_models}</b><span>${t('m_models')}</span></div><div class="stat"><b>${m.n_features}</b><span>${t('m_features')}</span></div>
  <div class="stat"><b>${(m.threshold*100).toFixed(1)}%</b><span>${t('m_thr')}</span></div><div class="stat"><b style="font-size:18px">${esc(m.start_date)}</b><span>${t('m_start')}</span></div></div>
  <div class="grid2"><div class="card"><h2>${t('m_imp')}</h2><div style="margin-top:12px">${m.importance.slice(0,15).map(x=>`<div class="fx"><div class="fx-l"><span style="color:var(--ink)">${esc(x.label[lang])}</span><span>${(x.importance*100).toFixed(1)}%</span></div><div class="fx-bar"><i style="background:var(--brand);width:${x.importance/mx*100}%"></i></div></div>`).join('')}</div></div>
  <div><div class="card"><h2>${t('m_replace')}</h2><p class="muted small">${t('m_thr_note')}</p><div class="drop" id="dropModel"></div></div>
  <div class="card"><div class="small"><p><b>${t('m_file')}:</b> <span class="ltr">${esc(m.file)} ${m.size_mb!=null?'('+m.size_mb+' MB)':''}</span></p><p><b>${t('m_loaded')}:</b> ${esc(m.loaded_at.replace('T',' '))}</p>
  <p><b>${t('m_cats')}:</b> ${Object.entries(m.categorical).map(([k,v])=>`${esc(k)} (${v})`).join('، ')||'—'}</p></div>
  <details><summary>${t('m_params')}</summary><pre>${esc(JSON.stringify(m.best_params,null,2))}</pre></details></div></div></div>`;
  dropzone($('#dropModel'),t('drop_t'),t('drop_s'),'.pkl,.joblib,.pickle',uploadModel);
}

/* ---------- API tab ---------- */
function renderApi(){
  if(!S.schema)return;const B=location.origin,ex=S.schema.example,js=JSON.stringify(ex,null,2);
  const eps=[['post','/api/predict','d_predict'],['post','/api/predict/batch','d_batch'],['post','/api/predict/file','d_file'],['get','/api/model','d_model'],['get','/api/schema','d_schema'],['post','/api/model/upload','d_upload'],['get','/api/health','d_health']];
  const key=S.health.auth_required?' \\\n  -H "X-API-Key: YOUR_KEY"':'';const pk=S.health.auth_required?', headers={"X-API-Key": "YOUR_KEY"}':'';
  const code={
   curl:`curl -X POST ${B}/api/predict \\\n  -H "Content-Type: application/json"${key} \\\n  -d '${JSON.stringify(ex)}'\n\n# whole file -> CSV with predictions\ncurl -X POST "${B}/api/predict/file?format=csv" ${key.trim()?key.trim()+' ':''}\\\n  -F "file=@orders.csv" -o predictions.csv`,
   python:`import requests\n\norder = ${js.replace(/null/g,'None')}\n\nr = requests.post("${B}/api/predict", json=order${pk})\nres = r.json()\nprint(res["probability"], res["risk_level"])\n\n# whole file\nwith open("orders.csv", "rb") as f:\n    r = requests.post("${B}/api/predict/file", files={"file": f}${pk})\nfor row in r.json()["results"][:5]:\n    print(row["order_id"], row["probability"], row["risk_level"])`,
   js:`const res = await fetch("${B}/api/predict", {\n  method: "POST",\n  headers: {"Content-Type": "application/json"${S.health.auth_required?', "X-API-Key": "YOUR_KEY"':''}},\n  body: JSON.stringify(${JSON.stringify(ex)})\n}).then(r => r.json());\nconsole.log(res.probability, res.risk_level);`};
  $('#apiBody').innerHTML=`<div class="grid2"><div class="card"><div class="card-h"><h2>${t('api_endpoints')}</h2><a class="ghost" style="text-decoration:none;color:inherit" href="/docs" target="_blank">${t('api_docs')} ↗</a></div>
  <p class="small"><b>${t('api_base')}:</b> <code class="ltr">${esc(B)}</code></p>${eps.map(([m,p,d])=>`<div class="ep"><span class="m ${m}">${m.toUpperCase()}</span><code>${p}</code><span class="d">${t(d)}</span></div>`).join('')}</div>
  <div class="card"><div class="card-h"><h2>${t('api_examples')}</h2><div class="row">${['curl','python','js'].map((k,i)=>`<button class="chip ${i?'':'on'}" data-code="${k}">${k==='js'?'JavaScript':k}</button>`).join('')}</div></div><pre id="codeBox"></pre><button class="ghost" id="btnCopyCode">${t('copy')}</button></div></div>`;
  const show=k=>{$('#codeBox').textContent=code[k];$$('[data-code]').forEach(b=>b.classList.toggle('on',b.dataset.code===k))};show('curl');
  $$('[data-code]').forEach(b=>b.onclick=()=>show(b.dataset.code));$('#btnCopyCode').onclick=()=>copy($('#codeBox').textContent);
}

setLang(lang);
loadState().catch(e=>toast(e.message,true));
</script>
</body>
</html>
"""


# =====================================================================================
#  ENTRY POINT
# =====================================================================================
def _free_port(host, port):
    for p in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return port


def main():
    ap = argparse.ArgumentParser(description="Review Prediction - API + dashboard")
    ap.add_argument("--model", help="path to the .pkl file (default: auto-detect next to app.py)")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your network")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=os.getenv("API_KEY"), help="require this key in the X-API-Key header")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    a = ap.parse_args()
    CFG["model_path"], CFG["api_key"] = a.model, a.api_key
    import uvicorn
    port = _free_port(a.host, a.port)
    shown = "127.0.0.1" if a.host in ("0.0.0.0", "127.0.0.1") else a.host
    url = "http://%s:%d" % (shown, port)
    print("\n  Review Prediction v%s" % VERSION)
    print("  Dashboard : %s" % url)
    print("  API docs  : %s/docs" % url)
    if a.host == "0.0.0.0":
        try:
            print("  Network   : http://%s:%d" % (socket.gethostbyname(socket.gethostname()), port))
        except OSError:
            pass
    if a.api_key:
        print("  API key   : required (X-API-Key header)")
    print("  Press CTRL+C to stop.\n")
    if not a.no_browser:
        threading.Timer(1.6, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
