"""
Task 3 — Serving predictions on a web interface.

Run locally with:
    pip install flask pandas numpy scikit-learn joblib matplotlib
    python app.py
then open http://127.0.0.1:5000

Expects a serialized pipeline produced by 02_modeling.py, e.g.
    model-14-07-2026-01-00-39-00.pkl
placed in the same folder as this file (or point MODEL_PATH at it).
"""
import io
import glob
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from flask import Flask, render_template, request, send_file, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rossmann_app")

app = Flask(__name__)

# ---------------------------------------------------------------
# Load the most recently serialized model at startup
# ---------------------------------------------------------------
MODEL_CANDIDATES = sorted(glob.glob("model-*.pkl")) or sorted(glob.glob("../model-*.pkl"))
if not MODEL_CANDIDATES:
    raise FileNotFoundError(
        "No model-*.pkl found. Run 02_modeling.py first, or copy the .pkl file next to app.py."
    )
MODEL_PATH = MODEL_CANDIDATES[-1]
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
NUM_FEATURES = bundle["num_features"]
CAT_FEATURES = bundle["cat_features"]
logger.info(f"Loaded model: {MODEL_PATH}")

STATE_HOLIDAY_MAP = {"0": 0, "a": 1, "b": 2, "c": 3}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the feature engineering used at training time (see 02_modeling.py)."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["StateHoliday"] = df.get("StateHoliday", "0").astype(str)
    df["StateHolidayCode"] = df["StateHoliday"].map(STATE_HOLIDAY_MAP).fillna(0).astype(int)

    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["IsMonthStart"] = (df["Day"] <= 10).astype(int)
    df["IsMonthMid"] = ((df["Day"] > 10) & (df["Day"] <= 20)).astype(int)
    df["IsMonthEnd"] = (df["Day"] > 20).astype(int)

    df["DaysToHoliday"] = df.get("DaysToHoliday", 0)
    df["DaysAfterHoliday"] = df.get("DaysAfterHoliday", 0)

    def col_or_default(name, default):
        if name in df.columns:
            return df[name].fillna(default) if hasattr(df[name], "fillna") else df[name]
        return pd.Series(default, index=df.index)

    df["CompetitionDistanceLog"] = np.log1p(col_or_default("CompetitionDistance", 1000))
    df["CompetitionOpenMonths"] = col_or_default("CompetitionOpenMonths", 12)
    df["HasCompetition"] = col_or_default("HasCompetition", 1)

    df["Promo2"] = col_or_default("Promo2", 0)
    df["IsPromo2Active"] = col_or_default("IsPromo2Active", 0)
    df["IsPromo2Month"] = col_or_default("IsPromo2Month", 0)

    df["StoreType"] = col_or_default("StoreType", "a")
    df["Assortment"] = col_or_default("Assortment", "a")
    df["SchoolHoliday"] = col_or_default("SchoolHoliday", 0)
    df["Promo"] = col_or_default("Promo", 0)
    return df


def predict(df: pd.DataFrame) -> pd.DataFrame:
    fe = engineer_features(df)
    X = fe[NUM_FEATURES + CAT_FEATURES]
    preds = pipeline.predict(X)
    fe["PredictedSales"] = preds.round(2)
    # rough customer estimate from the historical sales/customer ratio (~8.8)
    fe["PredictedCustomers"] = (fe["PredictedSales"] / 8.8).round(0)
    return fe


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict_form", methods=["POST"])
def predict_form():
    """Single-store, date-range prediction from the HTML form."""
    store_id = int(request.form["store_id"])
    start_date = request.form["start_date"]
    end_date = request.form["end_date"]
    promo = int(request.form.get("promo", 0))
    school_holiday = int(request.form.get("school_holiday", 0))

    dates = pd.date_range(start_date, end_date, freq="D")
    df = pd.DataFrame({
        "Store": store_id,
        "Date": dates,
        "Promo": promo,
        "SchoolHoliday": school_holiday,
        "StateHoliday": "0",
    })
    result = predict(df)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result["Date"], result["PredictedSales"], marker="o", color="#2b8cbe")
    ax.set_title(f"Predicted sales — Store {store_id}")
    ax.set_ylabel("Sales")
    fig.autofmt_xdate()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=110)
    plt.close()
    chart_b64 = base64.b64encode(buf.getvalue()).decode()

    table = result[["Date", "PredictedSales", "PredictedCustomers"]].to_dict(orient="records")
    csv_data = result[["Date", "PredictedSales", "PredictedCustomers"]].to_csv(index=False)

    return render_template("index.html", chart=chart_b64, table=table, csv_data=csv_data)


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    """Bulk prediction from an uploaded CSV (Date, Store, IsHoliday, IsWeekend, IsPromo, ...)."""
    file = request.files["csv_file"]
    df = pd.read_csv(file)
    rename = {"IsHoliday": "StateHoliday", "IsPromo": "Promo"}
    df = df.rename(columns=rename)
    result = predict(df)
    out = io.StringIO()
    result.to_csv(out, index=False)
    out.seek(0)
    return send_file(
        io.BytesIO(out.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictions.csv",
    )


if __name__ == "__main__":
    app.run(debug=True)
