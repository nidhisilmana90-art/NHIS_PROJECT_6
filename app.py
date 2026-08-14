"""Streamlit deployment for the Rossmann sales forecast model."""

import glob
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATE_HOLIDAY_MAP = {"0": 0, "a": 1, "b": 2, "c": 3}


@st.cache_resource
def load_model_bundle():
    """Load the newest serialized model artifact."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = sorted(glob.glob(os.path.join(base_dir, "model-*.pkl")))
    if not candidates:
        raise FileNotFoundError(
            "No model-*.pkl found in the project directory. Place the trained model next to app.py."
        )
    model_path = candidates[-1]
    bundle = joblib.load(model_path)
    return model_path, bundle["pipeline"], bundle["num_features"], bundle["cat_features"]


MODEL_PATH, pipeline, NUM_FEATURES, CAT_FEATURES = load_model_bundle()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the feature engineering used during training."""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        raise ValueError("The input dataframe must contain a 'Date' column.")

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

    df["DaysToHoliday"] = pd.to_numeric(df.get("DaysToHoliday", 0), errors="coerce").fillna(0)
    df["DaysAfterHoliday"] = pd.to_numeric(df.get("DaysAfterHoliday", 0), errors="coerce").fillna(0)

    def col_or_default(name, default):
        if name in df.columns:
            return df[name].fillna(default) if hasattr(df[name], "fillna") else df[name]
        return pd.Series(default, index=df.index)

    df["CompetitionDistanceLog"] = np.log1p(col_or_default("CompetitionDistance", 1000))
    df["CompetitionOpenMonths"] = pd.to_numeric(col_or_default("CompetitionOpenMonths", 12), errors="coerce").fillna(12)
    df["HasCompetition"] = pd.to_numeric(col_or_default("HasCompetition", 1), errors="coerce").fillna(1)

    df["Promo2"] = pd.to_numeric(col_or_default("Promo2", 0), errors="coerce").fillna(0)
    df["IsPromo2Active"] = pd.to_numeric(col_or_default("IsPromo2Active", 0), errors="coerce").fillna(0)
    df["IsPromo2Month"] = pd.to_numeric(col_or_default("IsPromo2Month", 0), errors="coerce").fillna(0)

    df["StoreType"] = col_or_default("StoreType", "a")
    df["Assortment"] = col_or_default("Assortment", "a")
    df["SchoolHoliday"] = pd.to_numeric(col_or_default("SchoolHoliday", 0), errors="coerce").fillna(0)
    df["Promo"] = pd.to_numeric(col_or_default("Promo", 0), errors="coerce").fillna(0)
    return df


def predict(df: pd.DataFrame) -> pd.DataFrame:
    fe = engineer_features(df)
    X = fe[NUM_FEATURES + CAT_FEATURES]
    preds = pipeline.predict(X)
    fe["PredictedSales"] = preds.round(2)
    fe["PredictedCustomers"] = (fe["PredictedSales"] / 8.8).round(0)
    return fe


st.set_page_config(page_title="Rossmann Sales Forecast", page_icon="📈", layout="wide")
st.title("Rossmann Sales Forecast")
st.caption(f"Model loaded from: {os.path.basename(MODEL_PATH)}")

with st.sidebar:
    st.header("Forecast settings")
    store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
    start_date = st.date_input("Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=14))
    end_date = st.date_input("End date", value=pd.Timestamp.today().date())
    promo = st.checkbox("Promo active", value=True)
    school_holiday = st.checkbox("School holiday", value=False)
    generate_button = st.button("Generate forecast")

uploaded_file = st.file_uploader("Upload a CSV for bulk predictions", type=["csv"], help="CSV should include Date and Store columns, with optional Promo and SchoolHoliday columns.")

if generate_button:
    if end_date < start_date:
        st.warning("The end date must be after the start date.")
    else:
        dates = pd.date_range(start_date, end_date, freq="D")
        df = pd.DataFrame(
            {
                "Store": store_id,
                "Date": dates,
                "Promo": int(promo),
                "SchoolHoliday": int(school_holiday),
                "StateHoliday": "0",
            }
        )
        result = predict(df)

        st.subheader(f"Forecast for Store {store_id}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result["Date"], result["PredictedSales"], marker="o", color="#2b8cbe")
        ax.set_title(f"Predicted Sales — Store {store_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        fig.autofmt_xdate()
        st.pyplot(fig)

        table = result[["Date", "PredictedSales", "PredictedCustomers"]].copy()
        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(table.rename(columns={"PredictedSales": "Sales", "PredictedCustomers": "Customers"}), use_container_width=True)

        csv_data = table.rename(columns={"PredictedSales": "Sales", "PredictedCustomers": "Customers"}).to_csv(index=False)
        st.download_button(
            "Download forecast CSV",
            csv_data,
            file_name=f"sales_forecast_store_{store_id}.csv",
            mime="text/csv",
        )

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        if "Date" not in input_df.columns:
            st.error("The uploaded CSV must contain a 'Date' column.")
        else:
            result = predict(input_df)
            st.subheader("Bulk prediction results")
            display_df = result[["Date", "PredictedSales", "PredictedCustomers"]].copy()
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_df.rename(columns={"PredictedSales": "Sales", "PredictedCustomers": "Customers"}), use_container_width=True)
            st.download_button(
                "Download full predictions",
                display_df.rename(columns={"PredictedSales": "Sales", "PredictedCustomers": "Customers"}).to_csv(index=False),
                file_name="bulk_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")

if not generate_button and uploaded_file is None:
    st.info("Use the sidebar to generate a daily sales forecast for a store, or upload a CSV file with dates to run bulk prediction.")

uploaded_file = st.file_uploader("Upload a CSV for bulk predictions", type=["csv"], help="CSV should include Date and Store columns, with optional Promo and SchoolHoliday columns.")

if generate_button:
    if end_date < start_date:
        st.warning("The end date must be after the start date.")
    else:
        dates = pd.date_range(start_date, end_date, freq="D")
        df = pd.DataFrame(
            {
                "Store": store_id,
                "Date": dates,
                "Promo": int(promo),
                "SchoolHoliday": int(school_holiday),
                "StateHoliday": "0",
            }
        )
        result = predict(df)

        st.subheader(f"Forecast for Store {store_id}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result["Date"], result["PredictedSales"], marker="o", color="#2b8cbe")
        ax.set_title(f"Predicted Sales — Store {store_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        fig.autofmt_xdate()
        st.pyplot(fig)

        table = result[["Date", "PredictedSales", "PredictedCustomers"]].copy()
        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(table.rename(columns={"PredictedSales": "Sales", "PredictedCustomers": "Customers"}), use_container_width=True)

        csv_data = table.rename(columns={"PredictedSales": "Sales", "PredictedCustomers": "Customers"}).to_csv(index=False)
        st.download_button(
            "Download forecast CSV",
            csv_data,
            file_name=f"sales_forecast_store_{store_id}.csv",
            mime="text/csv",
        )

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        if "Date" not in input_df.columns:
            st.error("The uploaded CSV must contain a 'Date' column.")
        else:
            result = predict(input_df)
            st.subheader("Bulk prediction results")
            display_df = result[["Date", "PredictedSales", "PredictedCustomers"]].copy()
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_df.rename(columns={"PredictedSales": "Sales", "PredictedCustomers": "Customers"}), use_container_width=True)
            st.download_button(
                "Download full predictions",
                display_df.rename(columns={"PredictedSales": "Sales", "PredictedCustomers": "Customers"}).to_csv(index=False),
                file_name="bulk_predictions.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")

if not generate_button and uploaded_file is None:
    st.info("Use the sidebar to generate a daily sales forecast for a store, or upload a CSV file with dates to run bulk prediction.")
>>>>>>> 05db47d (Initial Streamlit deploy commit)
