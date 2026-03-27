"""
ml_models.py
SVM, KNN classification + LSTM time-series prediction
for geospatial land cover analysis — Sehore & Astha.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from data_generator import generate_land_cover_data


FEATURE_COLS = ["B2", "B3", "B4", "B8", "NDVI", "NDWI", "SAVI", "Elevation"]
CLASS_NAMES  = ["Agricultural", "Forest", "Water", "Urban", "Barren"]


def _prepare_data():
    """Prepare data for ML models with error handling"""
    try:
        df = generate_land_cover_data()
        
        # Validate dataframe
        if df is None or df.empty:
            raise ValueError("Land cover data generation returned empty dataframe")
        
        # Check for required columns
        missing_cols = [col for col in FEATURE_COLS + ["Label"] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        X = df[FEATURE_COLS].values
        
        # Handle potential NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        le = LabelEncoder()
        y = le.fit_transform(df["Label"].values)
        sc = StandardScaler()
        X_s = sc.fit_transform(X)
        
        return train_test_split(X_s, y, test_size=0.25, random_state=42, stratify=y), le
    except Exception as e:
        print(f"❌ Error preparing data: {str(e)}")
        raise


def run_svm_classification():
    """Run SVM with error handling"""
    try:
        (X_tr, X_te, y_tr, y_te), le = _prepare_data()
        model = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        return {
            "accuracy":         float(accuracy_score(y_te, y_pred)),
            "precision":        float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
            "recall":           float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
            "f1":               float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
            "classes":          CLASS_NAMES,
        }
    except Exception as e:
        print(f"❌ Error in SVM classification: {str(e)}")
        raise


def run_knn_classification(k: int = 5):
    """Run KNN with error handling"""
    try:
        (X_tr, X_te, y_tr, y_te), le = _prepare_data()
        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        return {
            "accuracy":         float(accuracy_score(y_te, y_pred)),
            "precision":        float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
            "recall":           float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
            "f1":               float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
            "classes":          CLASS_NAMES,
        }
    except Exception as e:
        print(f"❌ Error in KNN classification: {str(e)}")
        raise


def run_lstm_prediction(ts_data: pd.DataFrame, region: str = "sehore"):
    """
    Lightweight LSTM-style prediction using numpy (no TF dependency).
    For a real project, swap the _lstm_numpy_approx for a Keras LSTM.
    """
    try:
        col_name = f"ndvi_{region}"
        
        # Validate input
        if ts_data is None or ts_data.empty:
            raise ValueError("Time-series data is empty")
        
        if col_name not in ts_data.columns:
            raise ValueError(f"Column '{col_name}' not found in data")
        
        series = ts_data[col_name].values.astype(float)
        dates  = pd.to_datetime(ts_data["date"])

        # Handle NaN values
        series = np.nan_to_num(series, nan=np.nanmean(series))

        # ── Train/test split (last 12 months = test) ──────────────────────────
        n_test   = 12
        n_train  = len(series) - n_test
        train    = series[:n_train]
        test     = series[n_train:]

        # ── Approximate LSTM with exponential smoothing + seasonal decompose ──
        predicted = _lstm_numpy_approx(train, n_test)
        future_12 = _lstm_numpy_approx(series, 12)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(test, predicted))
        mae  = mean_absolute_error(test, predicted)
        r2   = r2_score(test, predicted)

        future_dates = pd.date_range(
            start=dates.iloc[-1] + pd.DateOffset(months=1),
            periods=12, freq="MS"
        )

        return {
            "dates_actual": dates[:n_train].tolist(),
            "actual":       train.tolist(),
            "dates_pred":   dates[n_train:].tolist(),
            "predicted":    predicted.tolist(),
            "dates_future": future_dates.tolist(),
            "future":       future_12.tolist(),
            "rmse": float(rmse),
            "mae":  float(mae),
            "r2":   float(r2),
        }
    except Exception as e:
        print(f"❌ Error in LSTM prediction: {str(e)}")
        raise


def _lstm_numpy_approx(series, n_ahead):
    """
    Seasonal naive + exponential trend approximation.
    Mimics what an LSTM would learn from seasonal NDVI data.
    In production: replace with keras.layers.LSTM.
    """
    period = 12
    alpha  = 0.3   # smoothing factor
    n      = len(series)

    # Holt-Winters-like seasonal smoothing
    smoothed = np.zeros(n)
    smoothed[0] = series[0]
    for i in range(1, n):
        smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i-1]

    # Seasonal index
    seasonal = np.zeros(period)
    counts   = np.zeros(period)
    for i in range(n):
        seasonal[i % period] += series[i] - smoothed[i]
        counts[i % period]   += 1
    seasonal /= np.maximum(counts, 1)

    # Forecast
    forecast = np.zeros(n_ahead)
    for i in range(n_ahead):
        base = smoothed[-1] + (smoothed[-1] - smoothed[-2]) * (i+1) * 0.1
        forecast[i] = base + seasonal[(n + i) % period]

    return np.clip(forecast, 0.05, 0.90)


def compute_ndvi_stats(ts_data: pd.DataFrame) -> dict:
    s = ts_data["ndvi_sehore"].values
    a = ts_data["ndvi_astha"].values
    n = len(s)

    # Simple linear trend slope
    x = np.arange(n)
    trend_s = np.polyfit(x, s, 1)[0] * 12   # per year
    trend_a = np.polyfit(x, a, 1)[0] * 12

    return {
        "mean_sehore":  float(np.mean(s)),
        "mean_astha":   float(np.mean(a)),
        "trend_sehore": float(trend_s),
        "trend_astha":  float(trend_a),
        "max_ndvi":     float(max(s.max(), a.max())),
        "min_ndvi":     float(min(s.min(), a.min())),
    }
