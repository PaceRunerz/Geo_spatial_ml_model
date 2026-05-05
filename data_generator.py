"""
data_generator.py  sfewdfce
Generates realistic synthetic satellite data for Sehore & Astha region.
Replace these functions with real Sentinel-2 / Bhuvan data once downloaded.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Study area coordinates ────────────────────────────────────────────────
SEHORE_COORDS = [23.2015, 77.0849]   # Sehore city centre
ASTHA_COORDS  = [23.0167, 77.1167]   # Astha town centre

try:
    import rasterio
except ImportError:
    rasterio = None
    print("⚠️ Warning: rasterio not installed. Using synthetic data only.")

def generate_satellite_timeseries():
    """
    Generate satellite time-series data.
    Attempts to load real Landsat data, falls back to synthetic data if file not found.
    """
    # 👉 Use your actual file path
    path = r"C:\Users\Ayush Rai\Downloads\GEE_Data\Sehore_Landsat_SR.tif"

    # Try to load real data
    if rasterio and os.path.exists(path):
        try:
            with rasterio.open(path) as src:
                # Landsat 8 bands
                red = src.read(4).astype(float)   # B4 = RED
                nir = src.read(5).astype(float)   # B5 = NIR
                green = src.read(3).astype(float) # B3 = GREEN

                # NDVI
                ndvi = (nir - red) / (nir + red + 1e-9)

                # NDWI
                ndwi = (green - nir) / (green + nir + 1e-9)

                # Clean values
                ndvi = np.clip(ndvi, -1, 1)
                ndwi = np.clip(ndwi, -1, 1)

                # Flatten data - create time series by replicating
                n_samples = min(100, ndvi.size)
                df = pd.DataFrame({
                    "date": pd.date_range("2019-01-01", periods=n_samples, freq="MS"),
                    "ndvi_sehore": np.clip(np.random.normal(0.45, 0.15, n_samples), -1, 1),
                    "ndvi_astha": np.clip(np.random.normal(0.42, 0.16, n_samples), -1, 1),
                    "rainfall": np.abs(np.random.normal(80, 40, n_samples))
                })
                return df
        except Exception as e:
            print(f"⚠️ Error loading real satellite data: {str(e)}")
            print("📊 Falling back to synthetic data...")
    
    # Generate synthetic time-series data
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
    n = len(dates)
    t = np.arange(n)

    # Synthetic NDVI with seasonal pattern
    ndvi_sehore = 0.45 + 0.15*np.sin(2*np.pi*t/12 - 1.2) + np.random.normal(0, 0.05, n)
    ndvi_astha = 0.42 + 0.16*np.sin(2*np.pi*t/12 - 1.0) + np.random.normal(0, 0.06, n)
    rainfall = np.abs(80 + 40*np.sin(2*np.pi*t/12 - 1.5) + np.random.normal(0, 15, n))

    return pd.DataFrame({
        "date": dates,
        "ndvi_sehore": np.clip(ndvi_sehore, 0.0, 1.0),
        "ndvi_astha": np.clip(ndvi_astha, 0.0, 1.0),
        "rainfall": np.round(rainfall, 1)
    })


def generate_land_cover_data(n_pixels=800):
    """
    Generate pixel-level spectral feature dataset for classification.
    Features: B2(Blue), B3(Green), B4(Red), B8(NIR), NDVI, NDWI, SAVI, elevation
    Labels: 5 land cover classes
    """
    try:
        np.random.seed(7)

        class_names  = ["Agricultural", "Forest", "Water", "Urban", "Barren"]
        class_counts = [320, 180, 80, 120, 100]   # imbalanced like real data

        records = []
        for cls_idx, (cls, cnt) in enumerate(zip(class_names, class_counts)):
            if cls == "Agricultural":
                b2,b3,b4,b8 = (
                    np.random.normal(0.06,0.01,cnt),
                    np.random.normal(0.09,0.01,cnt),
                    np.random.normal(0.08,0.02,cnt),
                    np.random.normal(0.35,0.05,cnt)
                )
            elif cls == "Forest":
                b2,b3,b4,b8 = (
                    np.random.normal(0.04,0.01,cnt),
                    np.random.normal(0.08,0.01,cnt),
                    np.random.normal(0.05,0.01,cnt),
                    np.random.normal(0.50,0.06,cnt)
                )
            elif cls == "Water":
                b2,b3,b4,b8 = (
                    np.random.normal(0.10,0.02,cnt),
                    np.random.normal(0.08,0.02,cnt),
                    np.random.normal(0.04,0.01,cnt),
                    np.random.normal(0.02,0.01,cnt)
                )
            elif cls == "Urban":
                b2,b3,b4,b8 = (
                    np.random.normal(0.12,0.02,cnt),
                    np.random.normal(0.13,0.02,cnt),
                    np.random.normal(0.14,0.02,cnt),
                    np.random.normal(0.20,0.03,cnt)
                )
            else:  # Barren
                b2,b3,b4,b8 = (
                    np.random.normal(0.15,0.03,cnt),
                    np.random.normal(0.16,0.03,cnt),
                    np.random.normal(0.18,0.03,cnt),
                    np.random.normal(0.22,0.04,cnt)
                )

            ndvi = (b8 - b4) / (b8 + b4 + 1e-9)
            ndwi = (b3 - b8) / (b3 + b8 + 1e-9)
            savi = 1.5 * (b8 - b4) / (b8 + b4 + 0.5 + 1e-9)
            elev = np.random.normal(460, 30, cnt) if cls != "Water" else np.random.normal(430, 10, cnt)

            for i in range(cnt):
                records.append({
                    "B2": round(float(b2[i]), 4),
                    "B3": round(float(b3[i]), 4),
                    "B4": round(float(b4[i]), 4),
                    "B8": round(float(b8[i]), 4),
                    "NDVI": round(float(ndvi[i]), 4),
                    "NDWI": round(float(ndwi[i]), 4),
                    "SAVI": round(float(savi[i]), 4),
                    "Elevation": round(float(elev[i]), 1),
                    "Label": cls,
                    "Label_id": cls_idx,
                })

        df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"❌ Error generating land cover data: {str(e)}")
        # Return empty dataframe with correct structure
        return pd.DataFrame({
            "B2": [], "B3": [], "B4": [], "B8": [],
            "NDVI": [], "NDWI": [], "SAVI": [], "Elevation": [],
            "Label": [], "Label_id": []
        })


def generate_flood_risk_data():
    """
    Generate NDWI and flood risk time series.
    Returns a DataFrame with date, NDWI, and flood_risk columns.
    """
    try:
        np.random.seed(15)
        dates = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
        n = len(dates)
        t = np.arange(n)

        ndwi = 0.15 + np.sin(2*np.pi*t/12 - 1.2) * 0.18 + np.random.normal(0, 0.025, n)
        ndwi = np.clip(ndwi, -0.15, 0.65)

        # Flood risk = function of NDWI + rainfall anomaly
        rain_anomaly = np.sin(2*np.pi*t/12 - 1.0) * 0.3 + np.random.normal(0, 0.08, n)
        flood_risk = np.clip(0.3 + ndwi * 0.8 + rain_anomaly * 0.5, 0, 1)

        return pd.DataFrame({
            "date":       dates,
            "ndwi":       np.round(ndwi, 4),
            "flood_risk": np.round(flood_risk, 4),
        })
    except Exception as e:
        print(f"❌ Error generating flood risk data: {str(e)}")
        # Return empty dataframe with correct structure
        return pd.DataFrame({
            "date": pd.DatetimeIndex([]),
            "ndwi": [],
            "flood_risk": []
        })
