import pandas as pd
import numpy as np
import joblib
import os
from src.config import PROCESSED_DATA_DIR, MODEL_DIR

def generate_static_forecasts():
    """
    Loads the trained model and features, runs predictions for the test period
    and latest inventory date, and saves the results as tight .parquet files
    so Streamlit doesn't need to load the LightGBM model itself.
    """
    print("Loading data and model...")
    # 1. Load Data and Model
    data_path = os.path.join(PROCESSED_DATA_DIR, "features_v3.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run preprocessing first.")
        
    df = pd.read_parquet(data_path)
    df['product_category'] = df['product_category'].astype('category')
    
    model_path = os.path.join(MODEL_DIR, "lgbm_demand_v3.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Run training first.")
        
    model = joblib.load(model_path)
    
    TARGET = 'quantity_sold'
    ignore_cols = ['order_purchase_timestamp', 'year', TARGET]
    features = [c for c in df.columns if c not in ignore_cols]
    
    # ---------------------------------------------------------
    # PART 1: Demand Forecasting (Test Period)
    # ---------------------------------------------------------
    print("Generating Demand Forecasting data...")
    test_mask = (df['year'] == 2018) & (df['month'] >= 6)
    X_test = df.loc[test_mask, features]
    
    preds_test = model.predict(X_test)
    
    df_forecast = pd.DataFrame({
        'date': df.loc[test_mask, 'order_purchase_timestamp'],
        'category': df.loc[test_mask, 'product_category'],
        'actual': df.loc[test_mask, TARGET],
        'predicted': preds_test
    })
    
    forecast_out_path = os.path.join(PROCESSED_DATA_DIR, "dashboard_forecast_v3.parquet")
    df_forecast.to_parquet(forecast_out_path, index=False)
    print(f"Saved Forecasting Data to {forecast_out_path}")
    
    # ---------------------------------------------------------
    # PART 2: Inventory Replenishment (Latest Date)
    # ---------------------------------------------------------
    print("Generating Inventory Replenishment data...")
    latest_date = df['order_purchase_timestamp'].max()
    current_data = df[df['order_purchase_timestamp'] == latest_date].copy()
    
    preds_latest = model.predict(current_data[features])
    
    df_inventory = pd.DataFrame({
        'Category': current_data['product_category'],
        'Expected_Weekly_Demand': np.ceil(preds_latest),
        # Using sales_roll_std_4 for volatility. If NaN (no history), default to the predicted demand
        'Volatility_StdDev': current_data['sales_roll_std_4'].fillna(pd.Series(preds_latest, index=current_data.index))
    })
    
    # We add the latest_date so Streamlit can display it
    df_inventory['Planning_Date'] = latest_date
    
    inventory_out_path = os.path.join(PROCESSED_DATA_DIR, "dashboard_inventory_v3.parquet")
    df_inventory.to_parquet(inventory_out_path, index=False)
    print(f"Saved Inventory Data to {inventory_out_path}")
    
    print("Optimization Complete! The Streamlit app can now run offline without LightGBM.")

if __name__ == "__main__":
    generate_static_forecasts()
