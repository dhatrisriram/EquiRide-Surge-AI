import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the raw data for feature engineering.
    
    1. Combines 'Date' and 'Time' columns into a single 'Datetime' timestamp.
    2. Renames the zone identifier column to a standard 'zone'.
    3. Handles initial missing values and standardizes data types.
    
    Args:
        raw_df (pd.DataFrame): The raw data loaded from Data_set.csv.
        
    Returns:
        pd.DataFrame: The cleaned and standardized DataFrame.
    """
    if raw_df.empty:
        logger.warning("Raw DataFrame is empty, skipping preprocessing.")
        return pd.DataFrame()

    logger.info("Starting data preprocessing...")

    df = raw_df.copy()

    # --- 1. Datetime Standardization ---
    if 'Date' in df.columns and 'Time' in df.columns:
        # Combine 'Date' and 'Time' into a single Datetime column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df = df.dropna(subset=['Datetime'])
        df = df.drop(columns=['Date', 'Time'])
        
        # Set Datetime as index (required for time series models/aggregation)
        df = df.set_index('Datetime').sort_index()
    else:
        logger.error("Missing 'Date' or 'Time' columns needed for time series index.")
        return pd.DataFrame()

    # --- 2. Rename Spatial Identifier ---
    if 'h3_index' in df.columns:
        df = df.rename(columns={'h3_index': 'zone'})
    else:
        logger.error("Missing required spatial identifier 'h3_index'.")
        return pd.DataFrame()
        
    # --- 3. Initial Cleaning and Type Conversion ---
    # Ensure all columns used for modeling are numerical.
    # We focus on key columns relevant for surge/demand prediction and optimization
    
    # Target variable and key features
    demand_cols = ['Completed Trips', 'Bookings', 'Traffic Volume', 
                   "Drivers' Earnings", 'Average Distance per Trip (km)', 
                   'Distance Travelled (km)']
                   
    for col in demand_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors (e.g., non-numeric strings become NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Simple imputation for robustness (Member 3 should refine this)
    df = df.fillna(0) 

    # --- 4. Select Final Columns ---
    # Select a standard set of columns for the next pipeline steps
    final_cols = ['zone', 'Completed Trips', 'Bookings', 'Traffic Volume', 
                  'Congestion Level', "Drivers' Earnings", 'Distance Travelled (km)', 
                  'event type', 'event importance']
    
    # Filter to only keep columns present in the DataFrame
    valid_cols = [col for col in final_cols if col in df.columns]
    
    # Reset index to make 'Datetime' a regular column for downstream aggregation
    df_final = df[valid_cols].reset_index()

    logger.info(f"Preprocessing finished. Final columns: {df_final.columns.tolist()}")
    return df_final

def preprocess_events(events_json):
    df = pd.DataFrame(events_json['results'])
    df['bucket'] = pd.to_datetime(df['start']).dt.floor('15T')
    # Feature: event intensity (scale by expected attendees, type, proximity)
    df['weight'] = df['attendance'].fillna(1) * (df['type'].map({'concert':1.2, 'sports':1.1, 'other':1}))
    return df

def merge_traffic_weather_events(traffic, weather, events):
    # Merge/join all objects by location and time bucket, filling missing with zeros
    traffic_df = pd.DataFrame(traffic)
    weather_df = pd.DataFrame(weather)
    events_df = preprocess_events(events)
    merged = pd.merge(traffic_df, weather_df, on=['location','bucket'], how='outer')
    merged = pd.merge(merged, events_df, on=['location','bucket'], how='outer')
    merged = merged.fillna(0)
    return merged
