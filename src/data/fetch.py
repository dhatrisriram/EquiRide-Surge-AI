import requests
import pandas as pd
import logging
from src.utils import load_csv_data # Assuming load_config is in utils

logger = logging.getLogger(__name__)

# Define the location of the main dataset
MAIN_DATA_FILE = 'EquiRide-Surge-AI\datasets\Data_set.csv' 

def fetch_raw_data():
    """
    Fetches the raw, correct dataset for the project.
    Simulates a streaming fetch by reading the full CSV file.
    
    Returns:
        pd.DataFrame: The raw data loaded from Data_set.csv.
    """
    try:
        # Load the correct dataset from the project's root directory
        raw_df = pd.read_csv(MAIN_DATA_FILE)
        logger.info(f"Successfully loaded raw data from {MAIN_DATA_FILE}. Shape: {raw_df.shape}")
        return raw_df
    except FileNotFoundError:
        logger.error(f"FATAL: The main data file {MAIN_DATA_FILE} was not found.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading {MAIN_DATA_FILE}: {e}")
        return pd.DataFrame()

def fetch_traffic_data(api_key, loc, time):
    # Connect to HERE API or similar, returns dict with metrics
    # Placeholder example
    response = requests.get(f"http://traffic_api?key={api_key}&location={loc}&time={time}")
    return response.json()

def fetch_event_data(api_key, location, window_start, window_end):
    url = f"https://api.predicthq.com/v1/events/?location_around={location}&start={window_start}&end={window_end}"
    headers = {'Authorization': f'Bearer {api_key}'}
    return requests.get(url, headers=headers).json()

def fetch_weather_data(api_key, coords):
    # Example for OpenWeather API
    url = f"<http://api.openweathermap.org/data/2.5/weather?lat={coords}&lon={coords>[1]}&appid={api_key}"
    return requests.get(url).json()
