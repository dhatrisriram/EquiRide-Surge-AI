from dotenv import load_dotenv
import os
import requests
import pandas as pd
import time

load_dotenv()
HERE_API_KEY = os.getenv('HEREMAPS_API_KEY')

INPUT_FILE = "All-time Table-Bangalore-Wards.csv"
OUTPUT_FILE = "data/processed/Updated_Bangalore_Wards.csv"
BASE_URL = "https://geocode.search.hereapi.com/v1/geocode"

ride_data = pd.read_csv(INPUT_FILE)
ride_data['latitude'] = ride_data.get('latitude', None)
ride_data['longitude'] = ride_data.get('longitude', None)

def get_coordinates(ward_name):
    params = {
        'q': f'{ward_name}, Bangalore, India',
        'apiKey': HERE_API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['items']:
            position = data['items'][0]['position']
            return position['lat'], position['lng']
    return None, None

for idx, row in ride_data.iterrows():
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        lat, lng = get_coordinates(row['Ward'])
        ride_data.at[idx, 'latitude'] = lat
        ride_data.at[idx, 'longitude'] = lng
        print(f"Ward: {row['Ward']}, Lat: {lat}, Lng: {lng}")
        time.sleep(1)

ride_data.to_csv(OUTPUT_FILE, index=False)
print(f"Updated dataset saved to {OUTPUT_FILE}")
