################################################################################
# FILE 10: test_setup.py
# VERIFY YOUR SETUP
################################################################################

#!/usr/bin/env python3
"""
Quick Test Script - Verify Pipeline Setup
"""
import os
import sys

print("="*70)
print("MEMBER 3 PIPELINE - QUICK VERIFICATION TEST")
print("="*70)

# Test 1: Check Python version
print("\n[1/7] Checking Python version...")
version = sys.version_info
if version.major >= 3 and version.minor >= 8:
    print(f"    ✓ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"    ✗ Python {version.major}.{version.minor} (3.8+ required)")

# Test 2: Check required files
print("\n[2/7] Checking required files...")
required_files = [
    'config/config.yaml',
    'src/data_cleaning.py',
    'src/feature_engineering.py',
    'src/feature_store.py',
    'src/alert_system.py',
    'src/utils.py',
    'orchestration/run_pipeline.py',
    'cleaned_bangalore_traffic_wards.csv'
]

all_exist = True
for file in required_files:
    if os.path.exists(file):
        print(f"    ✓ {file}")
    else:
        print(f"    ✗ {file} (MISSING)")
        all_exist = False

# Test 3: Check directories
print("\n[3/7] Checking directories...")
required_dirs = ['config', 'src', 'orchestration', 'data', 'logs']
for directory in required_dirs:
    if os.path.isdir(directory):
        print(f"    ✓ {directory}/")
    else:
        print(f"    ⚠ {directory}/ (creating...)")
        os.makedirs(directory, exist_ok=True)

# Create subdirectories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/features', exist_ok=True)
os.makedirs('data/feature_store', exist_ok=True)

# Test 4: Import core modules
print("\n[4/7] Testing core imports...")
try:
    import pandas as pd
    print("    ✓ pandas")
except:
    print("    ✗ pandas (run: pip install pandas)")

try:
    import numpy as np
    print("    ✓ numpy")
except:
    print("    ✗ numpy (run: pip install numpy)")

try:
    import yaml
    print("    ✓ pyyaml")
except:
    print("    ✗ pyyaml (run: pip install pyyaml)")

try:
    import h3
    print("    ✓ h3")
except:
    print("    ✗ h3 (run: pip install h3)")

# Test 5: Test config loading
print("\n[5/7] Testing configuration...")
try:
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("    ✓ Configuration loaded successfully")
    print(f"    • Time buckets: {config['features']['time_buckets_minutes']} minutes")
    print(f"    • Anomaly threshold: {config['features']['anomaly_threshold']}")
except Exception as e:
    print(f"    ✗ Configuration error: {str(e)}")

# Test 6: Check input data
print("\n[6/7] Checking input data...")
try:
    import pandas as pd
    df = pd.read_csv('cleaned_bangalore_traffic_wards.csv', nrows=5)
    print(f"    ✓ Input CSV loaded")
    print(f"    • Columns: {len(df.columns)}")
    print(f"    • Sample rows: {len(df)}")
except Exception as e:
    print(f"    ✗ Input data error: {str(e)}")

# Test 7: Check Twilio setup
print("\n[7/7] Checking Twilio configuration...")
from dotenv import load_dotenv
load_dotenv()

twilio_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_token = os.getenv('TWILIO_AUTH_TOKEN')

if twilio_sid and twilio_token:
    print("    ✓ Twilio credentials found in .env")
else:
    print("    ⚠ Twilio not configured (optional)")
    print("    • Copy .env.example to .env and add credentials")
    print("    • Or disable in config.yaml: twilio.enabled: false")

# Final summary
print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

if all_exist:
    print("\n✅ All checks passed! Ready to run pipeline:")
    print("\n   python orchestration/run_pipeline.py")
else:
    print("\n⚠ Some issues found. Please:")
    print("   1. Install missing packages: pip install -r requirements.txt")
    print("   2. Ensure all files are present")
    print("   3. Run test again: python test_setup.py")

print("="*70)