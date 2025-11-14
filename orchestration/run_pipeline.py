#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
Runs the complete Member 3 data pipeline end-to-end
"""
import yaml
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.feature_store import FeatureStore
from src.alert_system import AlertSystem

def run_complete_pipeline(config_path='config/config.yaml'):
    """
    Execute the complete data pipeline:
    1. Data Cleaning & Alignment
    2. Feature Engineering
    3. Feature Store Management
    4. Surge Alert Detection & Notification
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(
        log_file=config['monitoring']['log_file'],
        log_level=config['monitoring']['log_level']
    )
    
    logger.info("")
    logger.info("="*80)
    logger.info("      EQUI-RIDE SURGE AI   ")
    logger.info("="*80)
    logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info("")
    
    try:
        # ========== STAGE 1: DATA CLEANING ==========
        logger.info("")
        logger.info("*" * 80)
        logger.info("STAGE 1: DATA CLEANING & ALIGNMENT")
        logger.info("*" * 80)
        
        cleaner = DataCleaner(config, logger)
        input_file = config['data']['input_csv']
        processed_file = os.path.join(config['data']['processed_data_path'], 'processed_data.csv')
        
        cleaned_df = cleaner.run_cleaning_pipeline(input_file, processed_file)
        logger.info(f"SUCCESS Stage 1 Complete: {len(cleaned_df)} records cleaned")
        
        # ========== STAGE 2: FEATURE ENGINEERING ==========
        logger.info("")
        logger.info("*" * 80)
        logger.info("STAGE 2: FEATURE ENGINEERING")
        logger.info("*" * 80)
        
        engineer = FeatureEngineer(config, logger)
        features_file = os.path.join(config['data']['features_path'], 'engineered_features.csv')
        
        features_df = engineer.run_feature_engineering_pipeline(processed_file, features_file)
        logger.info(f"SUCCESS Stage 2 Complete: {len(features_df.columns)} features generated")
        
        # ========== STAGE 3: FEATURE STORE ==========
        logger.info("")
        logger.info("*" * 80)
        logger.info("STAGE 3: FEATURE STORE MANAGEMENT")
        logger.info("*" * 80)
        
        store = FeatureStore(config, logger)
        store.connect()
        store.create_tables()
        
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        records_inserted = store.insert_features(features_df, batch_id=batch_id)
        logger.info(f"SUCCESS Stage 3 Complete: {records_inserted} records inserted into feature store")
        
        # ========== STAGE 4: SURGE ALERT DETECTION ==========
        logger.info("")
        logger.info("*" * 80)
        logger.info("STAGE 4: SURGE ALERT DETECTION & NOTIFICATION")
        logger.info("*" * 80)
        
        alert_system = AlertSystem(config, logger)
        
        # Check for surge conditions
        alerts = alert_system.check_surge_conditions(features_df)
        logger.info(f"Surge conditions detected: {len(alerts)}")
        
        # Process and send alerts
        if len(alerts) > 0:
            sent_count = alert_system.process_alerts(alerts, feature_store=store)
            logger.info(f"SUCCESS Stage 4 Complete: {sent_count} alerts processed")
        else:
            logger.info("SUCCESS Stage 4 Complete: No alerts triggered")
        
        # Close feature store connection
        store.close()
        
        # ========== PIPELINE SUCCESS ==========
        logger.info("")
        logger.info("="*80)
        logger.info(" PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        logger.info("SUMMARY:")
        logger.info(f"   Records processed: {len(features_df)}")
        logger.info(f"   Features generated: {len(features_df.columns)}")
        logger.info(f"   Batch ID: {batch_id}")
        logger.info(f"   Alerts detected: {len(alerts)}")
        logger.info("")
        logger.info("OUTPUT FILES:")
        logger.info(f"   Processed data: {processed_file}")
        logger.info(f"   Engineered features: {features_file}")
        logger.info(f"   Feature store: {config['data']['feature_store_path']}")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error("")
        logger.error("="*80)
        logger.error("PIPELINE FAILED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        logger.error("="*80)
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point"""
    success = run_complete_pipeline()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
