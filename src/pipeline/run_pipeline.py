from prefect import flow, task

@task
def load_data():
    from src.data.preprocess import load_and_validate
    df = load_and_validate('cleaned_bangalore_traffic_wards.csv')
    return df

@task
def preprocess_data(df):
    from src.data.preprocess import add_spatial_features, add_temporal_features
    df = add_spatial_features(df)
    df = add_temporal_features(df)
    return df

@task
def save_features(df):
    from src.data.feature_store import save_feature_store
    save_feature_store(df)

@flow
def run_full_pipeline():
    df = load_data()
    df_processed = preprocess_data(df)
    save_features(df_processed)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    run_full_pipeline()
