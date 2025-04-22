import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    return data


def preprocess_data(data):
    selected_columns = [
        "current_token_balance",
        "period_total_tx_count",
        "period_incoming_tx_count",
        "period_outgoing_tx_count",
        "period_total_volume_in",
        "period_total_volume_out",
        "period_avg_volume_in",
        "period_avg_volume_out",
        "period_unique_counterparties",
        "period_active_days"
    ]

    X = data[selected_columns + ['period_first_tx_date', 'period_last_tx_date', 'address']].copy()
    X_log = np.log1p(X[selected_columns])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_columns)

    X_scaled_df['period_first_tx_date'] = pd.to_datetime(data['period_first_tx_date'])
    X_scaled_df['period_last_tx_date'] = pd.to_datetime(data['period_last_tx_date'])
    X_scaled_df['date_difference_days'] = (
                X_scaled_df['period_last_tx_date'] - X_scaled_df['period_first_tx_date']).dt.days
    X_scaled_df = X_scaled_df.drop(columns=['period_first_tx_date', 'period_last_tx_date'])

    features = X_scaled_df.drop(columns=['address', 'date_difference_days'], errors='ignore')
    scaled_features = StandardScaler().fit_transform(features)

    return scaled_features, X_scaled_df