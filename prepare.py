
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Load your data into a DataFrame assuming it's stored in a CSV file
# Replace 'your_data.csv' with the actual path to your data file
data = pd.read_csv('data/data.csv', delimiter=',', header=None)

# Rename columns for clarity
data.columns = [
    'timestamp', 'best_ask', 'best_ask_quantity', 'best_bid', 'best_bid_quantity',
    'last_price', 'open_price', 'high_price', 'low_price', 'base_asset_volume',
    'quote_asset_volume', 'price_change', 'price_change_percent', 'last_trade_identifier'
]

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

# You may want to sort your data by timestamp if it's not already sorted
data = data.sort_values(by='timestamp')

# Normalize numerical features (e.g., OHLC prices, volumes)
numerical_cols = [
    'best_ask', 'best_ask_quantity', 'best_bid', 'best_bid_quantity',
    'last_price', 'open_price', 'high_price', 'low_price',
    'base_asset_volume', 'quote_asset_volume', 'price_change', 'price_change_percent'
]

scaler = MinMaxScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Optionally, you can drop or handle missing data if needed
data = data.dropna()

# Create 'label' column based on conditions
data['label'] = 0  # Initialize all labels to 0
for i in range(1, len(data)):
    price_change_percent = data.loc[i, 'price_change_percent']
    if price_change_percent > 0.7:
        data.loc[i, 'label'] = 1
    elif price_change_percent < -0.7:
        data.loc[i, 'label'] = -1

# Optionally, you can split your data into features (X) and labels (y)
X = data[numerical_cols].values
y = data['label'].values


X_file_path = './X_data.npy'
y_file_path = './y_labels.npy'

# Save X and y as binary files
np.save(X_file_path, X)
np.save(y_file_path, y)

print(f'X data saved to {X_file_path}')
print(f'y labels saved to {y_file_path}')