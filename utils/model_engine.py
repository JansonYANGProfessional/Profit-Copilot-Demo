import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

class ModelEngine:
    def __init__(self, db_path='data/profit.db'):
        self.db_path = db_path
        self.model_path = 'data/profit_model.h5'
        self.scaler_path = 'data/scaler.pkl'
        self.window_size = 30
        
        # Feature columns to use for prediction
        self.feature_cols = [
            'device_vib_mean', 'device_vib_std', 
            'device_temp_mean', 'device_temp_std',
            'daily_failure_rate', 
            'electricity_price', 'material_price_index'
        ]
        # Target columns to predict
        self.target_cols = ['revenue', 'cogs', 'expenses']

    def load_and_preprocess(self):
        """Load data from DB and prepare X, Y sequences."""
        print("Loading data from SQLite...")
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM unified_daily_data ORDER BY date ASC", conn)
        conn.close()
        
        # Feature Engineering targets
        # COGS = Material + Labor
        df['cogs'] = df['cost_material'] + df['cost_labor']
        # Expenses = Energy (for this MVP)
        df['expenses'] = df['cost_energy']
        
        # Handle missing values just in case
        df = df.fillna(0)
        
        # Select Features and Targets
        data_df = df[self.feature_cols + self.target_cols]
        
        # Scaling
        print("Scaling data...")
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data_df)
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        # Create Sequences
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i, :len(self.feature_cols)]) # Input: Past Features
            y.append(scaled_data[i, len(self.feature_cols):]) # Output: Current Targets
            
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build CNN-LSTM Model."""
        print(f"Building model with input shape: {input_shape}")
        model = Sequential([
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            LSTM(50, activation='relu'),
            Dense(3) # Revenue, COGS, Expenses
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self):
        """Train the model."""
        X, y = self.load_and_preprocess()
        
        if len(X) == 0:
            print("Not enough data to train.")
            return
            
        model = self.build_model((X.shape[1], X.shape[2]))
        
        print("Starting training...")
        history = model.fit(X, y, epochs=20, batch_size=32, verbose=1)
        
        model.save(self.model_path)
        final_loss = history.history['loss'][-1]
        print(f"Training complete. Final Loss: {final_loss:.6f}")
        return f"Training success. Loss: {final_loss:.6f}"

    def predict_future(self, days=30):
        """Generate future predictions."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Model or Scaler not found. Train first.")
            
        model = load_model(self.model_path, compile=False)
        model.compile(optimizer='adam', loss='mse')
        scaler = joblib.load(self.scaler_path)
        
        # Load last window of data
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(f"SELECT * FROM unified_daily_data ORDER BY date DESC LIMIT {self.window_size}", conn)
        conn.close()
        
        # Re-calculate derived columns for the input window
        df['cogs'] = df['cost_material'] + df['cost_labor']
        df['expenses'] = df['cost_energy']
        df = df.sort_values('date', ascending=True) # Ensure chronological order
        
        # Scale input
        input_data = df[self.feature_cols + self.target_cols].values
        scaled_input = scaler.transform(input_data)
        
        # Initial sequence (features only)
        # Shape: (1, 30, features)
        # Note: The model expects (batch, steps, features). 
        # Our X during training was (features). 
        # Wait, X during training was `scaled_data[..., :len(feature_cols)]`.
        # So we need to extract features from scaled_input.
        
        current_seq = scaled_input[:, :len(self.feature_cols)]
        current_seq = current_seq.reshape((1, self.window_size, len(self.feature_cols)))
        
        predictions = []
        
        # Recursive Prediction
        for _ in range(days):
            # Predict next day targets (scaled)
            pred_targets = model.predict(current_seq, verbose=0) # Shape (1, 3)
            
            # We need to append this prediction to the sequence to predict the next day.
            # But the sequence consists of FEATURES, not TARGETS.
            # Strategy: We need "Future Features". 
            # For this MVP, we will assume features stay constant (naive forecast) or use the last known values.
            # Let's use the last known feature values from the sequence.
            
            last_features = current_seq[0, -1, :] # Shape (features,)
            
            # Construct next step input: (features)
            # In a real scenario, we might forecast features too, or they might be known (like planned production).
            # Here we just repeat the last features.
            next_step_features = last_features.reshape(1, 1, -1)
            
            # Append to sequence (slide window)
            # Remove first, add new at end
            current_seq = np.concatenate([current_seq[:, 1:, :], next_step_features], axis=1)
            
            predictions.append(pred_targets[0])
            
        # Inverse Transform
        # We have predictions (days, 3). We need to inverse transform them.
        # The scaler expects (n_samples, features + targets).
        # We need to create a dummy matrix with placeholders for features to use inverse_transform.
        
        pred_array = np.array(predictions)
        dummy_features = np.zeros((days, len(self.feature_cols)))
        to_inverse = np.hstack([dummy_features, pred_array])
        
        inversed = scaler.inverse_transform(to_inverse)
        
        # Extract targets (last 3 columns)
        final_preds = inversed[:, len(self.feature_cols):]
        
        # Create DataFrame
        last_date = pd.to_datetime(df['date'].iloc[-1])
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        
        pred_df = pd.DataFrame(final_preds, columns=['Predicted_Revenue', 'Predicted_Cost', 'Predicted_Expenses'])
        pred_df['Date'] = future_dates
        
        return pred_df[['Date', 'Predicted_Revenue', 'Predicted_Cost', 'Predicted_Expenses']]

if __name__ == "__main__":
    engine = ModelEngine()
    result = engine.train()
    print(result)
