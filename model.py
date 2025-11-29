"""
Machine Learning Model Module
Defines, trains, and manages ML models for cryptocurrency price prediction.
Supports XGBoost, LightGBM, and LSTM models.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class CryptoPredictor:
    """
    Class to train and use ML models for cryptocurrency price prediction.
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize the predictor with a model type.
        
        Args:
            model_type (str): Type of model ('xgboost', 'lightgbm', or 'lstm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns = None
        print(f"Initialized {model_type} predictor")
    
    def prepare_data(self, df, target_col='target', test_size=0.2, validation_size=0.1):
        """
        Prepare data for training by splitting into train/validation/test sets.
        
        Args:
            df (pd.DataFrame): DataFrame with features and target
            target_col (str): Name of target column
            test_size (float): Proportion of data for testing
            validation_size (float): Proportion of training data for validation
            test_size (float): Proportion of data for testing
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, feature_columns)
        """
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_col]
        X = df[feature_columns].values
        y = df[target_col].values
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Split train into train and validation
        val_size = int(len(X_train) * validation_size)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_columns
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Train an XGBoost classifier.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
        
        Returns:
            xgb.XGBClassifier: Trained model
        """
        print("Training XGBoost model...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        print("XGBoost training completed")
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """
        Train a LightGBM classifier.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
        
        Returns:
            lgb.LGBMClassifier: Trained model
        """
        print("Training LightGBM model...")
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        print("LightGBM training completed")
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, sequence_length=60):
        """
        Train an LSTM neural network.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
            sequence_length (int): Length of sequences for LSTM
        
        Returns:
            keras.Model: Trained LSTM model
        """
        print("Training LSTM model...")
        
        # Scale features for LSTM
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Reshape data for LSTM (samples, timesteps, features)
        # For simplicity, we'll use a single timestep approach
        # In production, you'd want to create sequences
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("LSTM training completed")
        return model
    
    def train(self, df, target_col='target'):
        """
        Train the model on the provided data.
        
        Args:
            df (pd.DataFrame): DataFrame with features and target
            target_col (str): Name of target column
        """
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_columns = self.prepare_data(
            df, target_col=target_col
        )
        
        # Train model based on type
        if self.model_type == 'xgboost':
            self.model = self.train_xgboost(X_train, y_train, X_val, y_val)
        elif self.model_type == 'lightgbm':
            self.model = self.train_lightgbm(X_train, y_train, X_val, y_val)
        elif self.model_type == 'lstm':
            self.model = self.train_lstm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Evaluate on test set
        if self.model_type == 'lstm':
            X_test_scaled = self.scaler.transform(X_test)
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            y_pred = (self.model.predict(X_test_reshaped) > 0.5).astype(int).flatten()
        else:
            y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Set Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.array or pd.DataFrame): Feature data
        
        Returns:
            np.array: Predictions (1 for buy, 0 for sell)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert DataFrame to array if needed
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        
        # Make predictions
        if self.model_type == 'lstm':
            if self.scaler is None:
                raise ValueError("Scaler not initialized. Model may not be trained.")
            X_scaled = self.scaler.transform(X)
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            predictions = (self.model.predict(X_reshaped) > 0.5).astype(int).flatten()
        else:
            predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (np.array or pd.DataFrame): Feature data
        
        Returns:
            np.array: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert DataFrame to array if needed
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        
        # Get probabilities
        if self.model_type == 'lstm':
            if self.scaler is None:
                raise ValueError("Scaler not initialized. Model may not be trained.")
            X_scaled = self.scaler.transform(X)
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            probabilities = self.model.predict(X_reshaped).flatten()
        else:
            probabilities = self.model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if self.model_type == 'lstm':
            # Save Keras model
            self.model.save(filepath)
            # Save scaler separately
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            # Save sklearn-style model
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save feature columns
        feature_path = filepath.replace('.pkl', '_features.pkl').replace('.h5', '_features.pkl')
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        if self.model_type == 'lstm':
            # Load Keras model
            self.model = keras.models.load_model(filepath)
            # Load scaler
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            # Load sklearn-style model
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
        
        # Load feature columns
        feature_path = filepath.replace('.pkl', '_features.pkl').replace('.h5', '_features.pkl')
        with open(feature_path, 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_loader import CryptoDataLoader
    from features import FeatureEngineer
    
    # Load and prepare data
    loader = CryptoDataLoader('binance')
    df = loader.get_ohlcv_data('ETH/USDT', timeframe='1d', days=365)
    df = loader.clean_data(df)
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    features_df['target'] = engineer.create_target(features_df, prediction_horizon=1)
    
    # Train model
    predictor = CryptoPredictor(model_type='xgboost')
    predictor.train(features_df)
    
    # Save model
    predictor.save_model('models/eth_xgboost.pkl')

