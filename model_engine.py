"""
BiLSTM model and inference engine. Supports both checkpoint formats:
- dict with 'model_state_dict' (training checkpoint)
- raw state_dict
"""
import logging
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM for occupancy detection (must match training)."""

    def __init__(self, input_size, hidden_size_1=128, hidden_size_2=64, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        self.bn1 = nn.BatchNorm1d(hidden_size_1 * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1 * 2,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        self.bn2 = nn.BatchNorm1d(hidden_size_2 * 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size_2 * 2, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = lstm1_out.permute(0, 2, 1)
        lstm1_out = self.bn1(lstm1_out)
        lstm1_out = lstm1_out.permute(0, 2, 1)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, (hidden, cell) = self.lstm2(lstm1_out)
        hidden_forward = hidden[0, :, :]
        hidden_backward = hidden[1, :, :]
        hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)
        hidden_cat = self.bn2(hidden_cat)
        hidden_cat = self.dropout2(hidden_cat)
        fc1_out = self.fc1(hidden_cat)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout3(fc1_out)
        output = self.fc2(fc1_out)
        output = self.sigmoid(output)
        return output


class InferenceEngine:
    """Load model/scaler and run inference. Supports checkpoint with 'model_state_dict' or raw state_dict."""

    def __init__(self, model_path: str, scaler_path: str, receiver_name: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.receiver_name = receiver_name
        self.model_path = model_path
        self.scaler_path = scaler_path

        logger.info("Loading model from: %s", model_path)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = checkpoint

        self.model = BiLSTMModel(input_size=34).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Model loaded successfully")

        logger.info("Loading scaler from: %s", scaler_path)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")

    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Predict on feature sequence (30, 34). Returns (is_occupied, probability).
        """
        n_timesteps, n_features = features.shape
        features_reshaped = features.reshape(-1, n_features)
        features_normalized = self.scaler.transform(features_reshaped)
        features_normalized = features_normalized.reshape(1, n_timesteps, n_features)
        features_tensor = torch.FloatTensor(features_normalized).to(self.device)
        with torch.no_grad():
            output = self.model(features_tensor)
            probability = output.item()
        is_occupied = probability > 0.5
        return is_occupied, probability
