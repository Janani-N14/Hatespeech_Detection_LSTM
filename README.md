# Hatespeech_Detection_LSTM
Hate Speech Detection Model
Overview
This project implements a deep learning model for hate speech detection in tweets. The model uses a combination of embedding, LSTM (Long Short-Term Memory), and dense layers to classify text as either hate speech (1) or normal content (0).
Model Architecture
The model is a sequential neural network with the following layers:

Embedding Layer: Converts input tokens into dense vectors (embedding_dim=16)
Dropout Layer: Reduces overfitting (rate=0.4)
LSTM Layer: Processes sequential text data with regularization
Flatten Layer: Flattens the output for dense layers
Dense Layers: Multiple fully connected layers with regularization
Output Layer: Single sigmoid node for binary classification
Dataset
The dataset contains 31,962 tweets labeled as either hate speech (1) or not hate speech (0). Each row contains a tweet and its corresponding label.

Hyperparameters

max_features: 50,000 (vocabulary size)
embedding_dim: 16 (dimensionality of embedding space)
sequence_length: defined by maxlen (maximum length of input sequences)
Loss function: Binary Cross-Entropy
Optimizer: Adam with learning rate 1e-3
Metric: Binary Accuracy

Requirements

TensorFlow 2.x
NumPy
Pandas
scikit-learn (for data preprocessing)


Model Features

Regularization: L2 regularization used throughout to prevent overfitting
Dropout: Multiple dropout layers to improve generalization
LSTM: Captures sequential patterns in text data
Binary Classification: Outputs probability that a text contains hate speech

