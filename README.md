# AetherGuard

AetherGuard is a comprehensive solution designed to detect malicious activities in Ethereum transactions using machine learning. This project employs a Graph Neural Network (GNN) model, trained on datasets from the Etherscan API, and deploys the model through a Flask API. The solution is integrated into a Discord server, providing real-time predictions and an AI chatbot for inquiries about AI, machine learning, and web3.

## Table of Contents

1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
4. [Model Creation](#model-creation)
5. [Model Evaluation](#model-evaluation)
6. [Deployment](#deployment)
7. [Discord Integration](#discord-integration)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

## Introduction

AetherGuard aims to provide a robust framework for detecting malicious activities in Ethereum transactions. By leveraging machine learning, the solution can analyze transaction data and classify it as either malicious or non-malicious.

## Datasets

The datasets are sourced from the Etherscan API. The data includes information on Ethereum transactions and is used to train the machine learning model.

## Preprocessing and Feature Engineering

Before training the model, the dataset undergoes several preprocessing steps:
- **Data Cleaning:** Removing any irrelevant or noisy data.
- **Feature Engineering:** Creating meaningful features that help improve the model's performance.
- **Data Splitting:** Dividing the dataset into training and testing sets.

## Model Creation

A Graph Neural Network (GNN) model is created using the processed datasets. This involves the following steps:
- **Defining the Model Architecture:** Creating a neural network architecture suitable for the task.
- **Training the Model:** Using the training dataset to teach the model how to classify transactions.
- **Fine-tuning the Model:** Adjusting hyperparameters to optimize the model's performance.

## Model Evaluation

The GNN model is evaluated using various metrics to assess its performance:
- **Confusion Matrix:** Visualizing the true positives, true negatives, false positives, and false negatives.
- **Precision:** Measuring the accuracy of the positive predictions.
- **F1 Score:** Calculating the harmonic mean of precision and recall.
- **Recall:** Measuring the ability of the model to identify positive instances.

## Deployment

The trained GNN model is saved using the `pickle` module and deployed using a Flask API. This allows the model to be accessed and used for real-time predictions.

## Discord Integration

The solution is integrated into a Discord server to provide real-time predictions and an AI chatbot:
- **Prediction Bot:** A bot that predicts whether a transaction is malicious or not.
- **AI Chatbot:** An interactive chatbot that answers basic questions about AI, machine learning, and web3.

## Usage

To use AetherGuard, follow these steps:

1. **Obtain datasets from the Etherscan API:**
   - Access the Etherscan API to gather transaction data.

2. **Preprocess, clean, and perform feature engineering on the data:**
   - Clean the data to remove noise.
   - Engineer features to improve model performance.
   - Split the data into training and testing sets.

3. **Train the GNN model and evaluate it using the mentioned metrics:**
   - Define the model architecture.
   - Train the model using the training dataset.
   - Evaluate the model using confusion matrix, precision, F1 score, and recall.

4. **Save the trained model using `pickle` and deploy it using Flask API:**
   - Save the model using the `pickle` module.
   - Deploy the model using a Flask API to enable real-time predictions.

5. **Integrate the solution into a Discord server for real-time predictions and AI chatbot interactions:**
   - Set up a Discord server.
   - Add the prediction bot to the server.
   - Integrate the AI chatbot for inquiries about AI, machine learning, and web3.

## Contributing

We welcome contributions to enhance AetherGuard! Here are some ways you can contribute:
- Report bugs
- Suggest new features
- Submit pull requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

We would like to thank the following for their support and contributions:
- Etherscan API for providing the transaction data
- Contributors and the open-source community for their valuable insights
- Discord for providing the platform for integration

