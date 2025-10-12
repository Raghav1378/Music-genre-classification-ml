# Music Genre Classification ML

## Overview

This project aims to classify songs into genres (Hip-Hop vs Non-Hip-Hop) based on audio features such as acousticness, danceability, energy, loudness, tempo, and more.

Currently, the project includes:

- Data extraction from Spotify dataset
- Exploratory Data Analysis (EDA)
  - Histograms of features
  - Skewness analysis
  - Boxplots and outlier detection
  - Comparison with normal distribution
- Outlier capping and preprocessing

## Dataset

The dataset used is from [Kaggle: Spotify Classification Dataset](https://www.kaggle.com/datasets/geomack/spotifyclassification).

## Project Structure

music-genre-classification-ml/
│
├── notebooks/ # Jupyter notebooks for EDA and experiments
│ └── 01_data_eda.ipynb
├── data/ # Sample or referenced dataset
├── models/ # Trained models (future)
├── app/ # App / frontend (future)
├── README.md # Project overview and instructions
└── .gitignore # To ignore unnecessary files like venv, data
