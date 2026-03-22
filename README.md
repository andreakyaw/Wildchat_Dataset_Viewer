# WildChat Dataset Visualization Interface

## Overview
This is an visualization interface for exploring the **WildChat-1M dataset**, a dataset with real ChatGPT conversations.

## Dataset
**WildChat-1M dataset from AllenAI:**
https://huggingface.co/datasets/allenai/WildChat-1M

## Features

### Interactions
- Filter by language on the side bar
- Filter by model on the side bar
- View conversation text by double clicking

## Setup Instructions

Run all these commands in your terminal

### 1. Install dependencies
pip install streamlit pandas pyarrow datasets

### 2. Download the dataset
python scripts/download_dataset.py

### 3. Run the interface
python -m streamlit run interface.py
