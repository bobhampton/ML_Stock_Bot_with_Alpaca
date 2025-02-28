# Alpaca ML Crypto Stock Analysis

This repository is a group project for CPE-595 that leverages the Alpaca API to fetch cryptocurrency/stock data and uses three different machine learning algorithms to analyze and predict market trends. The project includes data fetching, preprocessing, and training.

## Features

- Fetch crypto/stock data using the Alpaca API
- Preprocess data for machine learning
- Train model to predict stock prices
- Visualize actual vs predicted stock prices

## Requirements

- Python 3.8+ (please note that this project uses tensorflow and at the time of writing this, only supports Python 3.9-3.12) 
- Alpaca API key and secret
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/alpaca-ml-crypto-stock-analysis.git
    cd alpaca-ml-crypto-stock-analysis
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a [.env](http://_vscodecontentref_/0) file in the root directory and add your Alpaca API credentials:
    ```env
    ALPACA_API_KEY=your_api_key
    ALPACA_SECRET_KEY=your_secret_key
    ```

## Usage

1. Run the main script:
    ```sh
    python main.py
    ```

2. The script will fetch cryptocurrency data for BTC/USD, train an LSTM model, and plot the actual vs predicted stock prices.

## Project Structure

- [main.py](http://_vscodecontentref_/1): Main script to fetch data, train the model, and visualize results.
- [requirements.txt](http://_vscodecontentref_/2): List of required Python packages.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
