# Stock Prediction App

This project is a Stock Prediction App built with Streamlit, yFinance, and TensorFlow. It allows users to explore stock financial data, visualise historical stock prices, compute moving averages, and train a machine learning model to predict stock prices.

## Features

- **Explore Stocks**: View financial data and historical prices for selected stocks.
- **Moving Averages**: Calculate and visualise moving averages for stock prices.
- **Train LSTM Model**: Train an LSTM model to predict future stock prices.

## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` (Python package installer)

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Create a virtual environment**:

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

4. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

5. **Run the application**:

    ```bash
    streamlit run streamlitML.py
    ```

## Usage

1. **Select a stock symbol** from the sidebar or enter a custom symbol.
2. **Choose an activity**:
    - **Explore Stocks**: View financials, price data, or moving averages.
    - **Train Model**: Set the number of epochs and batch size, then train an LSTM model to predict stock prices.
3. **View the results**: Visualise historical data, moving averages, and model predictions directly in the app.

## Project Structure

- `streamlitML.py`: Main application script.
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Git ignore file to exclude virtual environment and other unnecessary files.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
