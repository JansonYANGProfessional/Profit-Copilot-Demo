# Profit-Copilot MVP

**Profit-Copilot** is an AI-driven decision support system for manufacturing enterprises. It integrates multi-source data (IoT, Quality, Orders, Finance) to predict profit, analyze root causes, and simulate optimization strategies.
<img width="1539" height="839" alt="image" src="https://github.com/user-attachments/assets/9537cbc3-1753-4c47-ba4e-8d885d23d24e" />


## Features

- **📈 Profit Prediction**: AI-powered forecasting of future revenue, costs, and profit.
- **🔍 Attribution Analysis**: Identify the operational drivers (e.g., vibration, failure rate) behind financial variances.
- **💡 Decision Simulation**: "What-If" simulator to test strategies like improving maintenance or adjusting material procurement.
- **💾 Data Manager**: Full control over data ingestion, generation, and model training.

## Tech Stack

- **Frontend**: Streamlit
- **Data**: Pandas, NumPy, SQLite
- **AI/ML**: TensorFlow (Keras), Scikit-learn
- **Viz**: Plotly Express

## How to Run Locally

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run app.py
    ```

## Demo Instructions

1.  Go to **Data Source Manager**.
2.  Click **"Re-generate Mock Data"** (Select "Crisis Mode" to see anomaly detection).
3.  Click **"Run Full Pipeline"** to process data and retrain the AI model.
4.  Navigate to **Profit Prediction** to see the forecast.
5.  Use **Decision Simulation** to find a strategy to mitigate the crisis.
