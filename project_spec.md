# Project Specification: Profit-Copilot

## 1. Project Overview
**Profit-Copilot** is an AI-driven decision support system for manufacturing enterprises. It integrates multi-source data (IoT, Quality, Orders, Finance) to predict profit, analyze root causes, and simulate optimization strategies.

## 2. Tech Stack (Strictly Enforced)
- **Framework:** Streamlit (Python) for the Web UI.
- **Data Processing:** Pandas, NumPy.
- **Machine Learning:** TensorFlow/Keras (LSTM models) or PyTorch.
- **Database:** SQLite (via `sqlite3` or `SQLAlchemy`) for storing aligned data.
- **Visualization:** Plotly Express (Interactive charts).

## 3. Directory Structure
Project Root/
├── app.py                 # Main entry point (Navigation)
├── project_spec.md        # This file
├── mock_data/             # Folder containing the generated CSVs
├── data/                  # Folder for SQLite DB (`profit.db`)
├── pages/                 # Streamlit pages
│   ├── 1_📈_利润预测.py
│   ├── 2_🔍_差异分析.py
│   ├── 3_💡_决策模拟.py
│   ├── 4_💾_数据源管理.py
│   └── 5_🛠️_系统设置.py
├── utils/                 # Helper modules
│   ├── data_pipeline.py   # The ETL logic (Cleaning & Alignment)
│   ├── db_manager.py      # Database CRUD operations
│   └── model_engine.py    # Prediction & Training logic
└── requirements.txt       # Dependencies

## 4. UI/UX Guidelines
- **Theme:** Light Mode (Background: #FFFFFF or #F0F2F6).
- **Color Palette:** - Professional Grey: Text and neutral metrics.
  - Alert Red (#FF4B4B): For negative variances (Cost > Budget).
  - Optimization Green (#00C853): For positive gains/revenue.
- **Layout:** Clean, card-based dashboard. Sidebar for navigation only.

## 5. Core Data Flow
1. **Input:** User uploads CSVs (or system reads from `mock_data/`).
2. **Process:** `utils/data_pipeline.py` aligns multi-source data into a unified daily tensor.
3. **Store:** Aligned data is saved to SQLite (`data/profit.db`).
4. **Analyze:** `utils/model_engine.py` reads from SQLite to train/predict.
5. **Display:** Pages read from SQLite/Model results to visualize with Plotly.
