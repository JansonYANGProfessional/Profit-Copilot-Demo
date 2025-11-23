import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
import joblib
import os
from tensorflow.keras.models import load_model
from utils.model_engine import ModelEngine

st.set_page_config(page_title="决策模拟", layout="wide")

st.title("💡 决策模拟 (What-If Simulator)")
st.markdown("通过调整关键经营参数，模拟未来30天的利润变化。")
st.markdown("---")

# Check Resources
if not os.path.exists('data/profit_model.h5') or not os.path.exists('data/scaler.pkl'):
    st.warning("⚠️ 模型未训练。请前往数据源管理页面进行训练。")
    st.stop()

# --- Helper Function for Simulation ---
# We replicate the prediction logic here to allow injecting modified data
def run_simulation(vib_modifier, mat_modifier):
    # 1. Load Resources
    model = load_model('data/profit_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mse')
    scaler = joblib.load('data/scaler.pkl')
    
    engine = ModelEngine() # To get config
    window_size = engine.window_size
    feature_cols = engine.feature_cols
    target_cols = engine.target_cols
    
    # 2. Load Baseline Data (Last Window)
    conn = sqlite3.connect('data/profit.db')
    df = pd.read_sql(f"SELECT * FROM unified_daily_data ORDER BY date DESC LIMIT {window_size}", conn)
    conn.close()
    
    # Preprocess
    df['cogs'] = df['cost_material'] + df['cost_labor']
    df['expenses'] = df['cost_energy']
    df = df.sort_values('date', ascending=True)
    
    # 3. Apply Modifiers (The Simulation Step)
    # vib_modifier is percentage (e.g. -10 for -10%)
    # mat_modifier is percentage
    
    sim_df = df.copy()
    
    # Apply to Vibration columns
    sim_df['device_vib_mean'] = sim_df['device_vib_mean'] * (1 + vib_modifier / 100.0)
    sim_df['device_vib_std'] = sim_df['device_vib_std'] * (1 + vib_modifier / 100.0)
    
    # Apply to Material Price
    sim_df['material_price_index'] = sim_df['material_price_index'] * (1 + mat_modifier / 100.0)
    
    # 4. Predict
    # Scale
    input_data = sim_df[feature_cols + target_cols].values
    scaled_input = scaler.transform(input_data)
    
    # Initial Sequence
    current_seq = scaled_input[:, :len(feature_cols)]
    current_seq = current_seq.reshape((1, window_size, len(feature_cols)))
    
    predictions = []
    days = 30
    
    for _ in range(days):
        pred_targets = model.predict(current_seq, verbose=0)
        
        # Use last known features (from the simulated dataframe)
        last_features = current_seq[0, -1, :]
        next_step_features = last_features.reshape(1, 1, -1)
        
        current_seq = np.concatenate([current_seq[:, 1:, :], next_step_features], axis=1)
        predictions.append(pred_targets[0])
        
    # Inverse Transform
    pred_array = np.array(predictions)
    dummy_features = np.zeros((days, len(feature_cols)))
    to_inverse = np.hstack([dummy_features, pred_array])
    inversed = scaler.inverse_transform(to_inverse)
    final_preds = inversed[:, len(feature_cols):]
    
    # Calculate Total Profit
    # Pred cols: Revenue, COGS, Expenses
    # Profit = Revenue - (COGS + Expenses)
    total_rev = np.sum(final_preds[:, 0])
    total_cost = np.sum(final_preds[:, 1]) + np.sum(final_preds[:, 2])
    total_profit = total_rev - total_cost
    
    return total_profit

# --- UI Controls ---
col_controls, col_results = st.columns([1, 2])

with col_controls:
    st.subheader("🎛️ 参数调整")
    
    vib_slider = st.slider(
        "设备维护水平 (Device Maintenance)",
        min_value=-50, max_value=50, value=0, step=5,
        help="负值表示维护更好（震动减少），正值表示维护更差。"
    )
    
    mat_slider = st.slider(
        "原材料价格波动 (Material Price)",
        min_value=-20, max_value=20, value=0, step=1,
        help="模拟原材料市场价格的变化。"
    )
    
    st.info("调整滑块以模拟不同场景下的利润表现。")
    
    if st.button("💾 保存策略 (Apply Strategy)"):
        st.toast("Strategy saved to optimization log!", icon="💾")

# --- Simulation & Results ---
with col_results:
    st.subheader("📊 模拟结果预测")
    
    with st.spinner("正在计算模拟结果..."):
        # Run Baseline (0, 0)
        baseline_profit = run_simulation(0, 0)
        
        # Run Simulation
        sim_profit = run_simulation(vib_slider, mat_slider)
        
        profit_delta = sim_profit - baseline_profit
        
    # Metrics
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric("基准利润 (Baseline)", f"${baseline_profit:,.0f}")
        
    with m2:
        st.metric("模拟利润 (Simulated)", f"${sim_profit:,.0f}", delta=f"{sim_profit-baseline_profit:,.0f}")
        
    with m3:
        color = "normal" if profit_delta >= 0 else "inverse"
        st.metric("净收益/损失 (Net Gain/Loss)", f"${profit_delta:,.0f}", delta_color=color)
        
    # Visualization
    chart_data = pd.DataFrame({
        'Scenario': ['Baseline', 'Simulated'],
        'Profit': [baseline_profit, sim_profit],
        'Color': ['grey', 'blue']
    })
    
    fig = px.bar(
        chart_data, 
        x='Scenario', 
        y='Profit', 
        color='Scenario',
        title="利润对比 (Profit Comparison)",
        text_auto='.2s',
        color_discrete_map={'Baseline': 'grey', 'Simulated': '#2962FF'}
    )
    st.plotly_chart(fig, use_container_width=True)
