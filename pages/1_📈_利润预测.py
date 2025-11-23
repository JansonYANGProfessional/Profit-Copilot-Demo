import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import os
from utils.model_engine import ModelEngine

st.set_page_config(page_title="利润预测", layout="wide")

st.title("📈 利润预测 Dashboard")
st.markdown("---")

# Check if system is initialized
if not os.path.exists('data/profit.db') or not os.path.exists('data/profit_model.h5'):
    st.warning("⚠️ 系统尚未初始化。请前往 **💾 数据源管理** 页面上传数据并初始化模型。")
    st.stop()

try:
    # --- 1. Load Data ---
    engine = ModelEngine()
    
    # Forecast (Next 30 Days)
    with st.spinner("正在生成 AI 预测..."):
        forecast_df = engine.predict_future(days=30)
    
    # Historical (Last 90 Days)
    conn = sqlite3.connect('data/profit.db')
    history_df = pd.read_sql("SELECT * FROM unified_daily_data ORDER BY date DESC LIMIT 90", conn)
    conn.close()
    
    # Process History DF
    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df = history_df.sort_values('date')
    history_df['Type'] = 'Historical'
    
    # Process Forecast DF
    forecast_df = forecast_df.rename(columns={
        'Date': 'date',
        'Predicted_Revenue': 'revenue',
        'Predicted_Cost': 'cogs', # Mapping Cost -> COGS for consistency
        'Predicted_Expenses': 'expenses'
    })
    forecast_df['date'] = pd.to_datetime(forecast_df['date']) # Ensure datetime64
    forecast_df['profit'] = forecast_df['revenue'] - (forecast_df['cogs'] + forecast_df['expenses'])
    forecast_df['Type'] = 'Forecast'
    
    # Combine
    # We need to align columns. History has many, Forecast has few.
    # Let's select common columns for plotting.
    common_cols = ['date', 'revenue', 'profit', 'Type']
    # History has 'cost_material', 'cost_labor', 'cost_energy'. 
    # Forecast has 'cogs' (mat+lab) and 'expenses' (energy).
    # Let's construct 'total_cost' for comparison.
    
    history_df['total_cost'] = history_df['cost_material'] + history_df['cost_labor'] + history_df['cost_energy']
    forecast_df['total_cost'] = forecast_df['cogs'] + forecast_df['expenses']
    
    combined_df = pd.concat([
        history_df[['date', 'revenue', 'profit', 'total_cost', 'Type']],
        forecast_df[['date', 'revenue', 'profit', 'total_cost', 'Type']]
    ], ignore_index=True)
    
    # Ensure combined_df date is datetime
    combined_df['date'] = pd.to_datetime(combined_df['date'])

    # --- 2. Financial Overview ---
    
    # Calculate Metrics
    last_30_days_hist = history_df.iloc[-30:]
    next_30_days_pred = forecast_df
    
    total_rev_hist = last_30_days_hist['revenue'].sum()
    total_rev_pred = next_30_days_pred['revenue'].sum()
    rev_delta = total_rev_pred - total_rev_hist
    
    total_cost_hist = last_30_days_hist['total_cost'].sum()
    total_cost_pred = next_30_days_pred['total_cost'].sum()
    cost_delta = total_cost_pred - total_cost_hist
    
    total_profit_hist = last_30_days_hist['profit'].sum()
    total_profit_pred = next_30_days_pred['profit'].sum()
    profit_delta = total_profit_pred - total_profit_hist
    
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="预测总营收 (未来30天)",
            value=f"${total_rev_pred:,.0f}",
            delta=f"{rev_delta:,.0f} vs 上月",
            delta_color="normal" # Green if up
        )
        
    with col2:
        st.metric(
            label="预测总成本 (未来30天)",
            value=f"${total_cost_pred:,.0f}",
            delta=f"{cost_delta:,.0f} vs 上月",
            delta_color="inverse" # Red if up
        )
        
    with col3:
        st.metric(
            label="预测净利润 (未来30天)",
            value=f"${total_profit_pred:,.0f}",
            delta=f"{profit_delta:,.0f} vs 上月",
            delta_color="normal"
        )

    # --- 3. Main Chart ---
    st.subheader("📊 营收与利润趋势分析")
    
    # Melt for Plotly
    plot_df = combined_df.melt(id_vars=['date', 'Type'], value_vars=['revenue', 'total_cost', 'profit'], var_name='Metric', value_name='Value')
    
    # Rename metrics for display
    metric_map = {'revenue': '营收 (Revenue)', 'total_cost': '总成本 (Cost)', 'profit': '净利润 (Profit)'}
    plot_df['Metric'] = plot_df['Metric'].map(metric_map)
    
    # Define colors
    color_map = {
        '营收 (Revenue)': '#00C853', # Green
        '总成本 (Cost)': '#FF4B4B',   # Red
        '净利润 (Profit)': '#2962FF'  # Blue
    }
    
    fig = px.line(
        plot_df, 
        x='date', 
        y='Value', 
        color='Metric', 
        line_dash='Type', # Solid for Hist, Dashed for Forecast
        color_discrete_map=color_map,
        title="历史回顾 vs AI 预测",
        height=500
    )
    
    # Add a vertical line at the transition point
    last_hist_date = history_df['date'].max()
    # Use numeric timestamp (milliseconds) to avoid Pandas Timestamp addition error in Plotly
    fig.add_vline(x=last_hist_date.timestamp() * 1000, line_width=1, line_dash="dash", line_color="grey", annotation_text="预测开始")
    
    st.plotly_chart(fig, use_container_width=True)

    # --- 4. Data Table ---
    with st.expander("📋 查看详细预测数据"):
        st.dataframe(forecast_df.style.format({
            'revenue': '${:,.2f}',
            'cogs': '${:,.2f}',
            'expenses': '${:,.2f}',
            'profit': '${:,.2f}',
            'total_cost': '${:,.2f}'
        }))

except Exception as e:
    st.error(f"发生错误: {e}")
    st.info("请检查数据库连接或重新运行数据管道。")
