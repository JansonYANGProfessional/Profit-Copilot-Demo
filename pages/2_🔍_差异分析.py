import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import os

st.set_page_config(page_title="差异分析", layout="wide")

st.title("🔍 差异分析 (Attribution Analysis)")
st.markdown("---")

# Check DB
if not os.path.exists('data/profit.db'):
    st.warning("⚠️ 数据库未找到。请先在数据源管理中初始化系统。")
    st.stop()

# 1. Load Data
conn = sqlite3.connect('data/profit.db')
df = pd.read_sql("SELECT * FROM unified_daily_data", conn)
conn.close()

if df.empty:
    st.warning("⚠️ 数据为空。")
    st.stop()

# Preprocessing for Analysis
# Create explicit targets if they don't exist in the raw table (though pipeline creates them, let's be safe)
if 'cogs' not in df.columns:
    df['cogs'] = df['cost_material'] + df['cost_labor']
if 'expenses' not in df.columns:
    df['expenses'] = df['cost_energy']
    
# Selectable Targets
target_options = {
    'revenue': '营收 (Revenue)',
    'profit': '利润 (Profit)',
    'cogs': '成本 (COGS)',
    'expenses': '费用 (Expenses)',
    'daily_failure_rate': '次品率 (Failure Rate)'
}

# UI: Target Selection
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("🎯 分析目标")
    selected_target_key = st.selectbox("选择要分析的指标:", list(target_options.keys()), format_func=lambda x: target_options[x])
    
# 2. Correlation Analysis
# Define Operational Drivers (The "Why")
driver_features = [
    'device_vib_mean', 'device_vib_std', 
    'device_temp_mean', 'device_temp_std',
    'daily_failure_rate', 
    'electricity_price', 'material_price_index', 'labor_cost_rate'
]

# Ensure these columns exist in DF
available_drivers = [col for col in driver_features if col in df.columns]

if not available_drivers:
    st.error("未找到驱动因子列。请检查数据管道。")
    st.stop()

# Calculate Correlation
# We only care about correlation between Target and Drivers
corr_data = df[available_drivers + [selected_target_key]].corr()
target_corrs = corr_data[selected_target_key].drop(selected_target_key)

# Sort by absolute correlation to find strongest drivers
target_corrs_abs = target_corrs.abs().sort_values(ascending=False)
top_drivers = target_corrs.loc[target_corrs_abs.index[:5]] # Top 5

# 3. Visuals
with col2:
    st.subheader(f"📊 影响 {target_options[selected_target_key]} 的关键因素")
    st.caption("注：仅分析运营驱动因子，排除财务构成项。")
    
    # Bar Chart
    fig_bar = px.bar(
        x=top_drivers.values,
        y=top_drivers.index,
        orientation='h',
        title="Top 5 运营驱动因子 (Operational Drivers)",
        labels={'x': 'Correlation', 'y': 'Feature'},
        color=top_drivers.values,
        color_continuous_scale=px.colors.diverging.Tealrose
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# Deep Dive Scatter
st.subheader("🔎 因子深度洞察")

# Select a driver to visualize
selected_driver = st.selectbox("选择因子查看趋势:", top_drivers.index)

col_chart, col_text = st.columns([2, 1])

with col_chart:
    fig_scatter = px.scatter(
        df, 
        x=selected_driver, 
        y=selected_target_key, 
        trendline="ols",
        title=f"{selected_driver} vs {selected_target_key}",
        opacity=0.6
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_text:
    st.markdown("### 💡 智能洞察")
    
    corr_val = top_drivers[selected_driver]
    impact_type = "正向" if corr_val > 0 else "负向"
    strength = "强" if abs(corr_val) > 0.7 else ("中等" if abs(corr_val) > 0.3 else "弱")
    
    st.info(f"""
    分析显示 **{selected_driver}** 对 **{target_options[selected_target_key]}** 有 **{strength}{impact_type}** 影响。
    
    - 相关系数: `{corr_val:.2f}`
    - 建议: 如果希望优化 {target_options[selected_target_key]}，请重点关注 {selected_driver} 的变化趋势。
    """)
