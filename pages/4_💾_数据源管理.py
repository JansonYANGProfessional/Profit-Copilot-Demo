import streamlit as st
import os
import time
import datetime
import subprocess
import sys
from utils.data_pipeline import DataPipeline
from utils.model_engine import ModelEngine

st.set_page_config(page_title="数据源管理", layout="wide")

st.title("💾 数据源管理 (Data Source Manager)")
st.markdown("---")

# --- 1. System Status Monitor ---
st.subheader("🖥️ 系统状态监控")

db_path = 'data/profit.db'
model_path = 'data/profit_model.h5'

col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists(db_path):
        st.success("🟢 Database Connected")
    else:
        st.error("🔴 Database Missing")

with col2:
    if os.path.exists(model_path):
        st.success("🟢 Model Trained")
    else:
        st.error("🔴 Model Untrained")

with col3:
    if os.path.exists(db_path):
        mod_time = os.path.getmtime(db_path)
        dt_obj = datetime.datetime.fromtimestamp(mod_time)
        st.metric("上次更新时间", dt_obj.strftime('%Y-%m-%d %H:%M:%S'))
    else:
        st.metric("上次更新时间", "N/A")

st.markdown("---")

# --- 2. Data Ingestion ---
st.subheader("📥 数据接入")

ingest_col1, ingest_col2 = st.columns(2)

with ingest_col1:
    st.markdown("#### 📤 上传 CSV 文件")
    st.info("For MVP demo, please use the generated mock files.")
    uploaded_files = st.file_uploader("选择 CSV 文件", accept_multiple_files=True, type=['csv'])
    
    if uploaded_files:
        if not os.path.exists('mock_data'):
            os.makedirs('mock_data')
            
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            file_path = os.path.join('mock_data', uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(bytes_data)
            st.toast(f"Saved {uploaded_file.name}", icon="✅")
        st.success("文件上传成功！")

with ingest_col2:
    st.markdown("#### 🎲 生成模拟数据")
    st.write("点击下方按钮重新生成随机模拟数据 (覆盖现有数据)。")
    
    scenario = st.selectbox(
        "选择模拟场景 (Select Scenario):",
        ["Normal Operation", "📉 Crisis Mode (High Cost)"]
    )
    
    scenario_arg = 'crisis' if "Crisis" in scenario else 'normal'
    
    if st.button("🎲 Re-generate Mock Data"):
        try:
            with st.spinner(f"正在生成数据 (场景: {scenario_arg})..."):
                # Run the generate_data.py script
                # Use sys.executable to ensure we use the current environment's Python
                result = subprocess.run([sys.executable, 'generate_data.py', '--scenario', scenario_arg], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("模拟数据生成成功！")
                    if scenario_arg == 'crisis':
                        st.warning("⚠ 已生成 'Crisis' 数据。原材料价格飙升！请运行数据管道以查看影响。")
                    st.toast("Data Generated", icon="🎉")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"生成失败:\n{result.stderr}")
        except Exception as e:
            st.error(f"执行出错: {e}")

st.markdown("---")

# --- 3. Run Pipeline ---
st.subheader("🚀 系统更新")
st.write("运行全量数据管道：清洗数据 -> 存入数据库 -> 重新训练 AI 模型。")

if st.button("🚀 Run Full Pipeline & Retrain Model", type="primary", use_container_width=True):
    status_container = st.status("正在更新系统...", expanded=True)
    
    try:
        # Step 1: ETL
        status_container.write("Running Data Pipeline (ETL)...")
        pipeline = DataPipeline()
        etl_result = pipeline.run()
        status_container.write(f"✅ {etl_result}")
        
        # Step 2: Training
        status_container.write("Training Neural Network...")
        engine = ModelEngine()
        train_result = engine.train()
        status_container.write(f"✅ {train_result}")
        
        # Step 3: Finalize
        status_container.update(label="System Updated Successfully!", state="complete", expanded=False)
        st.success("系统更新完成！所有模块已同步最新数据。")
        st.balloons()
        
        # Refresh page to show new status
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        status_container.update(label="Update Failed", state="error")
        st.error(f"更新过程中发生错误: {e}")
