import pandas as pd
import sqlite3
import os

class DataPipeline:
    def __init__(self, db_path='data/profit.db'):
        self.db_path = db_path
        self.raw_data = {}
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def load_raw_csv(self):
        """Load raw CSVs and convert timestamps."""
        print("Loading raw CSVs...")
        
        # Define file paths
        files = {
            'device': 'mock_data/device_status.csv',
            'quality': 'mock_data/quality_log.csv',
            'master': 'mock_data/master_data_costs.csv',
            'financial': 'mock_data/financial_ledger.csv',
            # 'orders': 'mock_data/production_orders.csv' # Not strictly needed for daily aggregation if covered by others
        }
        
        for key, path in files.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                self.raw_data[key] = df
            else:
                raise FileNotFoundError(f"File not found: {path}")

        # Convert timestamps
        self.raw_data['device']['timestamp'] = pd.to_datetime(self.raw_data['device']['timestamp'])
        self.raw_data['quality']['check_time'] = pd.to_datetime(self.raw_data['quality']['check_time'])
        self.raw_data['master']['date'] = pd.to_datetime(self.raw_data['master']['date'])
        self.raw_data['financial']['entry_date'] = pd.to_datetime(self.raw_data['financial']['entry_date'])
        
        print("Data loaded successfully.")

    def process_alignment(self):
        """Align all data sources to Daily Granularity."""
        print("Processing alignment...")
        
        # Step A: Device Data (Minute -> Daily)
        device_df = self.raw_data['device'].set_index('timestamp')
        # Resample to daily, grouping by nothing (just time) or we could group by machine if needed.
        # Requirement says "Unified Daily Tensor", usually implies aggregate across whole factory or per machine?
        # "Integrates multi-source data... to predict profit". Profit is usually enterprise level or order level.
        # Let's aggregate across ALL machines for the daily enterprise view first.
        
        daily_device = device_df.resample('D')[['vibration', 'temperature', 'power_usage']].agg(['mean', 'std'])
        # Flatten columns
        daily_device.columns = ['_'.join(col).strip() for col in daily_device.columns]
        daily_device = daily_device.rename(columns={
            'vibration_mean': 'device_vib_mean', 'vibration_std': 'device_vib_std',
            'temperature_mean': 'device_temp_mean', 'temperature_std': 'device_temp_std',
            'power_usage_mean': 'device_power_mean', 'power_usage_std': 'device_power_std'
        })
        
        # Step B: Quality Data (Batch -> Daily)
        quality_df = self.raw_data['quality'].copy()
        quality_df['date'] = quality_df['check_time'].dt.normalize()
        daily_quality = quality_df.groupby('date')[['passed_qty', 'failed_qty']].sum()
        
        # Calculate failure rate
        total_qty = daily_quality['passed_qty'] + daily_quality['failed_qty']
        daily_quality['daily_failure_rate'] = daily_quality['failed_qty'] / total_qty
        daily_quality['daily_failure_rate'] = daily_quality['daily_failure_rate'].fillna(0)
        
        # Step C: Master Data (Already Daily)
        master_df = self.raw_data['master'].set_index('date')
        
        # Step D: Financial Data (Transactional -> Daily)
        fin_df = self.raw_data['financial'].copy()
        # entry_date is already date only (from generate_data logic), but ensure it's datetime normalized
        fin_df['date'] = pd.to_datetime(fin_df['entry_date'])
        daily_fin = fin_df.groupby('date')[['revenue', 'profit', 'cost_energy', 'cost_material', 'cost_labor']].sum()
        
        # Step E: Merge
        # Master DF is the anchor (continuous timeline)
        merged_df = master_df.join(daily_device, how='left')
        merged_df = merged_df.join(daily_quality[['daily_failure_rate', 'failed_qty']], how='left')
        merged_df = merged_df.join(daily_fin, how='left')
        
        # Fill NaNs
        # Device data might be missing on non-production days? Or just fill with 0/mean?
        # Financials might be missing if no orders shipped that day. Fill with 0.
        merged_df[['revenue', 'profit', 'cost_energy', 'cost_material', 'cost_labor', 'failed_qty']] = \
            merged_df[['revenue', 'profit', 'cost_energy', 'cost_material', 'cost_labor', 'failed_qty']].fillna(0)
            
        # Forward fill sensor data (assuming sensors stay same if idle? or 0?)
        # Better to fill with interpolation or ffill for sensors to avoid jumps.
        merged_df = merged_df.ffill().fillna(0)
        
        print(f"Aligned data shape: {merged_df.shape}")
        return merged_df

    def save_to_db(self, df):
        """Save processed data to SQLite."""
        print(f"Saving to {self.db_path}...")
        conn = sqlite3.connect(self.db_path)
        df.to_sql('unified_daily_data', conn, if_exists='replace', index=True)
        conn.close()
        print("Save complete.")

    def run(self):
        """Orchestrate the pipeline."""
        self.load_raw_csv()
        final_df = self.process_alignment()
        self.save_to_db(final_df)
        return f"Pipeline completed. Saved {len(final_df)} rows."

if __name__ == "__main__":
    pipeline = DataPipeline()
    result = pipeline.run()
    print(result)
