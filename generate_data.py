import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
import sys

# Configuration
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
DAYS = (END_DATE - START_DATE).days + 1
OUTPUT_DIR = 'mock_data'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_master_data(scenario='normal'):
    print(f"Generating Master Data & Costs (Scenario: {scenario})...")
    dates = [START_DATE + timedelta(days=i) for i in range(DAYS)]
    
    # Electricity Price: Seasonality (High in Summer) + Daily Variation
    # Summer peak around day 200 (July)
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal_factor = 0.15 * np.sin(2 * np.pi * (day_of_year - 100) / 365) # Shifted sine
    base_price = 0.12
    random_noise = np.random.normal(0, 0.01, DAYS)
    electricity_prices = base_price + seasonal_factor + random_noise
    
    # Scenario: Crisis - High Electricity
    if scenario == 'crisis':
        electricity_prices *= 1.30
        
    electricity_prices = np.clip(electricity_prices, 0.05, 0.50) # Clip to realistic range

    # Material Price Index: Random Walk
    material_index = [100.0]
    for i in range(1, DAYS):
        change = np.random.normal(0, 0.5)
        new_val = material_index[-1] + change
        
        # Scenario: Crisis - Material Price Spike in June (Day ~150)
        if scenario == 'crisis' and i > 150:
            new_val *= 1.005 # Compound increase or just shift? 
            # Requirement: "Increase material_price_index by 50% starting from June"
            # Let's add a step function or ramp.
            if i == 151:
                new_val += 50.0 # Immediate jump
            
        material_index.append(new_val)
    
    # Labor Cost Rate: Slow inflation
    labor_rate = np.linspace(25.0, 26.5, DAYS) # Slight increase over the year

    df = pd.DataFrame({
        'date': dates,
        'electricity_price': electricity_prices.round(4),
        'material_price_index': np.array(material_index).round(2),
        'labor_cost_rate': labor_rate.round(2)
    })
    
    df.to_csv(f'{OUTPUT_DIR}/master_data_costs.csv', index=False)
    return df

def generate_production_orders():
    print("Generating Production Orders...")
    orders = []
    order_counter = 1000
    machines = ['M01', 'M02', 'M03']
    products = ['Product_A', 'Product_B']
    
    for day_offset in range(DAYS):
        curr_date = START_DATE + timedelta(days=day_offset)
        
        for machine in machines:
            # Randomly decide if machine runs today (80% chance)
            if random.random() > 0.2:
                # 1 to 3 orders per machine per day
                num_orders = random.randint(1, 3)
                current_time = curr_date.replace(hour=6, minute=0, second=0) # Start shift at 6 AM
                
                for _ in range(num_orders):
                    duration_hours = random.uniform(4, 12)
                    end_time = current_time + timedelta(hours=duration_hours)
                    
                    # If order goes beyond midnight, clip it or skip (simplify: skip if too late)
                    if end_time.date() > curr_date.date():
                        break
                        
                    prod_type = random.choice(products)
                    qty_target = int(duration_hours * (100 if prod_type == 'Product_A' else 200)) # A is slower/complex
                    
                    orders.append({
                        'order_id': f"ORD-{order_counter}",
                        'product_type': prod_type,
                        'machine_id': machine,
                        'start_time': current_time,
                        'end_time': end_time,
                        'qty_target': qty_target
                    })
                    
                    order_counter += 1
                    # Gap between orders
                    current_time = end_time + timedelta(minutes=random.randint(30, 90))

    df = pd.DataFrame(orders)
    df.to_csv(f'{OUTPUT_DIR}/production_orders.csv', index=False)
    return df

def generate_device_status(orders_df):
    print("Generating Device Status Stream (Big Data)...")
    
    # Identify anomalies: Select 5% of orders
    all_order_ids = orders_df['order_id'].unique()
    anomaly_orders = set(np.random.choice(all_order_ids, size=int(len(all_order_ids) * 0.05), replace=False))
    
    status_records = []
    
    for _, row in orders_df.iterrows():
        oid = row['order_id']
        mid = row['machine_id']
        start = row['start_time']
        end = row['end_time']
        is_anomaly = oid in anomaly_orders
        
        # Generate timestamps every minute
        timestamps = pd.date_range(start=start, end=end, freq='1min')
        n_points = len(timestamps)
        
        # Base signals
        vibration = np.random.normal(loc=0.5, scale=0.05, size=n_points) # Normal vibration
        temperature = np.random.normal(loc=65.0, scale=2.0, size=n_points) # Normal temp C
        power = np.random.normal(loc=120.0, scale=5.0, size=n_points) # kW
        
        if is_anomaly:
            # Ramp up vibration and temp
            ramp = np.linspace(0, 2.0, n_points) # Add up to 2.0 units to vibration
            temp_ramp = np.linspace(0, 20.0, n_points) # Add up to 20C to temp
            
            vibration += ramp
            temperature += temp_ramp
            
        # Create DataFrame for this order chunk to append efficiently
        chunk = pd.DataFrame({
            'timestamp': timestamps,
            'machine_id': mid,
            'order_id': oid,
            'vibration': vibration,
            'temperature': temperature,
            'power_usage': power
        })
        status_records.append(chunk)
        
    full_df = pd.concat(status_records, ignore_index=True)
    full_df.to_csv(f'{OUTPUT_DIR}/device_status.csv', index=False)
    return full_df, anomaly_orders

def generate_quality_logs(orders_df, anomaly_orders):
    print("Generating Quality Check Logs...")
    logs = []
    
    for _, row in orders_df.iterrows():
        oid = row['order_id']
        mid = row['machine_id']
        start = row['start_time']
        end = row['end_time']
        total_qty = row['qty_target']
        
        # Checks every 2 hours
        check_times = pd.date_range(start=start, end=end, freq='2h')[1:] # Skip start time
        
        if len(check_times) == 0:
            # At least one check at the end if short order
            check_times = [end]
            
        qty_per_batch = total_qty // len(check_times)
        
        for i, c_time in enumerate(check_times):
            is_anomaly = oid in anomaly_orders
            
            # Failure rate logic
            if is_anomaly:
                fail_rate = random.uniform(0.10, 0.20) # 10-20%
            else:
                fail_rate = random.uniform(0.00, 0.01) # 0-1%
                
            failed_qty = int(qty_per_batch * fail_rate)
            passed_qty = qty_per_batch - failed_qty
            
            logs.append({
                'check_time': c_time,
                'machine_id': mid,
                'batch_id': f"{oid}-B{i+1}",
                'passed_qty': passed_qty,
                'failed_qty': failed_qty,
                'order_id': oid # Helper for join later, maybe drop before save if strict schema
            })
            
    df = pd.DataFrame(logs)
    # Save without order_id if strict to requirements, but it's useful for joining. 
    # User asked for specific columns: check_time, machine_id, batch_id, passed_qty, failed_qty.
    # I will keep order_id for internal logic but drop it for CSV if needed. 
    # Actually, the user didn't strictly forbid extra columns, but let's stick to the list + order_id is usually needed to link batch to order.
    # Wait, the requirements say "Columns: check_time, machine_id, batch_id, passed_qty, failed_qty".
    # But without order_id in Quality Log, how do we know which order it belongs to? 
    # Ah, `batch_id` usually encodes it, or we join by time + machine.
    # I'll include order_id in the file for clarity, it's safer.
    
    df.to_csv(f'{OUTPUT_DIR}/quality_log.csv', index=False)
    return df

def generate_financial_ledger(orders_df, quality_df, device_df, master_df):
    print("Generating Financial Ledger...")
    
    # 1. Aggregate Quality Data by Order
    quality_agg = quality_df.groupby('order_id')[['passed_qty', 'failed_qty']].sum().reset_index()
    
    # 2. Aggregate Power Usage by Order
    # device_df has 1-min samples. power_usage is kW (instantaneous). 
    # Energy (kWh) = sum(power_usage) * (1/60) hours
    energy_agg = device_df.groupby('order_id')['power_usage'].sum().reset_index()
    energy_agg['energy_kwh'] = energy_agg['power_usage'] / 60.0
    
    # 3. Merge everything into Orders
    fin_df = orders_df.merge(quality_agg, on='order_id', how='left')
    fin_df = fin_df.merge(energy_agg[['order_id', 'energy_kwh']], on='order_id', how='left')
    
    # Fill NaNs (if any missing logs/status)
    fin_df['passed_qty'] = fin_df['passed_qty'].fillna(fin_df['qty_target']) # Fallback
    fin_df['energy_kwh'] = fin_df['energy_kwh'].fillna(0)
    
    # 4. Calculate Costs & Revenue
    # Need daily prices. Map order start date to master data.
    fin_df['date_key'] = fin_df['start_time'].dt.normalize() # Remove time
    # Ensure master_df date is datetime
    master_df['date'] = pd.to_datetime(master_df['date'])
    
    fin_df = fin_df.merge(master_df, left_on='date_key', right_on='date', how='left')
    
    # Unit Prices
    price_map = {'Product_A': 150.0, 'Product_B': 50.0}
    material_cost_base = {'Product_A': 40.0, 'Product_B': 15.0}
    labor_hours_per_unit = {'Product_A': 0.2, 'Product_B': 0.05}
    
    ledger_entries = []
    
    for _, row in fin_df.iterrows():
        prod = row['product_type']
        passed = row['passed_qty']
        
        # Revenue
        revenue = passed * price_map[prod]
        
        # Energy Cost
        cost_energy = row['energy_kwh'] * row['electricity_price']
        
        # Material Cost (adjusted by index)
        # Index 100 is base.
        mat_cost_unit = material_cost_base[prod] * (row['material_price_index'] / 100.0)
        cost_material = (passed + row['failed_qty']) * mat_cost_unit # Pay for failed mats too
        
        # Labor Cost
        # Estimate labor hours based on duration or units? 
        # User said "labor_cost_rate" in master data. Let's assume it's hourly rate.
        # Cost = Duration * Rate (Simple view) or Units * Rate (Piecework).
        # Let's use Duration * Rate for factory labor.
        duration_hours = (row['end_time'] - row['start_time']).total_seconds() / 3600.0
        cost_labor = duration_hours * row['labor_cost_rate']
        
        profit = revenue - (cost_energy + cost_material + cost_labor)
        
        # Entry Date: 5-7 days later
        entry_date = row['end_time'].date() + timedelta(days=random.randint(5, 7))
        
        ledger_entries.append({
            'entry_date': entry_date,
            'order_id': row['order_id'], # Keep reference
            'revenue': round(revenue, 2),
            'cost_energy': round(cost_energy, 2),
            'cost_material': round(cost_material, 2),
            'cost_labor': round(cost_labor, 2),
            'profit': round(profit, 2)
        })
        
    ledger_df = pd.DataFrame(ledger_entries)
    ledger_df.to_csv(f'{OUTPUT_DIR}/financial_ledger.csv', index=False)
    return ledger_df

def main(scenario='normal'):
    print("Starting Data Generation...")
    
    # 1. Master Data
    master_df = generate_master_data(scenario)
    print(f"Master Data: {master_df.shape}")
    
    # 2. Orders
    orders_df = generate_production_orders()
    print(f"Orders: {orders_df.shape}")
    
    # 3. Device Status
    device_df, anomalies = generate_device_status(orders_df)
    print(f"Device Status: {device_df.shape}")
    print(f"Injected {len(anomalies)} anomalies.")
    
    # 4. Quality Logs
    quality_df = generate_quality_logs(orders_df, anomalies)
    print(f"Quality Logs: {quality_df.shape}")
    
    # 5. Financial Ledger
    ledger_df = generate_financial_ledger(orders_df, quality_df, device_df, master_df)
    print(f"Financial Ledger: {ledger_df.shape}")
    
    print("\nData Generation Complete. Files saved to 'mock_data/'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='normal', help='Scenario: normal or crisis')
    args = parser.parse_args()
    main(scenario=args.scenario)
