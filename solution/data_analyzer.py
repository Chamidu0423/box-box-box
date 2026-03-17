import json
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

# Path to the first 1000 races file
DATA_FILE = "data/historical_races/races_00000-00999.json"

def load_data():
    with open(DATA_FILE, 'r') as f:
        races = json.load(f)
    print(f"Loaded {len(races)} races.")
    return races

def extract_features(races):
    """
    Extracts relevant features from the raw JSON data to calculate total race times.
    """
    data_records = []
    
    for race in races:
        race_id = race['race_id']
        config = race['race_config']
        strategies = race['strategies']
        finishing_positions = race['finishing_positions']
        
        # Determine actual finishing position (1 to 20) for each driver
        driver_positions = {driver_id: idx + 1 for idx, driver_id in enumerate(finishing_positions)}
        
        for pos_key, strategy in strategies.items():
            driver_id = strategy['driver_id']
            
            # Record basic driver race info
            record = {
                'race_id': race_id,
                'track': config['track'],
                'total_laps': config['total_laps'],
                'base_lap_time': config['base_lap_time'],
                'pit_lane_time': config['pit_lane_time'],
                'track_temp': config['track_temp'],
                'driver_id': driver_id,
                'finishing_position': driver_positions[driver_id],
                'starting_tire': strategy['starting_tire'],
                'pit_stops': strategy['pit_stops']
            }
            data_records.append(record)
            
    return pd.DataFrame(data_records)

def simulate_race_time(params, df_driver):
    """
    Simulates the total race time for a single driver given an assumed set of parameters.
    
    params structure (example):
    [0] = SOFT compound base speed differential (e.g. -1.0 means 1 second faster than base)
    [1] = MEDIUM compound base speed differential
    [2] = HARD compound base speed differential
    [3] = SOFT degradation factor (base)
    [4] = MEDIUM degradation factor (base)
    [5] = HARD degradation factor (base)
    [6] = Temperature degradation multiplier
    """
    soft_diff, med_diff, hard_diff, soft_deg, med_deg, hard_deg, temp_mult = params
    
    total_time = 0.0
    current_tire = df_driver['starting_tire']
    pit_stops = sorted(df_driver['pit_stops'], key=lambda x: x['lap'])
    
    current_lap = 1
    tire_age = 0
    
    pidx = 0
    num_pit_stops = len(pit_stops)
    
    total_laps = df_driver['total_laps']
    base_lap_time = df_driver['base_lap_time']
    track_temp = df_driver['track_temp']
    pit_pen = df_driver['pit_lane_time']
    
def simulate_race_time_vectorized(params, df):
    # vectorized version for performance
    
    soft_diff, med_diff, hard_diff, soft_deg, med_deg, hard_deg, temp_mult = params
    
    # We will simulate the race for each row (driver in a race)
    # But for a simpler heuristic first: 
    # Let's write a function that calculates total time for a given row
    total_times = []
    
    for _, row in df.iterrows():
        total_time = 0.0
        current_tire = row['starting_tire']
        pit_stops = sorted(row['pit_stops'], key=lambda x: x['lap'])
        
        tire_age = 0
        pidx = 0
        num_pit_stops = len(pit_stops)
        
        base_lap_time = row['base_lap_time']
        track_temp = row['track_temp']
        pit_pen = row['pit_lane_time']
        
        for lap in range(1, row['total_laps'] + 1):
            tire_age += 1
            
            if current_tire == 'SOFT':
                diff, deg = soft_diff, soft_deg
            elif current_tire == 'MEDIUM':
                diff, deg = med_diff, med_deg
            else:
                diff, deg = hard_diff, hard_deg
                
            actual_deg = deg * (1 + temp_mult * (track_temp - 20))
            lap_time = base_lap_time + diff + (actual_deg * tire_age)
            total_time += lap_time
            
            if pidx < num_pit_stops and pit_stops[pidx]['lap'] == lap:
                total_time += pit_pen
                current_tire = pit_stops[pidx]['to_tire']
                tire_age = 0
                pidx += 1
                
        total_times.append(total_time)
        
    return np.array(total_times)

def prepare_regression_data(df):
    # We want to form an equation:
    # Total Time = BaseTimeLaps + sum(tire_diff * laps) + sum(tire_deg * sum_{lap}(1+temp_mult*(T-20))*lap) + pit_stops * pit_pen
    # Since we only know finishing position and not total time, we can approximate that
    # all cars finish roughly at the "average" race time, or we can use the position to 
    # estimate a tiny time delta (e.g. 1st place is 0s, 20th place is +40s).
    #
    # Actually, a better approach: let's use pairwise differences between drivers in the SAME race.
    # Time(driver A) < Time(driver B) if pos(A) < pos(B).
    # Time(B) - Time(A) ~ positive delta.
    
    # Features for each driver:
    # 0: SOFT laps
    # 1: MEDIUM laps
    # 2: HARD laps
    # 3: SOFT degradation sum
    # 4: MEDIUM degradation sum
    # 5: HARD degradation sum
    # 6: Temperature degradation sum (SOFT)
    # 7: Temperature degradation sum (MEDIUM)
    # 8: Temperature degradation sum (HARD)
    
    # We will compute these coefficients for every driver.
    
    X_rows = []
    
    for _, row in df.iterrows():
        current_tire = row['starting_tire']
        pit_stops = sorted(row['pit_stops'], key=lambda x: x['lap'])
        
        tire_age = 0
        pidx = 0
        num_pit_stops = len(pit_stops)
        
        track_temp = row['track_temp']
        t_factor = (track_temp - 20)
        
        features = {'SOFT_laps': 0, 'MEDIUM_laps': 0, 'HARD_laps': 0,
                    'SOFT_deg': 0, 'MEDIUM_deg': 0, 'HARD_deg': 0,
                    'SOFT_temp_deg': 0, 'MEDIUM_temp_deg': 0, 'HARD_temp_deg': 0}
        
        for lap in range(1, row['total_laps'] + 1):
            tire_age += 1
            
            tire_key = f"{current_tire}_laps"
            deg_key = f"{current_tire}_deg"
            temp_key = f"{current_tire}_temp_deg"
            
            features[tire_key] += 1
            features[deg_key] += tire_age
            features[temp_key] += tire_age * t_factor
            
            if pidx < num_pit_stops and pit_stops[pidx]['lap'] == lap:
                current_tire = pit_stops[pidx]['to_tire']
                tire_age = 0
                pidx += 1
                
        # Total pit penalty is known
        features['pit_penalty_total'] = num_pit_stops * row['pit_lane_time']
        features['base_time_total'] = row['total_laps'] * row['base_lap_time']
        
        X_rows.append(features)
        
    return pd.DataFrame(X_rows)

def extract_exact_parameters(df, X_df):
    import numpy as np
    from scipy.optimize import linprog
    
    print("Extracting exact parameters using Linear Programming feasibility...")
    
    # We want to find x = [soft_diff, med_diff, hard_diff, soft_deg, med_deg, hard_deg]
    # such that for all i < j in a race:
    # t_i < t_j  ==>  t_i - t_j <= -epsilon
    epsilon = 0.001
    
    # Pre-extract base times and features
    df['total_pit_time'] = df['pit_stops'].apply(len) * df['pit_lane_time']
    df['base_time_calc'] = df['total_laps'] * df['base_lap_time'] + df['total_pit_time']
    
    X_df['race_id'] = df['race_id'].values
    X_df['pos'] = df['finishing_position'].values
    X_df['base_time'] = df['base_time_calc'].values
    
    best_params = None
    best_acc = -1
    
    # We will test a grid of temp_mult since it's the only non-linear interaction
    temp_mults = np.linspace(0.000, 0.010, 101) # e.g. 0.000, 0.0001, 0.0002...
    
    # Prepare difference vectors (A_ub * x <= b_ub)
    base_diffs = []
    f_diffs_base = [] # [SL, ML, HL, SD, MD, HD, STD, MTD, HTD] diffs
    
    for race_id, group in X_df.groupby('race_id'):
        group = group.sort_values('pos')
        idx = group.index.tolist()
        
        bt = group['base_time'].values
        sl = group['SOFT_laps'].values
        ml = group['MEDIUM_laps'].values
        hl = group['HARD_laps'].values
        sd = group['SOFT_deg'].values
        md = group['MEDIUM_deg'].values
        hd = group['HARD_deg'].values
        std = group['SOFT_temp_deg'].values
        mtd = group['MEDIUM_temp_deg'].values
        htd = group['HARD_temp_deg'].values
        
        n = len(idx)
        for i in range(n):
            for j in range(i+1, n):
                # i finished before j, so t_i < t_j
                # t_i - t_j <= -epsilon
                # (f_i - f_j)*x <= b_j - b_i - epsilon
                
                b_diff = bt[j] - bt[i] - epsilon
                base_diffs.append(b_diff)
                
                f_vec = [
                    sl[i] - sl[j],
                    ml[i] - ml[j],
                    hl[i] - hl[j],
                    sd[i] - sd[j],
                    md[i] - md[j],
                    hd[i] - hd[j],
                    std[i] - std[j],
                    mtd[i] - mtd[j],
                    htd[i] - htd[j]
                ]
                f_diffs_base.append(f_vec)
                
    base_diffs = np.array(base_diffs)
    f_diffs_base = np.array(f_diffs_base)
    
    print(f"Total pairwise constraints: {len(base_diffs)}")
    
    # Bounds for the 6 linear parameters
    bounds = [
        (-3.0, 0.0),   # soft_diff
        (0.0, 0.0),    # med_diff (fixed to 0 as baseline usually)
        (0.0, 3.0),    # hard_diff
        (0.1, 0.5),    # soft_deg
        (0.05, 0.3),   # med_deg
        (0.01, 0.15)   # hard_deg
    ]
    
    count = 0
    for tm in temp_mults:
        count += 1
        
        # A_ub * x <= b_ub
        # x = [soft_diff, med_diff, hard_diff, soft_deg, med_deg, hard_deg]
        # feature 3 = SD + tm * STD
        A_ub = np.zeros((len(base_diffs), 6))
        A_ub[:, 0] = f_diffs_base[:, 0] # SL
        A_ub[:, 1] = f_diffs_base[:, 1] # ML
        A_ub[:, 2] = f_diffs_base[:, 2] # HL
        
        A_ub[:, 3] = f_diffs_base[:, 3] + tm * f_diffs_base[:, 6] # SD + tm*STD
        A_ub[:, 4] = f_diffs_base[:, 4] + tm * f_diffs_base[:, 7] # MD + tm*MTD
        A_ub[:, 5] = f_diffs_base[:, 5] + tm * f_diffs_base[:, 8] # HD + tm*HTD
        
        b_ub = base_diffs
        
        # We just want a feasible point, so objective is 0
        c = np.zeros(6)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            print(f"\nFound exact feasible solution at Temp_Mult = {tm:.4f}!")
            ax = res.x
            params = np.array([ax[0], ax[1], ax[2], ax[3], ax[4], ax[5], tm])
            return params
            
        elif count % 10 == 0:
            print(f"Checked {count}/{len(temp_mults)} temp mults...")
            
    print("Could not find a perfectly feasible linear solution. Data might have contradictions or formula is slightly different.")
    # Fallback to SGD if LP fails
    return None

def main():
    print("Loading historical data...")
    races = load_data()
    
    print("Extracting features into DataFrame...")
    df = extract_features(races)
    
    # For training, take a smaller subset of races to make it fast
    race_ids = df['race_id'].unique()[:100] # 100 races = 19,000 pairs
    train_df = df[df['race_id'].isin(race_ids)].copy()
    
    print(f"Training on {len(race_ids)} races ({len(train_df)} rows)...")

    print("Preparing feature matrices...")
    X_df = prepare_regression_data(train_df)
    
    params = extract_exact_parameters(train_df, X_df)
    
    if params is None:
        print("Failed to solve exact parameters.")
        return
        
    print("\n--- FINAL PARAMETERS TO USE IN SIMULATOR ---")
    print("\n--- FINAL PARAMETERS TO USE IN SIMULATOR ---")
    print(f"SOFT_BASE_DIFF = {params[0]:.4f}")
    print(f"MEDIUM_BASE_DIFF = {params[1]:.4f}")
    print(f"HARD_BASE_DIFF = {params[2]:.4f}")
    print(f"SOFT_DEG = {params[3]:.4f}")
    print(f"MEDIUM_DEG = {params[4]:.4f}")
    print(f"HARD_DEG = {params[5]:.4f}")
    print(f"TEMP_MULT = {params[6]:.5f}")
    
    # Evaluate accuracy
    train_df['predicted_time'] = simulate_race_time_vectorized(params, train_df)
    correct_races = 0
    for race_id, group in train_df.groupby('race_id'):
        actual_order = group.sort_values('finishing_position')['driver_id'].tolist()
        predicted_order = group.sort_values('predicted_time')['driver_id'].tolist()
        if actual_order == predicted_order:
            correct_races += 1
            
    print(f"Training Accuracy (Perfect Prediction): {correct_races}/{len(race_ids)} ({correct_races/len(race_ids)*100:.1f}%)")

if __name__ == "__main__":
    main()
