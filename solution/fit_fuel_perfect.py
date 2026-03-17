import json
import numpy as np
from scipy.optimize import minimize

def load_races(filename, count=100):
    with open(filename) as f:
        races = json.load(f)[:count]
    return races

def extract_features(strategy, laps_total, track_temp):
    current_tire = strategy['starting_tire']
    pit_stops = sorted(strategy['pit_stops'], key=lambda x: x['lap'])
    
    pidx = 0
    laps_on_tire = 1
    
    sum_is_S = sum_is_M = sum_is_H = 0
    sum_deg_S = sum_deg_M = sum_deg_H = 0
    
    sum_L_is_S = sum_L_is_M = sum_L_is_H = 0
    sum_L_deg_S = sum_L_deg_M = sum_L_deg_H = 0
    
    for L in range(1, laps_total + 1):
        if pidx < len(pit_stops) and pit_stops[pidx]['lap'] == L:
            current_tire = pit_stops[pidx]['to_tire']
            pidx += 1
            laps_on_tire = 1
            
        if current_tire == 'SOFT':
            sum_is_S += 1
            sum_deg_S += laps_on_tire
            sum_L_is_S += L
            sum_L_deg_S += L * laps_on_tire
        elif current_tire == 'MEDIUM':
            sum_is_M += 1
            sum_deg_M += laps_on_tire
            sum_L_is_M += L
            sum_L_deg_M += L * laps_on_tire
        else: # HARD
            sum_is_H += 1
            sum_deg_H += laps_on_tire
            sum_L_is_H += L
            sum_L_deg_H += L * laps_on_tire
            
        laps_on_tire += 1
        
    pits = len(pit_stops)
    
    features = np.array([
        sum_is_M, sum_is_H,
        sum_deg_S, sum_deg_M, sum_deg_H,
        sum_L_is_S, sum_L_is_M, sum_L_is_H,
        sum_L_deg_S, sum_L_deg_M, sum_L_deg_H,
        track_temp * sum_is_S, track_temp * sum_is_M, track_temp * sum_is_H,
        track_temp * sum_deg_S, track_temp * sum_deg_M, track_temp * sum_deg_H
    ], dtype=np.float64)
    
    return features, pits

def solve_regression():
    races = load_races('data/historical_races/races_00000-00999.json', 1000)
    
    X_diffs = []
    C_diffs = []
    
    skipped = 0
    for race in races:
        laps_total = race['race_config']['total_laps']
        track_temp = race['race_config']['track_temp']
        pit_time = race['race_config']['pit_lane_time']
        
        strats = list(race['strategies'].values())
        positions = {d: i for i, d in enumerate(race['finishing_positions'])}
        
        # Sort by finishing position (faster first)
        strats.sort(key=lambda s: positions[s['driver_id']])
        
        features_cache = [extract_features(s, laps_total, track_temp) for s in strats]
        
        for i in range(len(strats)-1):
            s1 = strats[i]     # Faster
            s2 = strats[i+1]   # Slower
            
            f1, p1 = features_cache[i]
            f2, p2 = features_cache[i+1]
            
            if np.allclose(f1, f2) and p1 == p2:
                skipped += 1
                continue
            
            diff_f = f2 - f1
            diff_c = (p2 - p1) * pit_time
            
            X_diffs.append(diff_f)
            C_diffs.append(diff_c)
            
    print(f"Extracted {len(X_diffs)} valid strict constraints. Skipped {skipped}.")
    
    X_mat = np.array(X_diffs)
    C_vec = np.array(C_diffs)
    
    def objective(params):
        diffs = X_mat.dot(params) + C_vec
        penalties = 0.1 - diffs
        penalties = np.maximum(0, penalties)
        loss = np.sum(penalties**2) + 0.0001 * np.sum(params**2)
        return loss
    
    initial_guess = np.zeros(17)
    initial_guess[2:5] = 0.01 # deg base
    
    bounds = [(None, None)] * 17
    
    res = minimize(
        objective, 
        initial_guess, 
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'disp': True}
    )
    
    print("\nOptimization Finished.")
    print("Success:", res.success)
    print("Final Loss:", res.fun)
    
    p = res.x
    names = [
        "Base_M", "Base_H", 
        "Deg_S", "Deg_M", "Deg_H", 
        "Fuel_is_S", "Fuel_is_M", "Fuel_is_H",
        "Fuel_deg_S", "Fuel_deg_M", "Fuel_deg_H",
        "Temp_is_S", "Temp_is_M", "Temp_is_H",
        "Temp_deg_S", "Temp_deg_M", "Temp_deg_H"
    ]
    for n, v in zip(names, p):
        print(f"{n:>12} = {v:10.5f}")
        
    diffs = X_mat.dot(p) + C_vec
    correct = np.sum(diffs > 0)
            
    print(f"Accuracy on strict constraints: {correct}/{len(X_diffs)} ({correct/len(X_diffs)*100:.2f}%)")

if __name__ == '__main__':
    solve_regression()
