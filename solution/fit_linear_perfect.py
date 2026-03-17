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
    
    laps_S, laps_M, laps_H = 0, 0, 0
    sum_S, sum_M, sum_H = 0, 0, 0
    
    for lap in range(1, laps_total + 1):
        if pidx < len(pit_stops) and pit_stops[pidx]['lap'] == lap:
            current_tire = pit_stops[pidx]['to_tire']
            pidx += 1
            laps_on_tire = 1
            
        if current_tire == 'SOFT':
            laps_S += 1
            sum_S += laps_on_tire
        elif current_tire == 'MEDIUM':
            laps_M += 1
            sum_M += laps_on_tire
        else: # HARD
            laps_H += 1
            sum_H += laps_on_tire
            
        laps_on_tire += 1
        
    pits = len(pit_stops)
        
    features = np.array([
        laps_M, laps_H,
        sum_S, sum_M, sum_H,
        track_temp * sum_S, track_temp * sum_M, track_temp * sum_H,
        track_temp * laps_S, track_temp * laps_M, track_temp * laps_H
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
            
            # If features and pit counts are exactly equal, they tie mathematically.
            # Grid positions broke the tie. Skip them!
            if np.allclose(f1, f2) and p1 == p2:
                skipped += 1
                continue
            
            diff_f = f2 - f1
            diff_c = (p2 - p1) * pit_time
            
            X_diffs.append(diff_f)
            C_diffs.append(diff_c)
            
    print(f"Extracted {len(X_diffs)} valid strict constraints. Skipped {skipped} purely grid tie-breakers.")
    
    X_mat = np.array(X_diffs)
    C_vec = np.array(C_diffs)
    
    # Vectorized objective function
    def objective(params):
        # Calculate all differences at once
        # diff = t2 - t1
        diffs = X_mat.dot(params) + C_vec
        
        # Penalize if diff < 0.1
        # Which means t2 is faster or barely slower than t1
        penalties = 0.1 - diffs
        # Only keep positive penalties
        penalties = np.maximum(0, penalties)
        
        loss = np.sum(penalties**2) + 0.0001 * np.sum(params**2)
        return loss

    # Initial guess
    initial_guess = np.array([0.0, 0.0, 0.1, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    bounds = [
        (None, None), (None, None), # bases
        (0.0001, None), (0.0001, None), (0.0001, None), # degs
        (None, None), (None, None), (None, None), # deg temp interaction
        (None, None), (None, None), (None, None)  # base temp interaction
    ]
    
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
    names = ["M_base", "H_base", "S_deg", "M_deg", "H_deg", "S_td", "M_td", "H_td", "S_tb", "M_tb", "H_tb"]
    for n, v in zip(names, p):
        print(f"{n:>8} = {v:10.5f}")
        
    diffs = X_mat.dot(p) + C_vec
    correct = np.sum(diffs > 0)
            
    print(f"Accuracy on strict constraints: {correct}/{len(X_diffs)} ({correct/len(X_diffs)*100:.2f}%)")

if __name__ == '__main__':
    solve_regression()
