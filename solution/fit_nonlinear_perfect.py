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
    
    # We need arrays of lap ages to apply exponents fast
    ages_S = []
    ages_M = []
    ages_H = []
    
    for lap in range(1, laps_total + 1):
        if pidx < len(pit_stops) and pit_stops[pidx]['lap'] == lap:
            current_tire = pit_stops[pidx]['to_tire']
            pidx += 1
            laps_on_tire = 1
            
        if current_tire == 'SOFT':
            laps_S += 1
            ages_S.append(laps_on_tire)
        elif current_tire == 'MEDIUM':
            laps_M += 1
            ages_M.append(laps_on_tire)
        else: # HARD
            laps_H += 1
            ages_H.append(laps_on_tire)
            
        laps_on_tire += 1
        
    pits = len(pit_stops)
    return laps_S, laps_M, laps_H, np.array(ages_S), np.array(ages_M), np.array(ages_H), pits

def solve_regression():
    races = load_races('data/historical_races/races_00000-00999.json', 1000)
    
    X_train = []
    
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
            
            f1 = features_cache[i]
            f2 = features_cache[i+1]
            
            # If features are exactly equal, they tie mathematically. Grid broke tie.
            if f1[0] == f2[0] and f1[1] == f2[1] and f1[2] == f2[2] and f1[6] == f2[6]:
                # Exact same lap counts and pit counts
                if np.array_equal(f1[3], f2[3]) and np.array_equal(f1[4], f2[4]) and np.array_equal(f1[5], f2[5]):
                    skipped += 1
                    continue
            
            X_train.append((f1, f2, pit_time))
            
    print(f"Extracted {len(X_train)} strict comparisons. Skipped {skipped}.")
    
    def objective(params):
        base_m, base_h, deg_s, deg_m, deg_h, exp_s, exp_m, exp_h = params
        base_s = 0.0
        
        loss = 0.0
        
        for f1, f2, pit_time in X_train:
            t1 = (f1[0] * base_s + f1[1] * base_m + f1[2] * base_h + 
                  deg_s * np.sum(f1[3] ** exp_s) + 
                  deg_m * np.sum(f1[4] ** exp_m) + 
                  deg_h * np.sum(f1[5] ** exp_h) +
                  f1[6] * pit_time)
                  
            t2 = (f2[0] * base_s + f2[1] * base_m + f2[2] * base_h + 
                  deg_s * np.sum(f2[3] ** exp_s) + 
                  deg_m * np.sum(f2[4] ** exp_m) + 
                  deg_h * np.sum(f2[5] ** exp_h) +
                  f2[6] * pit_time)
                  
            diff = t2 - t1
            if diff < 0.1:
                loss += (0.1 - diff) ** 2
                
        loss += 0.0001 * np.sum(np.array(params)**2)
        return loss

    initial_guess = np.array([0.0, 0.0, 0.1, 0.05, 0.02, 1.0, 1.0, 1.0])
    
    bounds = [
        (None, None), (None, None),
        (0.0001, None), (0.0001, None), (0.0001, None),
        (0.5, 3.0), (0.5, 3.0), (0.5, 3.0)
    ]
    
    # Still too slow? Not if we vectorize
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
    names = ["M_base", "H_base", "S_deg", "M_deg", "H_deg", "S_exp", "M_exp", "H_exp"]
    for n, v in zip(names, p):
        print(f"{n:>8} = {v:10.5f}")

if __name__ == '__main__':
    solve_regression()
