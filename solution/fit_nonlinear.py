import json
import numpy as np
from scipy.optimize import minimize

def load_races(filename, count=100):
    with open(filename) as f:
        races = json.load(f)[:count]
    return races

def lap_time_sum(strategy, laps_total, track_temp, params):
    base_s, base_m, base_h, deg_s, deg_m, deg_h, temp_mult, deg_exp = params
    
    total = 0.0
    current_tire = strategy['starting_tire']
    pit_stops = sorted(strategy['pit_stops'], key=lambda x: x['lap'])
    
    pidx = 0
    laps_on_tire = 0
    
    for lap in range(1, laps_total + 1):
        if pidx < len(pit_stops) and pit_stops[pidx]['lap'] == lap:
            current_tire = pit_stops[pidx]['to_tire']
            pidx += 1
            laps_on_tire = 0
            
        # The equation for tire degradation might actually be:
        # lap_time = Base_Tire + (Laps_on_Tire * Deg_Rate * Temp_Multiplier)
        # OR Temp_effect is standalone
        
        # Let's try quadratic degradation:
        # eff = base + deg * laps + deg2 * laps^2 + temp_factor
        if current_tire == 'SOFT':
            tire_eff = base_s + deg_s * laps_on_tire + deg_exp * (laps_on_tire ** 2)
        elif current_tire == 'MEDIUM':
            tire_eff = base_m + deg_m * laps_on_tire + deg_exp * (laps_on_tire ** 2)
        else: # HARD
            tire_eff = base_h + deg_h * laps_on_tire + deg_exp * (laps_on_tire ** 2)
            
        temp_eff = track_temp * temp_mult
        
        total += tire_eff + temp_eff
        laps_on_tire += 1
        
    return total

def objective(params, X_data):
    loss = 0.0
    for strat1, strat2, laps_total, track_temp in X_data:
        t1 = lap_time_sum(strat1, laps_total, track_temp, params)
        t2 = lap_time_sum(strat2, laps_total, track_temp, params)
        
        diff = t2 - t1
        if diff < 1.0:
            loss += (1.0 - diff) ** 2
            
    loss += 0.001 * np.sum(np.array(params)**2)
    return loss

def find_nonlinear_formula():
    races = load_races('data/historical_races/races_00000-00999.json', 1000)
    
    X_train = []
    
    for race in races:
        total_laps = race['race_config']['total_laps']
        track_temp = race['race_config']['track_temp']
        
        strats = list(race['strategies'].values())
        positions = {d: i for i, d in enumerate(race['finishing_positions'])}
        
        strats.sort(key=lambda s: positions[s['driver_id']])
        
        for i in range(len(strats)-1):
            s1 = strats[i]     # Faster
            s2 = strats[i+1]   # Slower
            X_train.append((s1, s2, total_laps, track_temp))
            
    print(f"Extracted {len(X_train)} pairwise comparisons. Starting optimization...")
    
    # params: base_s, base_m, base_h, deg_s, deg_m, deg_h, temp_mult, deg2(shared quadratic coeff)
    initial_guess = [0.0, 1.0, 2.0, 0.1, 0.05, 0.02, 0.0, 0.001]
    
    bounds = [
        (None, None), (None, None), (None, None),
        (0.0001, 2.0), (0.0001, 2.0), (0.0001, 2.0),
        (-1.0, 1.0),
        (-0.5, 0.5) 
    ]
    
    res = minimize(
        objective, 
        initial_guess, 
        args=(X_train,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'disp': False}
    )
    
    print("\nOptimization Finished.")
    print("Success:", res.success)
    p = res.x
    print(f"SOFT_BASE   = {p[0]:.4f}")
    print(f"MEDIUM_BASE = {p[1]:.4f}")
    print(f"HARD_BASE   = {p[2]:.4f}")
    print(f"SOFT_DEG_1  = {p[3]:.4f}")
    print(f"MEDIUM_DEG_1= {p[4]:.4f}")
    print(f"HARD_DEG_1  = {p[5]:.4f}")
    print(f"TEMP_MULT   = {p[6]:.4f}")
    print(f"QUAD_COEFF  = {p[7]:.4f}")
    
    correct = 0
    for s1, s2, laps, temp in X_train:
        t1 = lap_time_sum(s1, laps, temp, p)
        t2 = lap_time_sum(s2, laps, temp, p)
        if t1 < t2:
            correct += 1
            
    print(f"Accuracy on pairwise constraints: {correct}/{len(X_train)} ({correct/len(X_train)*100:.2f}%)")

if __name__ == '__main__':
    find_nonlinear_formula()
