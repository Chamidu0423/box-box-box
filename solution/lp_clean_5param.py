import json
import numpy as np
from scipy.optimize import linprog

def load_races(filename, count=1000):
    with open(filename) as f:
        races = json.load(f)[:count]
    return races

def extract_features(strategy, laps_total):
    """
    Extract features that matter for RELATIVE comparison within a race.
    base_lap_time cancels, temperature will test separately.
    
    Parameters:
    - offset_M (relative to S=0)
    - offset_H (relative to S=0)
    - deg_S, deg_M, deg_H
    
    lap_time = base + offset[tire] + deg[tire] * age_on_lap
    
    total_time = N*base + sum_laps(offset[tire]) + sum_laps(deg[tire] * age)
               = N*base + laps_M*offset_M + laps_H*offset_H + sum_age_S*deg_S + sum_age_M*deg_M + sum_age_H*deg_H
    """
    current_tire = strategy['starting_tire']
    pit_stops = sorted(strategy['pit_stops'], key=lambda x: x['lap'])
    
    pidx = 0
    laps_on_tire = 1
    
    laps_S, laps_M, laps_H = 0, 0, 0
    sum_age_S, sum_age_M, sum_age_H = 0, 0, 0
    
    for lap in range(1, laps_total + 1):
        if pidx < len(pit_stops) and pit_stops[pidx]['lap'] == lap:
            current_tire = pit_stops[pidx]['to_tire']
            pidx += 1
            laps_on_tire = 1
            
        if current_tire == 'SOFT':
            laps_S += 1
            sum_age_S += laps_on_tire
        elif current_tire == 'MEDIUM':
            laps_M += 1
            sum_age_M += laps_on_tire
        else: # HARD
            laps_H += 1
            sum_age_H += laps_on_tire
            
        laps_on_tire += 1
        
    # features: [laps_M, laps_H, sum_age_S, sum_age_M, sum_age_H]
    # These correspond to params [offset_M, offset_H, deg_S, deg_M, deg_H]
    features = np.array([laps_M, laps_H, sum_age_S, sum_age_M, sum_age_H], dtype=np.float64)
    
    pits = len(pit_stops)
    return features, pits

def run_lp():
    races = load_races('data/historical_races/races_00000-00999.json', 1000)
    
    A_ub = []
    b_ub = []
    skipped = 0
    total = 0
    
    for race in races:
        laps_total = race['race_config']['total_laps']
        pit_time = race['race_config']['pit_lane_time']
        
        strats = list(race['strategies'].values())
        positions = {d: i for i, d in enumerate(race['finishing_positions'])}
        
        orig_pos = {}
        for p, s in race['strategies'].items():
            orig_pos[s['driver_id']] = int(p.replace('pos', ''))
        
        strats.sort(key=lambda s: positions[s['driver_id']])
        features_cache = [extract_features(s, laps_total) for s in strats]
        
        for i in range(len(strats)-1):
            f1, p1 = features_cache[i]       # Faster driver
            f2, p2 = features_cache[i+1]     # Slower driver
            
            # If the features + pits are identical, times are identical,
            # grid position breaks the tie. SKIP these!
            if np.array_equal(f1, f2) and p1 == p2:
                skipped += 1
                continue
            
            total += 1
            
            # We need: T1 < T2
            # i.e., (f1 - f2) · params + (p1 - p2) * pit_time < 0
            # But LP is: A·x ≤ b
            # So: (f1 - f2) · params ≤ (p2 - p1) * pit_time - epsilon
            
            diff = f1 - f2
            b_val = (p2 - p1) * pit_time - 0.001  # epsilon margin
            
            A_ub.append(diff)
            b_ub.append(b_val)
    
    print(f"Built system: {len(A_ub)} constraints, {skipped} skipped (exact ties).")
    
    # LP with 5 params: [offset_M, offset_H, deg_S, deg_M, deg_H]
    # Minimize 0 (just feasibility check)
    c = np.zeros(5)
    
    # Constraints: deg must be positive
    bounds = [
        (None, None),  # offset_M
        (None, None),  # offset_H
        (0.0001, None),  # deg_S >= 0
        (0.0001, None),  # deg_M >= 0
        (0.0001, None),  # deg_H >= 0
    ]
    
    print("Running LP solver...")
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    print(f"\nResult: {'FEASIBLE ✓' if res.success else 'INFEASIBLE ✗'}")
    
    if res.success:
        p = res.x
        print(f"\nFound exact parameters:")
        print(f"  offset_M  = {p[0]:+.6f}  (M is {'faster' if p[0]<0 else 'slower'} than S)")
        print(f"  offset_H  = {p[1]:+.6f}  (H is {'faster' if p[1]<0 else 'slower'} than S)")
        print(f"  deg_S     =  {p[2]:.6f}")
        print(f"  deg_M     =  {p[3]:.6f}")
        print(f"  deg_H     =  {p[4]:.6f}")
        
        # Check accuracy
        A = np.array(A_ub)
        b = np.array(b_ub)
        satisfied = np.sum(A.dot(p) <= b)
        print(f"\n  Constraints satisfied: {satisfied}/{len(A_ub)} ({satisfied/len(A_ub)*100:.1f}%)")
    else:
        print("\nNo exact solution found. Formula needs more parameters!")
        print("(i.e. temperature or non-linear term is essential)")
        print(f"Message: {res.message}")

if __name__ == '__main__':
    run_lp()
