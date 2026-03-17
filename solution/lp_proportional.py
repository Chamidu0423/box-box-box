import json
import numpy as np
from scipy.optimize import linprog

def load_races(filename, count=1000):
    with open(filename) as f:
        races = json.load(f)[:count]
    return races

def extract_features(strategy, laps_total, track_temp, base_lap_time):
    """
    Key insight: compound offsets may be PROPORTIONAL to base_lap_time.
    
    Formula: lap_time = base_lap_time * (1 + compound_factor[tire]) + deg[tire] * age * (1 + temp_factor * temp)
    
    Which means:
    compound cost = base_lap_time * compound_factor[tire] * laps_on_tire
    
    Parameters:
    - factor_M, factor_H  (compound factors relative to S=0)
    - deg_S, deg_M, deg_H  (absolute degradation per lap)
    - temp_deg_S, temp_deg_M, temp_deg_H  (temperature × deg interaction)
    
    Features:
    - base * laps_M, base * laps_H  (for factor_M, factor_H)
    - sum_age_S, sum_age_M, sum_age_H  (for deg_S, deg_M, deg_H)
    - temp * sum_age_S, temp * sum_age_M, temp * sum_age_H  (for temp_deg_*)
    """
    current_tire = strategy['starting_tire']
    pit_stops = sorted(strategy['pit_stops'], key=lambda x: x['lap'])
    
    pidx = 0
    laps_on_tire = 1
    
    laps_S = laps_M = laps_H = 0
    sum_age_S = sum_age_M = sum_age_H = 0
    
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
        else:
            laps_H += 1
            sum_age_H += laps_on_tire
            
        laps_on_tire += 1
        
    pits = len(pit_stops)
    
    # KEY: multiply compound counts by base_lap_time
    features = np.array([
        base_lap_time * laps_M,             # factor_M
        base_lap_time * laps_H,             # factor_H
        sum_age_S,                          # deg_S
        sum_age_M,                          # deg_M
        sum_age_H,                          # deg_H
        track_temp * sum_age_S,             # temp_deg_S
        track_temp * sum_age_M,             # temp_deg_M
        track_temp * sum_age_H,             # temp_deg_H
    ], dtype=np.float64)
    
    return features, pits

def run_lp(num_races=1000):
    races = load_races('data/historical_races/races_00000-00999.json', num_races)
    
    A_ub = []
    b_ub = []
    skipped = 0
    
    for race in races:
        laps_total = race['race_config']['total_laps']
        pit_time = race['race_config']['pit_lane_time']
        track_temp = race['race_config']['track_temp']
        base_lap_time = race['race_config']['base_lap_time']
        
        strats = list(race['strategies'].values())
        positions = {d: i for i, d in enumerate(race['finishing_positions'])}
        
        strats.sort(key=lambda s: positions[s['driver_id']])
        features_cache = [extract_features(s, laps_total, track_temp, base_lap_time) for s in strats]
        
        for i in range(len(strats)-1):
            f1, p1 = features_cache[i]
            f2, p2 = features_cache[i+1]
            
            if np.allclose(f1, f2) and p1 == p2:
                skipped += 1
                continue
            
            diff = f1 - f2
            b_val = (p2 - p1) * pit_time - 0.001
            
            A_ub.append(diff)
            b_ub.append(b_val)
    
    print(f"Built system: {len(A_ub)} constraints, {skipped} skipped.")
    
    c = np.zeros(8)
    bounds = [
        (None, None),    # factor_M (compound proportional offset)
        (None, None),    # factor_H
        (0.0001, None),  # deg_S
        (0.0001, None),  # deg_M
        (0.0001, None),  # deg_H
        (None, None),    # temp_deg_S
        (None, None),    # temp_deg_M
        (None, None),    # temp_deg_H
    ]
    
    print("Running LP solver (8 params with base_lap_time proportional offsets)...")
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    print(f"\nResult: {'FEASIBLE ✓' if res.success else 'INFEASIBLE ✗'}")
    
    if res.success:
        p = res.x
        names = ["factor_M", "factor_H", "deg_S", "deg_M", "deg_H", "tdeg_S", "tdeg_M", "tdeg_H"]
        print("\n🎉 Exact parameters found!")
        for n, v in zip(names, p):
            print(f"  {n:>10} = {v:+.6f}")
            
        A = np.array(A_ub)
        b = np.array(b_ub)
        satisfied = np.sum(A.dot(p) <= b)
        print(f"\n  Constraints satisfied: {satisfied}/{len(A_ub)} ({satisfied/len(A_ub)*100:.1f}%)")
        
        print(f"\n  Formula:")
        print(f"  lap_time = base_lap_time + ({p[0]:+.6f})*base*[M laps] + ({p[1]:+.6f})*base*[H laps]")
        print(f"           + ({p[2]:+.6f} + {p[5]:+.6f}*temp) * age_S")
        print(f"           + ({p[3]:+.6f} + {p[6]:+.6f}*temp) * age_M")
        print(f"           + ({p[4]:+.6f} + {p[7]:+.6f}*temp) * age_H")
    else:
        print(f"\n  {res.message}")
        print("  → May need non-linear terms (quadratic age) or base also affects deg")

if __name__ == '__main__':
    run_lp()
