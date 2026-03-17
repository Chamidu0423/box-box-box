import json
import numpy as np
from scipy.optimize import linprog

def load_races(filename, count=1000):
    with open(filename) as f:
        races = json.load(f)[:count]
    return races

def extract_features(strategy, laps_total, track_temp):
    """
    Full set of features:
    Params: [offset_M, offset_H, deg_S, deg_M, deg_H, 
             temp_int_S, temp_int_M, temp_int_H,  # temp × deg × age
             tbase_S, tbase_M, tbase_H]            # temp × laps (base temp effect per tire)
    
    lap_time = base + offset[T] + tbase[T]*temp + (deg[T] + temp_int[T]*temp) * age
    
    total = N*base + 
            laps_M*offset_M + laps_H*offset_H +
            tbase_S*temp*laps_S + tbase_M*temp*laps_M + tbase_H*temp*laps_H +
            deg_S*sum_age_S + deg_M*sum_age_M + deg_H*sum_age_H +
            temp_int_S*temp*sum_age_S + temp_int_M*temp*sum_age_M + temp_int_H*temp*sum_age_H
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
    
    features = np.array([
        laps_M, laps_H,
        sum_age_S, sum_age_M, sum_age_H,
        track_temp * sum_age_S, track_temp * sum_age_M, track_temp * sum_age_H,
        track_temp * laps_S, track_temp * laps_M, track_temp * laps_H
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
        
        strats = list(race['strategies'].values())
        positions = {d: i for i, d in enumerate(race['finishing_positions'])}
        
        strats.sort(key=lambda s: positions[s['driver_id']])
        features_cache = [extract_features(s, laps_total, track_temp) for s in strats]
        
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
    
    c = np.zeros(11)
    bounds = [
        (None, None),     # offset_M
        (None, None),     # offset_H
        (0.0001, None),   # deg_S
        (0.0001, None),   # deg_M
        (0.0001, None),   # deg_H
        (None, None),     # temp×age _S
        (None, None),     # temp×age _M
        (None, None),     # temp×age _H
        (None, None),     # temp×laps_S
        (None, None),     # temp×laps_M
        (None, None),     # temp×laps_H
    ]
    
    print("Running LP solver (11 params)...")
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    print(f"\nResult: {'FEASIBLE ✓' if res.success else 'INFEASIBLE ✗'}")
    
    if res.success:
        p = res.x
        names = ["offset_M", "offset_H", "deg_S", "deg_M", "deg_H", 
                 "tdeg_S", "tdeg_M", "tdeg_H", "tbase_S", "tbase_M", "tbase_H"]
        print("\nExact parameters found!")
        for n, v in zip(names, p):
            print(f"  {n:>10} = {v:+.6f}")
        
        A = np.array(A_ub)
        b = np.array(b_ub)
        satisfied = np.sum(A.dot(p) <= b)
        print(f"\n  Constraints satisfied: {satisfied}/{len(A_ub)} ({satisfied/len(A_ub)*100:.1f}%)")
        
        print(f"\n  Formula discovered:")
        print(f"  lap_time = base_lap_time")
        print(f"           + ({p[0]:+.4f}) * [M laps]  + ({p[1]:+.4f}) * [H laps]")
        print(f"           + ({p[2]:+.4f} + {p[5]:+.4f}*temp) * age_S")
        print(f"           + ({p[3]:+.4f} + {p[6]:+.4f}*temp) * age_M")
        print(f"           + ({p[4]:+.4f} + {p[7]:+.4f}*temp) * age_H")
        print(f"           + ({p[8]:+.4f}*temp) * [S laps]")
        print(f"           + ({p[9]:+.4f}*temp) * [M laps]")
        print(f"           + ({p[10]:+.4f}*temp) * [H laps]")
    else:
        print("\n  Still infeasible! The formula must have a non-linear degradation curve.")
        print(f"  Message: {res.message}")

if __name__ == '__main__':
    run_lp()
