import json
import numpy as np
from scipy.optimize import linprog

def load_races(filename, count=1000):
    with open(filename) as f:
        races = json.load(f)[:count]
    return races

def extract_features(strategy, laps_total, track_temp, base_lap_time):
    """
    Extended LP with quadratic age terms:
    lap_time = base + factor[T]*base + deg1[T]*age + deg2[T]*age^2 + tdeg1[T]*temp*age
    
    Params: factor_M, factor_H, deg1_S, deg1_M, deg1_H, deg2_S, deg2_M, deg2_H, tdeg1_S, tdeg1_M, tdeg1_H
    """
    current_tire = strategy['starting_tire']
    pit_stops = sorted(strategy['pit_stops'], key=lambda x: x['lap'])
    
    pidx = 0
    laps_on_tire = 1
    
    laps_M = laps_H = 0
    sum1_S = sum1_M = sum1_H = 0   # Σ age
    sum2_S = sum2_M = sum2_H = 0   # Σ age²
    
    for lap in range(1, laps_total + 1):
        if pidx < len(pit_stops) and pit_stops[pidx]['lap'] == lap:
            current_tire = pit_stops[pidx]['to_tire']
            pidx += 1
            laps_on_tire = 1
            
        if current_tire == 'SOFT':
            sum1_S += laps_on_tire
            sum2_S += laps_on_tire ** 2
        elif current_tire == 'MEDIUM':
            laps_M += 1
            sum1_M += laps_on_tire
            sum2_M += laps_on_tire ** 2
        else:
            laps_H += 1
            sum1_H += laps_on_tire
            sum2_H += laps_on_tire ** 2
            
        laps_on_tire += 1
        
    pits = len(pit_stops)
    
    features = np.array([
        base_lap_time * laps_M,   # factor_M
        base_lap_time * laps_H,   # factor_H
        sum1_S, sum1_M, sum1_H,   # deg1 * age
        sum2_S, sum2_M, sum2_H,   # deg2 * age²
        track_temp * sum1_S, track_temp * sum1_M, track_temp * sum1_H,  # tdeg * temp * age
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
    
    c = np.zeros(11)
    bounds = [
        (None, None), (None, None),   # factor_M, factor_H
        (0.0001, None), (0.0001, None), (0.0001, None),  # deg1
        (0, None),    (0, None),     (0, None),           # deg2 >= 0 (quadratic)
        (None, None), (None, None), (None, None),         # temp_deg
    ]
    
    print("Running LP (11 params: proportional offsets + quadratic deg + temp)...")
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    print(f"\nResult: {'FEASIBLE ✓' if res.success else 'INFEASIBLE ✗'}")
    
    if res.success:
        p = res.x
        names = ["factor_M", "factor_H", "deg1_S", "deg1_M", "deg1_H", "deg2_S", "deg2_M", "deg2_H", "tdeg_S", "tdeg_M", "tdeg_H"]
        print("\n🎉 EXACT FORMULA FOUND!")
        for n, v in zip(names, p):
            print(f"  {n:>10} = {v:+.8f}")
            
        print(f"\n  lap_time = base_lap_time")
        print(f"           + ({p[0]:+.6f}) * base * [M laps]  + ({p[1]:+.6f}) * base * [H laps]")
        print(f"           + ({p[2]:+.6f} + {p[8]:+.6f}*temp + {p[5]:+.6f}*age) * age_S")
        print(f"           + ({p[3]:+.6f} + {p[9]:+.6f}*temp + {p[6]:+.6f}*age) * age_M")
        print(f"           + ({p[4]:+.6f} + {p[10]:+.6f}*temp + {p[7]:+.6f}*age) * age_H")
        
        A = np.array(A_ub)
        b = np.array(b_ub)
        sat = np.sum(A.dot(p) <= b)
        print(f"\n  Constraints satisfied: {sat}/{len(A_ub)} ({sat/len(A_ub)*100:.1f}%)")
    else:
        print(f"\n  {res.message}")

if __name__ == '__main__':
    run_lp()
