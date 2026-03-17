import json
import numpy as np
from scipy.optimize import linprog

def extract_features(strategy, laps_total, track_temp, pit_lane_time):
    # Returns np.array(x), constant_term
    
    current_tire = strategy['starting_tire']
    pit_stops = sorted(strategy['pit_stops'], key=lambda x: x['lap'])
    
    pidx = 0
    laps_on_tire = 1
    
    laps_S = 0
    laps_M = 0
    laps_H = 0
    
    sum_S = 0
    sum_M = 0
    sum_H = 0
    
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
        
    x = [
        laps_S, laps_M, laps_H,
        sum_S, sum_M, sum_H,
        track_temp * sum_S, track_temp * sum_M, track_temp * sum_H,
        track_temp * laps_S, track_temp * laps_M, track_temp * laps_H
    ]
    
    constant_term = pit_lane_time * len(pit_stops)
    return np.array(x), constant_term

def run_solver():
    with open('data/historical_races/races_00000-00999.json') as f:
        races = json.load(f)[:100]
        
    A_ub = []
    b_ub = []
    
    skipped = 0
    
    for race in races:
        laps_total = race['race_config']['total_laps']
        track_temp = race['race_config']['track_temp']
        pit_lane_time = race['race_config']['pit_lane_time']
        
        strats = list(race['strategies'].values())
        positions = {d: i for i, d in enumerate(race['finishing_positions'])}
        
        # Sort by finishing position
        strats.sort(key=lambda s: positions[s['driver_id']])
        
        for i in range(len(strats)-1):
            for j in range(i+1, len(strats)):
                s1 = strats[i]     # Faster
                s2 = strats[j]     # Slower
                
                x1, c1 = extract_features(s1, laps_total, track_temp, pit_lane_time)
                x2, c2 = extract_features(s2, laps_total, track_temp, pit_lane_time)
                
                diff_x = x1 - x2
                
                # If they are exactly the same geometrically, 
                # their times should be identical. We cannot enforce t1 < t2.
                if np.allclose(diff_x, 0):
                    skipped += 1
                    continue
                    
                b_val = c2 - c1 - 0.001
                
                A_ub.append(diff_x)
                b_ub.append(b_val)
            
    print(f"Extracted {len(A_ub)} constraints. Skipped {skipped} identical strategies.")
    
    bounds = [(None, None)] * 12
    # But deg should be positive
    bounds[3] = (0.0001, None)
    bounds[4] = (0.0001, None)
    bounds[5] = (0.0001, None)
    
    # Let's fix S_base to 0 so the system isn't overdetermined
    bounds[0] = (0, 0)
    
    c = np.zeros(12)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    print("\nSolver Finished.")
    print("Success:", res.success)
    if res.success:
        print("Parameters found:")
        p = res.x
        names = ["S_base", "M_base", "H_base", "S_deg", "M_deg", "H_deg", "S_td", "M_td", "H_td", "S_tb", "M_tb", "H_tb"]
        for n, v in zip(names, p):
            print(f"{n:>8} = {v:10.5f}")
            
    else:
        print("Could not find a perfect set of parameters.")

if __name__ == '__main__':
    run_solver()
