import json
import sys
import os
import math

# Predict race finishing order using an analytical lap-time model.
def predict_race(race, params):
    base = race['race_config']['base_lap_time']
    tf = race['race_config']['track_temp'] / 30.0
    
    offs = {'SOFT': 0.0, 'MEDIUM': base * params.get("O_M", 0.094), 'HARD': base * params.get("O_H", 0.165)}
    degs = {'SOFT': base * params.get("D_S", 0.0044), 'MEDIUM': base * params.get("D_M", 0.0008), 'HARD': base * params.get("D_H", 0.0002)}
    
    P_Pos = params.get("P_Pos", 0.0014)
    F_B = params.get("F_B", 0.0)
    
    max_laps = {
        'SOFT': params.get("M_S", 19.0), 
        'MEDIUM': params.get("M_M", 23.0), 
        'HARD': params.get("M_H", 38.0)
    }
    C_P = params.get("C_P", 0.096) 
    
    # Store total race time per driver, then sort by fastest total time.
    res = []
    for p_str, strat in race['strategies'].items():
        pos = int(p_str.replace('pos', ''))
        t = math.sqrt(pos - 1) * base * P_Pos
        tire, age = strat['starting_tire'], 1
        pits = {p['lap']: p['to_tire'] for p in strat.get('pit_stops', [])}
        
        for l in range(1, race['race_config']['total_laps'] + 1):
            lap_time = base + offs[tire] + (degs[tire] * (age**2) * tf) - (F_B * l)
            
            # Add an extra time penalty when tire age exceeds its max-lap threshold.
            if age > max_laps[tire]:
                lap_time += (base * C_P) * (age - max_laps[tire])
                
            t += lap_time
            age += 1
            
            if l in pits:
                t += race['race_config']['pit_lane_time']
                tire, age = pits[l], 1
                
        res.append((t, strat['driver_id']))
        
    return [r[1] for r in sorted(res, key=lambda x: x[0])]

def main():
    # Read one race JSON object from standard input.
    data = sys.stdin.read()
    if data.strip():
        race = json.loads(data)
        params_file = os.path.join(os.path.dirname(__file__), "optimal_params.json")
        
        # Use saved tuned parameters when available.
        params = {}
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                params = json.load(f)
                
        finishing_positions = predict_race(race, params)
        
        # Keep output format aligned with submission requirements.
        race_id = race.get("race_id", race.get("race_config", {}).get("race_id"))
        output = {"race_id": race_id, "finishing_positions": finishing_positions}
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()