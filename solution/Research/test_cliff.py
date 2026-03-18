import json, os, glob, math
from scipy.optimize import differential_evolution

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_INPUTS = os.path.join(ROOT_DIR, "data", "test_cases", "inputs")
TEST_ANSWERS = os.path.join(ROOT_DIR, "data", "test_cases", "expected_outputs")
PARAMS_FILE = os.path.join(ROOT_DIR, "solution", "optimal_params.json")

# Load all test races paired with expected outcomes.
def load_cases():
    cases = []
    for f in glob.glob(os.path.join(TEST_INPUTS, "test_*.json")):
        ans = os.path.join(TEST_ANSWERS, os.path.basename(f))
        if os.path.exists(ans):
            cases.append((json.load(open(f)), json.load(open(ans))['finishing_positions']))
    return cases

# Tune base model + tire-cliff parameters together.
def test_tire_cliff():
    cases = load_cases()
    print("🌍 Final Run: Saving The 'Progressive Tire Cliff' Penalty...")

    def calc_error(params):
        O_M, O_H, D_S, D_M, D_H, P_Pos, F_B, M_S, M_M, M_H, C_P = params
        err = 0
        
        for race, exp in cases:
            base = race['race_config']['base_lap_time']
            tf = race['race_config']['track_temp'] / 30.0
            offs = {'SOFT': 0, 'MEDIUM': base*O_M, 'HARD': base*O_H}
            degs = {'SOFT': base*D_S, 'MEDIUM': base*D_M, 'HARD': base*D_H}
            max_laps = {'SOFT': M_S, 'MEDIUM': M_M, 'HARD': M_H}
            
            res = []
            for p_str, strat in race['strategies'].items():
                pos = int(p_str.replace('pos', ''))
                t = math.sqrt(pos-1) * base * P_Pos
                tire, age = strat['starting_tire'], 1
                pits = {pt['lap']: pt['to_tire'] for pt in strat.get('pit_stops', [])}
                
                for l in range(1, race['race_config']['total_laps']+1):
                    lap_time = base + offs[tire] + (degs[tire]*(age**2)*tf) - (F_B*l)
                    
                    if age > max_laps[tire]:
                        # Apply the progressive tire-cliff penalty here.
                        lap_time += (base * C_P) * (age - max_laps[tire])
                        
                    t += lap_time
                    age += 1
                    
                    if l in pits:
                        t += race['race_config']['pit_lane_time']
                        tire, age = pits[l], 1
                res.append((t, strat['driver_id']))
            pred = [r[1] for r in sorted(res, key=lambda x: x[0])]
            err += sum(abs(exp.index(d) - pred.index(d)) for d in exp)
        return err

    # Search ranges for all model parameters, including cliff controls.
    bounds = [
        (0.01,0.20), (0.10,0.30), (0.001,0.015), (0.0001,0.005), (1e-5,0.001), 
        (0.0001,0.005), (0,0.02),
        (10, 25), (20, 40), (30, 60), (0.0, 0.1) 
    ]
    
    # Use maxiter=50 for a more thorough optimization search.
    res = differential_evolution(calc_error, bounds, strategy='best1bin', maxiter=50, popsize=10, disp=True)
    
    p = res.x
    out_dict = {
        "O_M": p[0], "O_H": p[1], "D_S": p[2], "D_M": p[3], "D_H": p[4],
        "P_Pos": p[5], "F_B": p[6], "M_S": p[7], "M_M": p[8], "M_H": p[9], "C_P": p[10]
    }
    
    with open(PARAMS_FILE, 'w') as f:
        json.dump(out_dict, f, indent=4)
        
    print("\n✅ Done! Saved to optimal_params.json")
    print(f"Error Score: {res.fun}")

if __name__ == "__main__":
    test_tire_cliff()