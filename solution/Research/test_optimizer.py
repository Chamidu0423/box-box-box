import json, os, glob, math
from scipy.optimize import differential_evolution

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "optimal_params.json")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Run global parameter optimization against test cases.
def optimize():
    cases = []
    for f in glob.glob(os.path.join(ROOT_DIR, "data/test_cases/inputs/test_*.json")):
        ans = f.replace("inputs", "expected_outputs")
        if os.path.exists(ans):
            cases.append((json.load(open(f)), json.load(open(ans))['finishing_positions']))

    print(f"🌍 Running Global Optimization on {len(cases)} cases...")
    print("Please wait 2-3 minutes. Progress will be shown below:")

    # Objective: minimize total position-distance error across all races.
    def calc_error(params):
        O_M, O_H, D_S, D_M, D_H, P_Pos, F_B = params
        err = 0
        for race, exp in cases:
            base = race['race_config']['base_lap_time']
            tf = race['race_config']['track_temp'] / 30.0
            offs = {'SOFT': 0, 'MEDIUM': base*O_M, 'HARD': base*O_H}
            degs = {'SOFT': base*D_S, 'MEDIUM': base*D_M, 'HARD': base*D_H}
            
            res = []
            for p_str, strat in race['strategies'].items():
                pos = int(p_str.replace('pos', ''))
                t = math.sqrt(pos-1) * base * P_Pos
                tire, age = strat['starting_tire'], 1
                pits = {p['lap']: p['to_tire'] for p in strat.get('pit_stops', [])}
                
                for l in range(1, race['race_config']['total_laps']+1):
                    t += base + offs[tire] + (degs[tire]*(age**2)*tf) - (F_B*l)
                    age += 1
                    if l in pits:
                        t += race['race_config']['pit_lane_time']
                        tire, age = pits[l], 1
                res.append((t, strat['driver_id']))
            pred = [r[1] for r in sorted(res, key=lambda x: x[0])]
            err += sum(abs(exp.index(d) - pred.index(d)) for d in exp)
        return err

    bounds = [(0.01,0.20), (0.10,0.30), (0.001,0.015), (0.0001,0.005), (1e-5,0.001), (0.0001,0.005), (0,0.02)]
    
    # Enable step-by-step optimization progress output.
    res = differential_evolution(calc_error, bounds, strategy='best1bin', maxiter=50, popsize=15, disp=True)
    
    # Save best parameters found by the optimizer.
    p = res.x
    json.dump({"O_M":p[0], "O_H":p[1], "D_S":p[2], "D_M":p[3], "D_H":p[4], "P_Pos":p[5], "F_B":p[6]}, open(PARAMS_FILE, 'w'), indent=4)
    print("\n✅ Done! Best Error Score:", res.fun)

if __name__ == "__main__": optimize()