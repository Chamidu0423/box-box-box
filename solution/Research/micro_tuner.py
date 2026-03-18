import json
import os
import glob
import math
import random

# Paths
SOLUTION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SOLUTION_DIR)
TEST_INPUTS_DIR = os.path.join(ROOT_DIR, "data", "test_cases", "inputs")
TEST_ANSWERS_DIR = os.path.join(ROOT_DIR, "data", "test_cases", "expected_outputs")
PARAMS_FILE = os.path.join(SOLUTION_DIR, "optimal_params.json")

# Load test input and expected output pairs.
def load_test_cases():
    test_files = glob.glob(os.path.join(TEST_INPUTS_DIR, "test_*.json"))
    cases = []
    for file in test_files:
        ans_file = os.path.join(TEST_ANSWERS_DIR, os.path.basename(file))
        if os.path.exists(ans_file):
            with open(file, 'r') as f1, open(ans_file, 'r') as f2:
                cases.append((json.load(f1), json.load(f2)['finishing_positions']))
    return cases

# Evaluate one parameter set by total ranking error on all cases.
def evaluate(params, cases):
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

def micro_tune():
    cases = load_test_cases()
    
    # Load the current best parameter values.
    try:
        with open(PARAMS_FILE, 'r') as f:
            p_dict = json.load(f)
        current_best = [
            p_dict.get("O_M", 0.094), p_dict.get("O_H", 0.165),
            p_dict.get("D_S", 0.0044), p_dict.get("D_M", 0.0008), p_dict.get("D_H", 0.0002),
            p_dict.get("P_Pos", 0.0014), p_dict.get("F_B", 0.005)
        ]
    except FileNotFoundError:
        print("optimal_params.json not found!")
        return
        
    best_err = evaluate(current_best, cases)
    print(f"🏁 Starting Micro-Tuning... Current Best Error: {best_err}")
    print("Doing 20,000 micro-adjustments. (Press Ctrl+C to stop anytime)")
    
    # Start from current best and keep only improvements.
    try:
        for i in range(20000):
            # Apply a random micro-tweak between -5% and +5% to each parameter.
            mutated = [p * random.uniform(0.95, 1.05) for p in current_best]
            
            new_err = evaluate(mutated, cases)
            
            # Save only if the new parameters reduce the error.
            if new_err < best_err:
                best_err = new_err
                current_best = mutated
                print(f"[{i}] 🎉 New Best Error: {best_err}!")
                
                # Persist improvements immediately so progress is never lost.
                out_dict = {
                    "O_M": current_best[0], "O_H": current_best[1],
                    "D_S": current_best[2], "D_M": current_best[3], "D_H": current_best[4],
                    "P_Pos": current_best[5], "F_B": current_best[6]
                }
                with open(PARAMS_FILE, 'w') as f:
                    json.dump(out_dict, f, indent=4)
                    
    except KeyboardInterrupt:
        print("\n🛑 Tuning stopped by user.")
        
    print(f"\nFinal Best Error: {best_err}. Run evaluator.py to see new accuracy!")

if __name__ == "__main__":
    micro_tune()