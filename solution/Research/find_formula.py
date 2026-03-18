import json, os, glob
from scipy.optimize import minimize

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_INPUTS = os.path.join(ROOT_DIR, "data", "test_cases", "inputs")
TEST_ANSWERS = os.path.join(ROOT_DIR, "data", "test_cases", "expected_outputs")

# Load test races and expected finishing orders.
def load_cases():
    cases = []
    for file in glob.glob(os.path.join(TEST_INPUTS, "test_*.json")):
        ans_file = os.path.join(TEST_ANSWERS, os.path.basename(file))
        if os.path.exists(ans_file):
            with open(file, 'r') as f1, open(ans_file, 'r') as f2:
                cases.append((json.load(f1), json.load(f2)['finishing_positions']))
    return cases

# Evaluate a given tire-degradation formula type with parameter fitting.
def test_formula(formula_type, init_params):
    cases = load_cases()
    
    # Minimize total rank-distance error across all races.
    def calc_error(params):
        O_M, O_H, D_S, D_M, D_H = params
        total_err = 0
        
        for race, expected in cases:
            base_t = race['race_config']['base_lap_time']
            pit_t = race['race_config']['pit_lane_time']
            tf = race['race_config']['track_temp'] / 30.0
            offs = {'SOFT': 0, 'MEDIUM': base_t * O_M, 'HARD': base_t * O_H}
            degs = {'SOFT': base_t * D_S, 'MEDIUM': base_t * D_M, 'HARD': base_t * D_H}
            
            results = []
            for pos_str, strat in race['strategies'].items():
                time = 0
                tire = strat['starting_tire']
                age = 0
                pits = {p['lap']: p['to_tire'] for p in strat.get('pit_stops', [])}
                
                for l in range(1, race['race_config']['total_laps'] + 1):
                    # Compare linear vs quadratic tire-aging formulas.
                    if formula_type == 'linear':
                        time += base_t + offs[tire] + (degs[tire] * age * tf)
                    elif formula_type == 'quadratic':
                        time += base_t + offs[tire] + (degs[tire] * (age**2) * tf)
                        
                    age += 1
                    if l in pits:
                        time += pit_t
                        tire = pits[l]
                        age = 0
                results.append((time, strat['driver_id']))
                
            predicted = [r[1] for r in sorted(results, key=lambda x: x[0])]
            total_err += sum(abs(expected.index(d) - predicted.index(d)) for d in expected)
        return total_err

    res = minimize(calc_error, init_params, method='Nelder-Mead', options={'maxiter': 500})
    print(f"Formula [{formula_type}] -> Error Score: {res.fun}")

if __name__ == "__main__":
    print("Testing hidden mathematical structures...\n")
    test_formula('linear', [0.09, 0.16, 0.004, 0.0008, 0.0002])
    test_formula('quadratic', [0.09, 0.16, 0.004, 0.0008, 0.0002])