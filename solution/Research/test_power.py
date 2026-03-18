import json, os, glob, math
from scipy.optimize import minimize

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_INPUTS = os.path.join(ROOT_DIR, "data", "test_cases", "inputs")
TEST_ANSWERS = os.path.join(ROOT_DIR, "data", "test_cases", "expected_outputs")

# Load test races with expected finishing orders.
def load_cases():
    cases = []
    for f in glob.glob(os.path.join(TEST_INPUTS, "test_*.json")):
        ans = os.path.join(TEST_ANSWERS, os.path.basename(f))
        if os.path.exists(ans):
            cases.append((json.load(open(f)), json.load(open(ans))['finishing_positions']))
    return cases

# Compare different tire-age exponents and report the best error per exponent.
def scan_powers():
    cases = load_cases()
    
    # Use our current best-known parameter values as the starting point.
    init_guess = [0.094, 0.165, 0.0044, 0.0008, 0.0002, 0.0014, 0.005]
    
    # Powers to test, slightly above 2.0.
    powers_to_test = [2.00, 2.02, 2.05, 2.08, 2.10, 2.15]
    
    print("🔍 Testing Powers slightly above 2...")
    print(f"{'Power':<6} | {'Error Score':<12}")
    print("-" * 25)
    
    # Refit parameters for each candidate power value.
    for p in powers_to_test:
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
                    pits = {pt['lap']: pt['to_tire'] for pt in strat.get('pit_stops', [])}
                    
                    for l in range(1, race['race_config']['total_laps']+1):
                        # This is where the degradation power value is applied.
                        t += base + offs[tire] + (degs[tire]*(age**p)*tf) - (F_B*l)
                        age += 1
                        if l in pits:
                            t += race['race_config']['pit_lane_time']
                            tire, age = pits[l], 1
                    res.append((t, strat['driver_id']))
                pred = [r[1] for r in sorted(res, key=lambda x: x[0])]
                err += sum(abs(exp.index(d) - pred.index(d)) for d in exp)
            return err

        # Use Nelder-Mead to quickly fit parameters for this power value.
        res = minimize(calc_error, init_guess, method='Nelder-Mead', options={'maxiter': 100})
        print(f"{p:<6} | {res.fun}")

if __name__ == "__main__":
    scan_powers()