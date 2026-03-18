import json
import os
from race_simulator import predict_race

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_INPUT = os.path.join(ROOT_DIR, "data", "test_cases", "inputs", "test_001.json")
TEST_ANSWER = os.path.join(ROOT_DIR, "data", "test_cases", "expected_outputs", "test_001.json")
PARAMS_FILE = os.path.join(ROOT_DIR, "solution", "optimal_params.json")

# Print side-by-side expected vs predicted order for one race.
def debug_first_race():
    with open(TEST_INPUT, 'r') as f: race_data = json.load(f)
    with open(TEST_ANSWER, 'r') as f: expected_data = json.load(f)
    with open(PARAMS_FILE, 'r') as f: params = json.load(f)

    expected_order = expected_data['finishing_positions']
    predicted_order = predict_race(race_data, params)

    print(f"{'Rank':<5} | {'Expected (True)':<15} | {'Predicted (Ours)':<15} | {'Diff (Places)':<15}")
    print("-" * 60)
    
    # Show per-rank comparison and positional difference for expected driver.
    for i in range(20):
        e_driver = expected_order[i]
        p_driver = predicted_order[i]
        
        # Calculate how far off our prediction was for the driver who SHOULD have been here
        our_guess_for_this_driver = predicted_order.index(e_driver)
        diff = our_guess_for_this_driver - i 
        
        print(f"{i+1:<5} | {e_driver:<15} | {p_driver:<15} | {diff:<15}")

if __name__ == "__main__":
    debug_first_race()