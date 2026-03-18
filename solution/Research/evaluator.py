import json
import os
import glob
from race_simulator import predict_race

SOLUTION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SOLUTION_DIR)
TEST_INPUTS_DIR = os.path.join(ROOT_DIR, "data", "test_cases", "inputs")
TEST_ANSWERS_DIR = os.path.join(ROOT_DIR, "data", "test_cases", "expected_outputs")
PARAMS_FILE = os.path.join(SOLUTION_DIR, "optimal_params.json")

# Load tuned model parameters used by race_simulator.
def load_params():
    with open(PARAMS_FILE, 'r') as f:
        return json.load(f)

# Evaluate prediction quality over all available test races.
def evaluate_models():
    if not os.path.exists(TEST_ANSWERS_DIR):
        print("❌ Error: Expected outputs directory not found.")
        return

    test_files = glob.glob(os.path.join(TEST_INPUTS_DIR, "test_*.json"))
    if not test_files:
        print("❌ Error: No test files found.")
        return

    params = load_params()
    # Aggregate metrics across races.
    total_error = 0
    total_drivers = 0
    perfect_matches = 0

    print(f"🔍 Evaluating {len(test_files)} test cases with Ultimate Physics...")

    for file in test_files:
        filename = os.path.basename(file)
        answer_file = os.path.join(TEST_ANSWERS_DIR, filename)

        if not os.path.exists(answer_file): continue

        with open(file, 'r') as f:
            race_data = json.load(f)
        with open(answer_file, 'r') as f:
            expected_data = json.load(f)

        expected_order = expected_data['finishing_positions']
        predicted_order = predict_race(race_data, params)

        race_error = sum(abs(expected_order.index(d) - predicted_order.index(d)) for d in expected_order)
        total_error += race_error
        total_drivers += len(expected_order)

        if race_error == 0:
            perfect_matches += 1

    # Report both average error and normalized accuracy estimate.
    avg_pos_error = total_error / total_drivers
    max_error_per_race = 200 
    total_max_error = len(test_files) * max_error_per_race
    accuracy = (1.0 - (total_error / total_max_error)) * 100

    print("\n" + "="*50)
    print(" 🏆  ANALYTICAL MODEL EVALUATION RESULTS  🏆")
    print("="*50)
    print(f"Total Test Races      : {len(test_files)}")
    print(f"Perfect Matches       : {perfect_matches} / {len(test_files)}")
    print(f"Avg Position Error    : {avg_pos_error:.3f} places per driver")
    print(f"Estimated Accuracy    : {accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate_models()