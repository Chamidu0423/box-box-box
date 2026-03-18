import json
import os
import glob

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_ANSWERS_DIR = os.path.join(ROOT_DIR, "data", "test_cases", "expected_outputs")

# Compute average finishing rank per driver from expected test outputs.
def check_skills():
    # Create a list to store finishing ranks for all 20 drivers.
    driver_ranks = {f"D{i:03d}": [] for i in range(1, 21)}
    
    # Read all expected output files used as ground truth.
    answer_files = glob.glob(os.path.join(TEST_ANSWERS_DIR, "test_*.json"))
    if not answer_files:
        print("❌ Error: Expected outputs not found!")
        return
        
    for file in answer_files:
        with open(file, 'r') as f:
            data = json.load(f)
            # Record each driver's finishing position for every race.
            for rank, driver in enumerate(data['finishing_positions']):
                if driver in driver_ranks:
                    driver_ranks[driver].append(rank + 1)
                    
    print("🏁 Driver Power Rankings (Average Finishing Position over 100 races)")
    print("-" * 65)
    
    # Convert collected ranks into per-driver averages.
    avg_ranks = []
    for driver, ranks in driver_ranks.items():
        if ranks:
            avg = sum(ranks) / len(ranks)
            avg_ranks.append((driver, avg))
            
    # Sort drivers from best to worst average rank.
    avg_ranks.sort(key=lambda x: x[1])
    
    for driver, avg in avg_ranks:
        print(f"Driver {driver} : Average Rank = {avg:.2f}")

if __name__ == "__main__":
    check_skills()