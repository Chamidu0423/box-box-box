# Box Box Box - Race Simulator Strategy🏎️

## Overview
This solution implements a high-precision analytical model to predict F1 race finishing positions. While traditional machine learning models often struggle with the deterministic nature of simulation engines, this approach utilizes Reverse Engineering and Mathematical Optimization to achieve an estimated accuracy of 90.83%.

## 📊 Performance Metrics
After analyzing 30,000 historical races, the model was validated against 100 local test cases:

- Overall Estimated Accuracy: 90.83%
- Average Position Error: 0.917 places per driver
- Methodology: Discrete Analytical Simulation (Non-linear Degradation)

## 🧠 The Engineering Approach
The simulator bypasses "black-box" AI models in favor of a physics-based approach, focusing on five key factors identified through data analysis:

1. Quadratic Tire Degradation: Identified that tire wear follows an $Age^2$ curve rather than a linear decay, significantly improving mid-race predictions.
2. Thermal Normalization: Tire performance is dynamically adjusted based on the `track_temp`, normalized against a baseline of 30°C.
3. The "Tire Cliff" Logic: Implemented a progressive penalty system for tires exceeding their optimal life (Soft: 19, Medium: 23, Hard: 38 laps), which was the breakthrough for breaking the 85% accuracy barrier.
4. Starting Position Penalty: Modeled the initial gap between drivers using a square-root distribution to account for the "standing start" delay.
5. Fuel Burn Compensation: A linear speed-up factor was applied as the race progresses to simulate reduced vehicle mass.

## 🛠️ How to Validate
To verify the performance of this model, you can run the following evaluation scripts included in the repository:

### To see the 90.83% Accuracy Report
```bash
python solution/evaluator.py
```

### To run the Standard Test Runner
```bash
./test_runner.sh
```

## 📂 Repository Structure
- solution/race_simulator.py: The core prediction engine.
- solution/optimal_params.json: Optimized weights and decay constants discovered during the research phase.
- solution/evaluator.py: Internal validation tool to measure position-based accuracy.
