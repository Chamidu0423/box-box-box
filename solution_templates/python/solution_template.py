#!/usr/bin/env python3
"""
Box Box Box - F1 Race Simulator Template (Python)

This template shows the required input/output structure.
Implement your race simulation logic to predict finishing positions.
"""

import json
import sys


def main():
    # Read test case from stdin
    test_case = json.load(sys.stdin)

    race_id = test_case['race_id']
    race_config = test_case['race_config']
    strategies = test_case['strategies']

    # TODO: Implement your race simulation logic here
    # Analyze the historical data in data/historical_races/ to understand
    # how to accurately simulate races and predict finishing positions

    finishing_positions = []  # Replace with your simulation results

    # Output result to stdout
    output = {
        'race_id': race_id,
        'finishing_positions': finishing_positions  # List of 20 driver IDs (1st to 20th)
    }

    print(json.dumps(output))


if __name__ == '__main__':
    main()
