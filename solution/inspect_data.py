import json
from collections import defaultdict

def verify_data():
    with open('data/historical_races/races_00000-00999.json') as f:
        races = json.load(f)[:10]

    # Let me check if the data actually gives us LAP TIMES
    # Problem Statement says: "Your program must output a JSON object to stdout containing: finishing_positions"
    # Does historical data contain actual lap times?
    
    race = races[0]
    print("Keys in a race object:", list(race.keys()))
    print("Do we have lap times?", 'lap_times' in race or 'total_times' in race)
    
verify_data()
