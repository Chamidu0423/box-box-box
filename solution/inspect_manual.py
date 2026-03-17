import json

def brute_force_simple_race():
    with open('data/historical_races/races_00000-00999.json') as f:
        race = json.load(f)[0]
        
    print(f"Race Config:")
    print(race['race_config'])
    
    positions = {d: i+1 for i, d in enumerate(race['finishing_positions'])}
    
    strats = list(race['strategies'].values())
    strats.sort(key=lambda s: positions[s['driver_id']])
    
    for i, s in enumerate(strats):
        pits = sorted(s['pit_stops'], key=lambda x: x['lap'])
        total_laps = race['race_config']['total_laps']
        
        # We need to trace lap by lap carefully.
        # "At the start of each lap, tire age increments by 1 before calculating lap time"
        # Wait, if I pit on lap 20...
        # Lap 20: I am on the OLD tire. I finish lap 20, then I pit.
        # Lap 21: I am on the NEW tire, age 1.
        
        laps_S, laps_M, laps_H = 0, 0, 0
        current = s['starting_tire']
        
        stints = []
        current_len = 0
        
        pidx = 0
        for lap in range(1, total_laps + 1):
            # This lap is driven on `current`
            current_len += 1
            if current == 'SOFT': laps_S += 1
            if current == 'MEDIUM': laps_M += 1
            if current == 'HARD': laps_H += 1
            
            # Check if we pit at the END of this lap
            if pidx < len(pits) and pits[pidx]['lap'] == lap:
                stints.append(f"{current_len}{current[0]}")
                current = pits[pidx]['to_tire']
                current_len = 0
                pidx += 1
                
        # Append the final stint
        stints.append(f"{current_len}{current[0]}")
                
        print(f"Pos {i+1:2d} | Driver: {s['driver_id']} | Pits: {len(pits)} | S:{laps_S:2d} M:{laps_M:2d} H:{laps_H:2d} | Stints: {'-'.join(stints)}")

if __name__ == '__main__':
    brute_force_simple_race()
