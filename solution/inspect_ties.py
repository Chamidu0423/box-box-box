import json

def investigate_ties():
    with open('data/historical_races/races_00000-00999.json') as f:
        races = json.load(f)[:100]
        
    ties_broken_by_pos = 0
    ties_broken_against_pos = 0
    
    for race in races:
        positions = {d: i+1 for i, d in enumerate(race['finishing_positions'])}
        
        # Build original pos mapping
        orig_pos = {}
        for p, s in race['strategies'].items():
            orig_pos[s['driver_id']] = int(p.replace('pos', ''))
            
        laps_total = race['race_config']['total_laps']
        
        by_strategy = {}
        
        for _, strategy in race['strategies'].items():
            driver = strategy['driver_id']
            pos = positions[driver]
            start_p = orig_pos[driver]
            
            SL, ML, HL = 0, 0, 0
            current = strategy['starting_tire']
            pit_stops = sorted(strategy['pit_stops'], key=lambda x: x['lap'])
            
            pidx = 0
            for lap in range(1, laps_total + 1):
                if current == 'SOFT': SL += 1
                elif current == 'MEDIUM': ML += 1
                elif current == 'HARD': HL += 1
                
                if pidx < len(pit_stops) and pit_stops[pidx]['lap'] == lap:
                    current = pit_stops[pidx]['to_tire']
                    pidx += 1
            
            sig = (SL, ML, HL, len(pit_stops))
            if sig not in by_strategy:
                by_strategy[sig] = []
            
            by_strategy[sig].append({
                'driver': driver,
                'pos': pos,  # final finishing position
                'start_p': start_p, # original pos1, pos2...
                'start_tire': strategy['starting_tire'],
                'pits': strategy['pit_stops'],
                'strategy': strategy
            })
            
        for sig, drivers in by_strategy.items():
            if len(drivers) > 1:
                drivers = sorted(drivers, key=lambda d: d['pos'])
                
                # Compare D1 and D2
                d1 = drivers[0] # faster
                d2 = drivers[1] # slower
                
                if d1['start_p'] < d2['start_p']:
                    ties_broken_by_pos += 1
                else:
                    ties_broken_against_pos += 1
                    
    print(f"When identical tire counts occur:")
    print(f"Faster driver started ahead (posN < posM): {ties_broken_by_pos}")
    print(f"Faster driver started behind (posN > posM): {ties_broken_against_pos}")

if __name__ == '__main__':
    investigate_ties()
