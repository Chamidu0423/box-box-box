import json
from collections import defaultdict

def analyze_identical_times():
    with open('data/historical_races/races_00000-00999.json') as f:
        races = json.load(f)
        
    for race in races[:200]:
        positions = {d: i+1 for i, d in enumerate(race['finishing_positions'])}
        laps_total = race['race_config']['total_laps']
        
        by_strategy = defaultdict(list)
        
        for _, strategy in race['strategies'].items():
            driver = strategy['driver_id']
            pos = positions[driver]
            
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
            
            by_strategy[sig].append({
                'driver': driver,
                'pos': pos,
                'start_tire': strategy['starting_tire'],
                'pits': strategy['pit_stops'],
                'strategy': strategy
            })
            
        for sig, drivers in by_strategy.items():
            if len(drivers) > 1:
                drivers = sorted(drivers, key=lambda d: d['pos'])
                
                # We want to look at those with DIFFERENT orders, meaning they theoretically have the same time
                # if the model is perfectly linear and independent of order.
                orders = []
                for d in drivers:
                    order = [d['start_tire']]
                    for p in d['pits']: order.append(p['to_tire'])
                    orders.append(tuple(order))
                
                if len(set(orders)) > 1 and sig[3] == 1:
                    d1 = drivers[0]
                    d2 = drivers[1]
                    
                    t1_1 = d1['start_tire']
                    t2_1 = d2['start_tire']
                    
                    if t1_1 != t2_1:
                        # Check if driver ID sorting explains it
                        print(f"Race {race['race_id']} | Faster Pos {d1['pos']}: {d1['driver']} | Slower Pos {d2['pos']}: {d2['driver']} | Diff: {d1['driver'] < d2['driver']}")

if __name__ == '__main__':
    analyze_identical_times()
