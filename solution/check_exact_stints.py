import json

def check_stints():
    with open('data/historical_races/races_00000-00999.json') as f:
        races = json.load(f)[:500]
        
    ties_broken_by_pos = 0
    ties_broken_against_pos = 0
    
    order_dependent_results = []
    
    for race in races:
        total_laps = race['race_config']['total_laps']
        positions = {d: i+1 for i, d in enumerate(race['finishing_positions'])}
        
        orig_pos = {}
        for p, s in race['strategies'].items():
            orig_pos[s['driver_id']] = int(p.replace('pos', ''))
            
        by_counts = {}
        for _, s in race['strategies'].items():
            c = s['starting_tire']
            p = sorted(s['pit_stops'], key=lambda x: x['lap'])
            
            # Determine stint sequence: e.g. "SOFT_10, HARD_20"
            stints = []
            pidx = 0
            curr_len = 0
            L_S = L_M = L_H = 0
            
            for lap in range(1, total_laps + 1):
                if pidx < len(p) and p[pidx]['lap'] == lap:
                    stints.append(f"{c}_{curr_len}")
                    c = p[pidx]['to_tire']
                    pidx += 1
                    curr_len = 0
                curr_len += 1
                if c == 'SOFT': L_S += 1
                if c == 'MEDIUM': L_M += 1
                if c == 'HARD': L_H += 1
            stints.append(f"{c}_{curr_len}")
            
            seq = "-".join(stints)
            counts = (L_S, L_M, L_H, len(p))
            
            if counts not in by_counts:
                by_counts[counts] = []
            
            by_counts[counts].append({
                'driver': s['driver_id'],
                'pos': positions[s['driver_id']],
                'start_pos': orig_pos[s['driver_id']],
                'seq': seq
            })
            
        # Analyze
        for counts, drivers in by_counts.items():
            if len(drivers) > 1:
                drivers.sort(key=lambda d: d['pos'])
                d1 = drivers[0]
                d2 = drivers[1]
                
                if d1['seq'] == d2['seq']:
                    # EXACTLY identical sequence
                    if d1['start_pos'] < d2['start_pos']:
                        ties_broken_by_pos += 1
                    else:
                        ties_broken_against_pos += 1
                else:
                    # Same counts, DIFFERENT sequence
                    order_dependent_results.append((d1['seq'], d2['seq'], d1['start_pos'] < d2['start_pos']))

    print(f"When EXACT sequence is same:")
    print(f"Faster started ahead: {ties_broken_by_pos}")
    print(f"Faster started behind: {ties_broken_against_pos}")
    
    print(f"\nWhen counts are same but sequence is different:")
    ahead_won = sum(1 for _, _, started_ahead in order_dependent_results if started_ahead)
    behind_won = sum(1 for _, _, started_ahead in order_dependent_results if not started_ahead)
    print(f"Faster started ahead: {ahead_won}")
    print(f"Faster started behind: {behind_won}")
    
    print("\nSample different sequence matchups (Winner vs Loser):")
    for w, l, _ in order_dependent_results[:10]:
        print(f"  {w} beat {l}")

if __name__ == '__main__':
    check_stints()
