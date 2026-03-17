import json
import sys

# ============================================================
# BEST PARAMETERS (from fit_quad_perfect.py - 72% accuracy)
# Formula: lap_time = base_lap_time
#        + M_base * laps_on_M + H_base * laps_on_H   (S_base = 0)
#        + S_deg1 * age_S + M_deg1 * age_M + H_deg1 * age_H
#        + S_deg2 * age_S^2 + M_deg2 * age_M^2 + H_deg2 * age_H^2
#        + S_td * temp * age_S + M_td * temp * age_M + H_td * temp * age_H
#        + S_tb * temp * laps_S + M_tb * temp * laps_M + H_tb * temp * laps_H
#
# Note: base_lap_time is the same for everyone in a race, so
#       it doesn't affect relative ordering between drivers.
# ============================================================

PARAMS = {
    # Compound base offsets (relative to SOFT = 0)
    'M_base':  0.00149,
    'H_base':  0.00771,
    # Linear degradation (age)
    'S_deg1':  0.05756,
    'M_deg1':  0.00878,
    'H_deg1':  0.02278,
    # Quadratic degradation (age^2)
    'S_deg2':  0.00000,
    'M_deg2':  0.00103,
    'H_deg2':  0.00043,
    # Temperature × age interaction (degradation)
    'S_td':    0.00995,
    'M_td':    0.00296,
    'H_td':    0.00054,
    # Temperature × laps interaction (base per tire)
    'S_tb':   -0.04792,
    'M_tb':    0.00966,
    'H_tb':    0.03822,
}


def compute_total_time(strategy, laps_total, track_temp, pit_lane_time):
    """
    Compute estimated total race time for a strategy.
    base_lap_time is NOT included since it's constant for all drivers.
    """
    p = PARAMS

    current_tire = strategy['starting_tire']
    pit_stops = sorted(strategy['pit_stops'], key=lambda x: x['lap'])

    pidx = 0
    laps_on_tire = 1

    laps_S = laps_M = laps_H = 0
    sum_age_S = sum_age_M = sum_age_H = 0
    sum_sq_S = sum_sq_M = sum_sq_H = 0

    for lap in range(1, laps_total + 1):
        if pidx < len(pit_stops) and pit_stops[pidx]['lap'] == lap:
            current_tire = pit_stops[pidx]['to_tire']
            pidx += 1
            laps_on_tire = 1

        if current_tire == 'SOFT':
            laps_S += 1
            sum_age_S += laps_on_tire
            sum_sq_S  += laps_on_tire ** 2
        elif current_tire == 'MEDIUM':
            laps_M += 1
            sum_age_M += laps_on_tire
            sum_sq_M  += laps_on_tire ** 2
        else:  # HARD
            laps_H += 1
            sum_age_H += laps_on_tire
            sum_sq_H  += laps_on_tire ** 2

        laps_on_tire += 1

    # Compound base offsets (relative to SOFT=0)
    base_cost = (p['M_base'] * laps_M) + (p['H_base'] * laps_H)

    # Linear degradation
    deg1_cost = (p['S_deg1'] * sum_age_S +
                 p['M_deg1'] * sum_age_M +
                 p['H_deg1'] * sum_age_H)

    # Quadratic degradation
    deg2_cost = (p['S_deg2'] * sum_sq_S +
                 p['M_deg2'] * sum_sq_M +
                 p['H_deg2'] * sum_sq_H)

    # Temperature × age degradation interaction
    td_cost = track_temp * (p['S_td'] * sum_age_S +
                            p['M_td'] * sum_age_M +
                            p['H_td'] * sum_age_H)

    # Temperature × laps interaction (compound-level)
    tb_cost = track_temp * (p['S_tb'] * laps_S +
                            p['M_tb'] * laps_M +
                            p['H_tb'] * laps_H)

    # Pit lane penalty
    pit_cost = len(pit_stops) * pit_lane_time

    return base_cost + deg1_cost + deg2_cost + td_cost + tb_cost + pit_cost


def simulate_race(race_config, strategies):
    """
    Simulate a race and return finishing_positions list (fastest to slowest).
    Tiebreaker: starting grid position (pos1 beats pos20).
    """
    total_laps = race_config['total_laps']
    track_temp = race_config['track_temp']
    pit_lane_time = race_config['pit_lane_time']

    results = []
    for pos_key, strategy in strategies.items():
        driver_id = strategy['driver_id']
        grid_pos = int(pos_key.replace('pos', ''))

        time = compute_total_time(strategy, total_laps, track_temp, pit_lane_time)
        results.append((time, grid_pos, driver_id))

    # Sort by time ascending, then grid_pos ascending as tiebreaker
    results.sort(key=lambda x: (x[0], x[1]))

    return [r[2] for r in results]


def main():
    input_data = json.load(sys.stdin)

    race_config = input_data['race_config']
    strategies = input_data['strategies']
    race_id = input_data['race_id']

    finishing_positions = simulate_race(race_config, strategies)

    output = {
        'race_id': race_id,
        'finishing_positions': finishing_positions
    }

    print(json.dumps(output))


if __name__ == '__main__':
    main()
