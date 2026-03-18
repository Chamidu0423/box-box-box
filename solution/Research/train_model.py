import json
import glob
import os
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

SOLUTION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SOLUTION_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "historical_races")
MODEL_FILE = os.path.join(SOLUTION_DIR, "race_ranker.xgb")

# Build a training table from historical race JSON files.
def load_dataset():
    print("Extracting PURE features from 30,000 races (No hardcoded physics)...")
    all_data = []
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    
    # Encode tire names as numeric values for model features.
    tire_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
    
    for file in tqdm(json_files):
        with open(file, 'r') as f:
            data = json.load(f)
            races = data if isinstance(data, list) else ([data] if "race_config" in data else list(data.values()))
                
            for race in races:
                cfg = race.get('race_config')
                finishing_order = race.get('finishing_positions', [])
                if not cfg or not finishing_order: continue
                
                for pos_str, strategy in race['strategies'].items():
                    d_id = strategy['driver_id']
                    if d_id not in finishing_order: continue
                    
                    target_rank = finishing_order.index(d_id) + 1
                    start_pos = int(pos_str.replace('pos', ''))
                    pit_stops = strategy.get('pit_stops', [])
                    current_tire = strategy['starting_tire']
                    
                    laps_on = {'SOFT': 0, 'MEDIUM': 0, 'HARD': 0}
                    prev_lap = 0
                    for stop in pit_stops:
                        laps_on[current_tire] += (stop['lap'] - prev_lap)
                        current_tire = stop['to_tire']
                        prev_lap = stop['lap']
                    laps_on[current_tire] += (cfg['total_laps'] - prev_lap)
                    
                    # Create non-linear and interaction features for tire behavior.
                    laps_S = laps_on['SOFT']
                    laps_M = laps_on['MEDIUM']
                    laps_H = laps_on['HARD']
                    
                    all_data.append([
                        cfg['race_id'], target_rank - 1, 
                        cfg['base_lap_time'], cfg['track_temp'], cfg['pit_lane_time'], cfg['total_laps'],
                        start_pos, len(pit_stops), tire_map[strategy['starting_tire']],
                        laps_S, laps_M, laps_H,
                        # Quadratic features to catch exponential tire wear
                        laps_S ** 2, laps_M ** 2, laps_H ** 2,
                        # Temperature interaction (Hot track + Soft tires)
                        cfg['track_temp'] * laps_S, cfg['track_temp'] * laps_M, cfg['track_temp'] * laps_H,
                        len(pit_stops) * cfg['pit_lane_time'] # Total pit loss
                    ])
                    
    columns = [
        'race_id', 'target_rank', 'base_time', 'track_temp', 'pit_time', 'total_laps',
        'start_pos', 'num_pits', 'start_tire_code', 
        'laps_S', 'laps_M', 'laps_H', 
        'laps_S_sq', 'laps_M_sq', 'laps_H_sq',
        'temp_x_S', 'temp_x_M', 'temp_x_H', 'pit_loss'
    ]
    return pd.DataFrame(all_data, columns=columns)

def train():
    df = load_dataset()
    if df.empty: return
    
    # Group rows by race_id for ranking-aware training.
    df.sort_values(by='race_id', inplace=True)
    groups = df.groupby('race_id').size().values
    
    features = [col for col in df.columns if col not in ['race_id', 'target_rank']]
    X = df[features]
    y = df['target_rank']
    
    print("Training Pure XGBoost Ranker (Unleashing AI on Data)...🚀")
    model = xgb.XGBRanker(
        objective='rank:pairwise',
        n_estimators=1500,  # More trees to learn complex rules
        learning_rate=0.03, # Slower learning for better accuracy
        max_depth=8,        # Deeper trees to find hidden patterns
        random_state=42
    )
    
    model.fit(X, y, group=groups, verbose=True)
    model.save_model(MODEL_FILE)
    print("✅ Pure AI Model Saved!")

if __name__ == "__main__":
    train()