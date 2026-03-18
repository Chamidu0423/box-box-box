# Data Format Specification

## Historical Race Record

Each historical race contains complete information about the race and its outcome.

```json
{
  "race_id": "R00001",
  "race_config": {
    "track": "Monza",
    "total_laps": 53,
    "base_lap_time": 82.5,
    "pit_lane_time": 22.0,
    "track_temp": 32
  },
  "strategies": {
    "pos1": {
      "driver_id": "D001",
      "starting_tire": "SOFT",
      "pit_stops": [
        {
          "lap": 18,
          "from_tire": "SOFT",
          "to_tire": "MEDIUM"
        }
      ]
    },
    "pos2": { ... },
    ...
    "pos20": { ... }
  },
  "finishing_positions": [
    "D005",
    "D012",
    "D001",
    ...
  ]
}
```

## Test Case Format

Test cases contain race configuration and strategies, but NOT finishing positions (you must predict those).

```json
{
  "race_id": "TEST_001",
  "race_config": {
    "track": "Silverstone",
    "total_laps": 45,
    "base_lap_time": 85.3,
    "pit_lane_time": 21.5,
    "track_temp": 28
  },
  "strategies": {
    "pos1": {
      "driver_id": "D001",
      "starting_tire": "MEDIUM",
      "pit_stops": [
        {
          "lap": 22,
          "from_tire": "MEDIUM",
          "to_tire": "SOFT"
        }
      ]
    },
    ...
  }
}
```

## Field Descriptions

### Race Configuration

| Field | Type | Description |
|-------|------|-------------|
| `track` | string | Track name |
| `total_laps` | integer | Number of laps in the race |
| `base_lap_time` | float | Baseline lap time in seconds |
| `pit_lane_time` | float | Time penalty for pit stop in seconds |
| `track_temp` | integer | Track temperature in Celsius |

### Strategy

| Field | Type | Description |
|-------|------|-------------|
| `driver_id` | string | Unique identifier (D001-D020) |
| `starting_tire` | string | Initial tire compound: "SOFT", "MEDIUM", or "HARD" |
| `pit_stops` | array | List of planned pit stops |

### Pit Stop

| Field | Type | Description |
|-------|------|-------------|
| `lap` | integer | Lap number when pit stop occurs |
| `from_tire` | string | Current tire compound |
| `to_tire` | string | New tire compound to fit |

## Your Output Format

Your program must output predictions in this format:

```json
{
  "race_id": "TEST_001",
  "finishing_positions": [
    "D012",
    "D005",
    "D018",
    "D001",
    ...
  ]
}
```

### Requirements

- `race_id`: Must match the input test case
- `finishing_positions`: Array of exactly 20 driver IDs
  - Ordered from 1st place (fastest time) to 20th place (slowest time)
  - Must contain all drivers from D001 to D020
  - No duplicates

## Tire Compounds

Three tire compounds are available, each with different characteristics:

- **SOFT** (Red): Fastest lap times but wears quickly
- **MEDIUM** (Yellow): Balanced performance and durability
- **HARD** (White): Slowest lap times but most durable

The exact performance characteristics must be determined through analysis of the historical data.

## Race IDs

- Historical races: `R00001` through `R30000`
- Test cases: `TEST_001` through `TEST_100`

**Note**: Test case filenames use lowercase format (`test_001.json`), but the `race_id` field inside the JSON uses uppercase format (`TEST_001`).

## Loading Data

### Python Example
```python
import json

# Load historical races
with open('data/historical_races/races_00000-00999.json', 'r') as f:
    races = json.load(f)

# Load test case
with open('data/test_cases/inputs/test_001.json', 'r') as f:
    test_case = json.load(f)

# Load expected output (for validation)
with open('data/test_cases/expected_outputs/test_001.json', 'r') as f:
    expected = json.load(f)
```

### JavaScript Example
```javascript
const fs = require('fs');

// Load historical races
const races = JSON.parse(
  fs.readFileSync('data/historical_races/races_00000-00999.json', 'utf8')
);

// Load test case
const testCase = JSON.parse(
  fs.readFileSync('data/test_cases/inputs/test_001.json', 'utf8')
);

// Load expected output (for validation)
const expected = JSON.parse(
  fs.readFileSync('data/test_cases/expected_outputs/test_001.json', 'utf8')
);
```

### Java Example
```java
import com.google.gson.Gson;
import java.io.FileReader;

Gson gson = new Gson();

// Load test case
TestCase testCase = gson.fromJson(
    new FileReader("data/test_cases/inputs/test_001.json"),
    TestCase.class
);

// Load expected output (for validation)
ExpectedOutput expected = gson.fromJson(
    new FileReader("data/test_cases/expected_outputs/test_001.json"),
    ExpectedOutput.class
);
```

## Notes

- All times are in seconds
- All temperatures are in Celsius
- Grid positions (pos1-pos20) are starting positions only
- Finishing positions are determined by total race time (fastest wins)
