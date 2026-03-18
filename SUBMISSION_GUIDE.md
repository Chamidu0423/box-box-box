# Submission Guide

## Getting Started

### 1. Fork the Repository

Fork this repository to your own GitHub account:

```bash
# On GitHub, click the "Fork" button
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/box-box-box.git
cd box-box-box
```

### 2. Create Your Solution

All your code goes in the `solution/` directory. You can organize it however you like:

```
solution/
├── run_command.txt          # REQUIRED: Command to run your solution
├── race_simulator.py        # Your main entry point
├── helper.py                # Supporting files (optional)
├── requirements.txt         # Dependencies (optional)
└── .gitignore               # Customize as needed
```

**Python example:**
```bash
cd solution/
cp ../solution_templates/python/solution_template.py race_simulator.py
```

**JavaScript example:**
```bash
cd solution/
cp ../solution_templates/javascript/solution_template.js race_simulator.js
npm init -y
npm install  # if you need packages
```

**Java example:**
```bash
cd solution/
cp ../solution_templates/java/SolutionTemplate.java RaceSimulator.java
```

### 3. Configure Run Command

**CRITICAL**: Edit `solution/run_command.txt` with the exact command to run your solution **from the repository root**.

The file should contain a single line:

**Python:**
```
python solution/race_simulator.py
```

**JavaScript:**
```
node solution/race_simulator.js
```

**Java (simple):**
```
java -cp solution RaceSimulator
```

**Java (with Maven):**
```
cd solution && mvn compile exec:java -Dexec.mainClass="RaceSimulator"
```

**TypeScript:**
```
cd solution && npm start
```

**Go:**
```
go run solution/race_simulator.go
```

The command will be executed from the repository root, so use paths accordingly.

## Development Workflow

### Test Your Solution

Run the test runner (no arguments needed):

```bash
./test_runner.sh
```

The script will:
- Read your command from `solution/run_command.txt`
- Run it from the repository root
- Test against all 100 test cases
- Show your pass rate

### Check Results

```
╔════════════════════════════════════════════════════════╗
║          Box Box Box - Test Runner                    ║
╚════════════════════════════════════════════════════════╝

Solution Command: python solution/race_simulator.py
Test Cases Found: 100

Running tests...

✓ TEST_001
✗ TEST_002 - Incorrect prediction
✓ TEST_003
...

╔════════════════════════════════════════════════════════╗
║                    Results                             ║
╚════════════════════════════════════════════════════════╝

Total Tests:    100
Passed:         75
Failed:         25

Pass Rate:      75.0%
```

### Iterate

1. Analyze failing test cases
2. Refine your algorithm
3. Re-run `./test_runner.sh`
4. Commit your progress

```bash
git add solution/
git commit -m "Improve tire degradation model"
```

**IMPORTANT**: Only commit files in the `solution/` directory. Do not modify files outside of `solution/`.

## Output Format Requirements

Your program must:
- Read test case JSON from stdin
- Output prediction JSON to stdout
- Match this exact format:

```json
{
  "race_id": "TEST_001",
  "finishing_positions": [
    "D012",
    "D005",
    "D018",
    "D001",
    ...20 driver IDs total
  ]
}
```

### Validation Rules

- Valid JSON format
- `race_id` matches input
- `finishing_positions` contains exactly 20 driver IDs
- Ordered from 1st place (fastest) to 20th place (slowest)
- All drivers D001-D020 present, no duplicates

## Submission Process

### 1. Commit Your Final Solution

```bash
git add solution/
git commit -m "Final solution"
git push origin main
```

### 2. Verify Your Repository

Make sure your fork contains:
- `solution/run_command.txt` with the correct command
- Your solution files in `solution/`
- No modifications outside `solution/` directory

### 3. Submit Your GitHub Link

Submit the URL of your forked repository. We will:
- Clone your repository
- Read the command from `solution/run_command.txt`
- Run `./test_runner.sh` from the repository root
- Evaluate based on pass rate

**Note:** Only commits made before the deadline will be considered.

## Testing Tips

### Test Individual Cases

```bash
python solution/race_simulator.py < data/test_cases/inputs/test_001.json > output.json
cat output.json | jq .
```

### Validate Output Format

```bash
python solution/race_simulator.py < data/test_cases/inputs/test_001.json | jq .
```

### Compare With Expected Output

```bash
python solution/race_simulator.py < data/test_cases/inputs/test_001.json > my_output.json
diff <(jq -S . my_output.json) <(jq -S . data/test_cases/expected_outputs/test_001.json)
```

### Debug With Historical Data

Test on historical races where you know the answer:

```python
import json
import sys
sys.path.insert(0, 'solution')
from race_simulator import simulate_race

# Load historical race
with open('data/historical_races/races_00000-00999.json') as f:
    races = json.load(f)

race = races[0]
predicted = simulate_race(race['race_config'], race['strategies'])
expected = race['finishing_positions']

print(f"Match: {predicted == expected}")
```

## Common Issues

### Wrong Run Command

Make sure your command in `solution/run_command.txt`:
- Is a single line
- Works when executed from the **repository root**
- Uses correct paths (`solution/...`)

```bash
# Test it manually from root
cd /path/to/box-box-box
cat solution/run_command.txt
# Copy the command and run it:
python solution/race_simulator.py < data/test_cases/inputs/test_001.json
```

### Path Issues

Remember: your program runs from the repository root, not from `solution/`:

```python
# Correct - paths from root
with open('data/test_cases/inputs/test_001.json') as f:
    ...

# Wrong - assumes running from solution/
with open('../data/test_cases/inputs/test_001.json') as f:
    ...
```

### Build Artifacts

Add unwanted files to `solution/.gitignore`:

```gitignore
# Uncomment what you need
__pycache__/
*.pyc
node_modules/
*.class
target/
```

## File Organization Examples

**Python:**
```
solution/
├── run_command.txt          → "python solution/race_simulator.py"
├── race_simulator.py        → Main entry point
├── tire_model.py            → Supporting module
└── requirements.txt         → pip install -r requirements.txt
```

**JavaScript:**
```
solution/
├── run_command.txt          → "node solution/race_simulator.js"
├── race_simulator.js        → Main entry point
├── helpers.js               → Supporting module
├── package.json             → npm install
└── node_modules/            → (gitignored)
```

**Java with Maven:**
```
solution/
├── run_command.txt          → "cd solution && mvn compile exec:java -Dexec.mainClass='RaceSimulator'"
├── pom.xml                  → Maven config
├── src/
│   └── main/java/
│       └── RaceSimulator.java
└── target/                  → (gitignored)
```

## Getting Help

If you're stuck:
1. Check [FAQ](docs/faq.md)
2. Study the [Data Format](docs/data_format.md)
3. Analyze simple historical races first
4. Make sure `solution/run_command.txt` is correct

Good luck! 🏁
