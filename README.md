# Box Box Box 🏁
## F1 Pit Strategy Optimization Challenge

> *"Box, Box, Box"* - F1 radio call for pitting

---

### 🎯 The Challenge

Reverse-engineer the race simulation algorithm from 30,000 historical F1 races. Your goal is to predict race finishing positions based on pit stop strategies, tire choices, and track conditions.

**Your Mission:** Analyze the data, discover the hidden patterns, and build a race simulator that accurately predicts outcomes.

---

### 📊 What's Included

- **30,000 Historical Races** - Complete race data with strategies and results
- **100 Test Cases** - Validate your solution against known results
- **Comprehensive Documentation** - F1 rules, data formats, and regulations
- **Language Agnostic** - Solve in any programming language you prefer
- **Starter Templates** - Example templates in Python, JavaScript, and Java

---

### 🚀 Quick Start

```bash
# 1. Fork this repository on GitHub
# 2. Clone your fork
git clone https://github.com/yourusername/box-box-box.git
cd box-box-box

# 3. Read the problem statement
cat PROBLEM_STATEMENT.md

# 4. Explore the historical data
ls data/historical_races/

# 5. Implement your solution in the solution/ directory
# See solution_templates/ for starter templates

# 6. Configure your run command
echo "python solution/race_simulator.py" > solution/run_command.txt

# 7. Test your solution
./test_runner.sh
```

---

### 🏆 Scoring

Your solution is tested against 100 test cases with varying complexity.

**Your Score = (Correct Predictions / 100) × 100%**

A prediction is correct only if the entire finishing order (all 20 positions) matches exactly.

---

### 🧠 What You'll Learn

- **Data Analysis**: Extract patterns from large datasets
- **Reverse Engineering**: Discover hidden mechanics from examples
- **Algorithm Design**: Build accurate simulation models
- **Optimization**: Balance computational efficiency with accuracy
- **Problem Solving**: Work through ambiguity to find solutions

---

### 💡 Recommended Approach

1. **Explore the Data**: Start by examining a few historical races manually
2. **Identify Patterns**: Look for relationships between tire compounds, lap times, and conditions
3. **Build Incrementally**: Start with a simple model, then add complexity
4. **Test Frequently**: Validate against historical results before running test cases
5. **Refine**: Use test results to identify and fix inaccuracies

---

### 📖 Documentation

- **[PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md)** - Formal problem description with requirements
- **[SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)** - Step-by-step submission instructions
- **[docs/regulations.md](docs/regulations.md)** - F1 rules and racing constraints
- **[docs/data_format.md](docs/data_format.md)** - JSON structure specification
- **[docs/faq.md](docs/faq.md)** - Frequently asked questions

---

### 🎯 Success Criteria

Your solution should:
- ✅ Read race configuration and strategies from stdin (JSON format)
- ✅ Simulate the race lap-by-lap for all 20 drivers
- ✅ Calculate lap times based on tire compound, degradation, and temperature
- ✅ Handle pit stops and apply time penalties correctly
- ✅ Output finishing positions (1st to 20th) to stdout (JSON format)
- ✅ Achieve high accuracy on test cases (aim for 80%+)

---

### 🔧 Testing Your Solution

```bash
# Run all 100 test cases
./test_runner.sh

# The test runner will:
# - Read your command from solution/run_command.txt
# - Run your solution against all test cases
# - Display pass/fail results
# - Show your final score
```

**Note**: Your solution must read from stdin and write to stdout. See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) for details.

---

### 🏁 Ready to Race?

```bash
# Read the problem
cat PROBLEM_STATEMENT.md

# Choose your language and copy a template
cp solution_templates/python/solution_template.py solution/race_simulator.py

# Update run command
echo "python solution/race_simulator.py" > solution/run_command.txt

# Start coding!
code solution/race_simulator.py

# Test your solution
./test_runner.sh
```

**Good luck, and happy racing! 🏎️💨**

---

### 📧 Contact

For questions or issues, reach out to: **azeem@sansatech.com**

---

*A coding challenge for algorithm enthusiasts and F1 fans*
*Prepared by an F1 fan*
