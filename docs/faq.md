# Frequently Asked Questions

## General

### Q: What programming language should I use?

A: Any language you're comfortable with! The challenge is language-agnostic. As long as your program can read JSON input and produce JSON output, you're good to go.

### Q: Can I use external libraries?

A: Yes! Use whatever libraries help you (JSON parsing, data analysis, etc.). The challenge is about the algorithm, not reinventing the wheel.

### Q: How long should this take?

A: Experienced engineers typically spend 2-4 hours. Take your time to understand the data before coding.

### Q: Can I use AI/LLMs?

A: Yes! But the dataset is too large to paste directly. You'll need to understand the problem to use AI effectively.

## Data

### Q: Do I need to use all 30,000 historical races?

A: Not necessarily. You can start with a subset to develop your model, then validate on more data. Some patterns may be visible in just a few hundred races.

### Q: What if I find inconsistencies in the data?

A: The data is generated from a consistent algorithm. If you see "inconsistencies", you might be missing a pattern or rule.

### Q: Are the historical races sorted in any way?

A: No, they're randomized to mix difficulty levels. Don't assume any ordering.

## Strategy

### Q: Where should I start?

A: 1. Load and explore a few races manually
2. Identify the key variables (tire compound, stint length, temperature)
3. Build a simple lap time calculator
4. Test and iterate

### Q: How do I know if my model is correct?

A: Compare your predictions against historical race finishing positions. Start with simple cases and work up to complex ones.

### Q: What's the most important factor?

A: There are multiple factors that interact. Don't focus on just one aspect.

## Technical

### Q: My predictions are close but not exact. Is that okay?

A: No - finishing positions must match exactly. Even one position off counts as incorrect. The algorithm is deterministic.

### Q: How precise should my lap time calculations be?

A: Work with floating-point precision (seconds with decimals). Don't round lap times until the final sorting.

### Q: Do I need to simulate lap-by-lap?

A: Yes, because tire state changes each lap, and drivers can pit on any lap. You need to track state throughout the race.

### Q: What happens if two drivers have the exact same total time?

A: In practice, this won't happen due to floating-point precision differences in the dataset.

## Debugging

### Q: My code works on some races but fails on others. Why?

A: Different races have different characteristics (temperature, length, strategies). Your model might work for simple cases but miss something in complex ones.

### Q: I'm passing 20% of tests. How do I improve?

A: You're likely handling the simplest cases. Look at the races where you fail - what's different about them?

### Q: I'm stuck at 70%. What am I missing?

A: You've probably modeled the main mechanics. Look for additional factors that might affect performance (check all provided variables).

## Submission

### Q: How do I test my solution locally?

A: Put your command in `solution/run_command.txt` and run:
```bash
./test_runner.sh
```
The script will automatically read your command and test your solution.

### Q: What's the expected output format?

A: JSON with the race_id and an array of driver_ids in finishing order. See `docs/data_format.md` for details.

### Q: How do I submit my solution?

A: Fork the repository, create your `race_simulator.*` file, and submit your fork URL. See `SUBMISSION_GUIDE.md` for complete instructions.

## Scoring

### Q: What determines my score?

A: Percentage of test cases where your predicted finishing order exactly matches the expected result.

### Q: Is partial credit given?

A: No - either the finishing order matches exactly (1 point) or it doesn't (0 points).

### Q: What's a good score?

A: - 20-30%: Basic understanding
- 60-70%: Good model
- 90-100%: Excellent! You've found the patterns

## Strategy Questions

### Q: When should drivers pit?

A: Analyze the historical data to find optimal pit windows. It depends on tire compound, stint length, and track conditions.

### Q: Why do some drivers use different strategies?

A: Different strategies can lead to different outcomes based on tire behavior. Some may prioritize track position, others tire life.

### Q: Do all races follow the same rules?

A: Yes - the same underlying mechanics apply to all races. Temperature and race length vary, but the core algorithm is consistent.

## Still Stuck?

1. Double-check your lap time calculations
2. Verify you're handling pit stops correctly
3. Test on simpler races first
4. Compare your lap times with what makes sense physically

Remember: The answer is in the data. Happy analyzing! 🏁
