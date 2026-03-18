# F1 Strategy Regulations

## Overview

This document outlines the rules and regulations governing the Box Box Box F1 Strategy Challenge. These regulations define how races are simulated and how finishing positions are determined.

---

## Race Structure

### Grid Start
- All races start with 20 cars assigned to positions pos1 through pos20
- Starting grid position is only used for strategy assignment
- **Starting position does NOT affect lap times or race outcome**

### Independent Simulation
- **Each car races independently in a "virtual lane"**
- No car-to-car interaction or overtaking
- No blocking, drafting, or racing line disputes
- Each car's performance depends only on its own strategy and tire management
- Pure time trial: best total time wins

### Race Length
- Each race consists of a fixed number of laps
- Total laps varies by track (typically 50-70 laps)
- All cars must complete the same number of laps

### Timing
- Races are won by the shortest **total race time**
- Total race time = sum of all lap times + pit stop penalties
- Positions 1-20 are determined by ascending total time

---

## Vehicle and Driver Parity

### Equal Performance
- **All 20 cars have identical performance capabilities**
- Car performance does not vary between teams or positions
- Aerodynamics, engine power, and handling are equal
- No mechanical advantages or disadvantages

### Equal Skill
- **All 20 drivers have equal skill levels**
- Driver performance does not vary
- No driver errors or exceptional performances
- Strategy and tire management are the only differentiating factors

---

## Tire Compounds

Three tire compounds are available in every race. Each compound has distinct performance characteristics.

### SOFT Compound (Red)
- **Performance**: Fastest lap times
- **Durability**: Lowest durability
- **Usage**: Best for short stints when maximum speed is needed
- **Trade-off**: Performance degrades with usage

### MEDIUM Compound (Yellow)
- **Performance**: Moderate lap times
- **Durability**: Balanced durability
- **Usage**: Versatile option for various strategies
- **Trade-off**: Balance between speed and longevity

### HARD Compound (White)
- **Performance**: Slowest lap times initially
- **Durability**: Highest durability
- **Usage**: Optimal for long stints
- **Trade-off**: Slower but maintains performance longer

### Compound Selection
- Teams can choose any compound at any time
- Different compounds can be used throughout the race
- Compound choice significantly impacts race strategy

---

## Tire Degradation

### Performance Evolution
- Tire performance changes as tires are used
- Each compound degrades at different rates
- Degradation affects lap times progressively
- Fresh tires provide optimal performance

### Degradation Characteristics
- **SOFT**: Degrades faster due to softer rubber compound
- **MEDIUM**: Moderate degradation rate
- **HARD**: Degrades slower due to harder rubber compound

### Initial Performance Period
- New tires exhibit a period of consistent performance
- After this period, degradation effects become noticeable
- The duration of optimal performance varies by compound

---

## Pit Stops

### Mandatory Requirements
- **Compound Rule**: Drivers must use at least 2 different tire compounds during the race
- This rule ensures strategic variety

### Pit Lane Time Penalty
- Entering the pit lane incurs a time penalty
- Penalty duration is specified per race (typically 20-25 seconds)
- This time is added to total race time
- Penalty represents time lost while other cars stay on track

### Pit Stop Mechanics
- Drivers can pit at the end of any lap
- Tire changes are instantaneous (no additional time beyond pit lane penalty)
- Any compound can be fitted during a pit stop
- Multiple pit stops are allowed
- There is no limit on number of stops

### Strategic Considerations
- Fewer pit stops = less time penalty
- More pit stops = fresher tires more often
- Optimal strategy balances tire performance vs. time lost

---

## Track Conditions

### Track Temperature
- Track temperature significantly affects tire behavior
- Temperature impacts how tires degrade
- Higher temperatures generally increase degradation
- Lower temperatures generally reduce degradation
- Temperature remains constant throughout each race

### Base Lap Time
- Each track has a characteristic base lap time
- Base lap time represents the track's inherent speed
- Faster tracks have lower base lap times
- Base lap time is the reference point for all calculations

---

## Lap Time Calculation

### Components
A lap time consists of:
1. **Base lap time** - Track's characteristic speed
2. **Tire compound effect** - Compound speed differential
3. **Tire degradation effect** - Performance loss from tire usage
4. **Temperature effect** - Impact of track temperature on tire behavior

### Factors to Consider
- Each compound has inherent speed characteristics
- Tire age (laps completed on current tires) affects performance
- Temperature interacts with degradation behavior
- All factors combine to determine final lap time

---

## Race Environment

### Controlled Conditions
- Track conditions remain constant during each race
- No weather changes during the race
- No safety cars or virtual safety cars
- No accidents or mechanical failures
- No traffic effects or car interaction
- Pure strategy competition

---

## Strategy Elements

### Key Strategic Decisions
1. **Starting tire compound**: Which compound to start the race on
2. **Pit stop timing**: When to stop for fresh tires
3. **Compound progression**: Which compounds to use and in what order
4. **Number of stops**: How many pit stops to make

### Common Strategy Types
- **One-stop**: Minimize time lost in pits (requires durable tire choices)
- **Two-stop**: Balance performance and pit losses
- **Multi-stop**: Maximize tire freshness (rare, but possible)

### Strategy Trade-offs
- Stopping early: Fresh tires but time penalty early
- Stopping late: Extended stint but degraded performance
- Aggressive compounds: Faster but require more stops
- Conservative compounds: Slower but fewer stops needed

---

## Simulation Mechanics

### Lap-by-Lap Simulation
- Race is simulated lap by lap
- Each car's lap time is calculated independently per lap
- Pit stops occur at the end of specified laps
- Total time accumulates throughout the race

### Tire Age Tracking
- System tracks laps completed on current tire set
- Fresh tires start at age 0 when fitted
- At the start of each lap, tire age increments by 1 before calculating lap time
- The first lap on fresh tires is driven at age 1

### Result Determination
- After final lap, cars are sorted by total race time
- Finishing positions are assigned: 1st (fastest) to 20th (slowest)
- Driver IDs are returned in finishing order

---

## Important Notes

- All regulations are consistently applied across all races
- The same rules govern every simulation
- Understanding these regulations is essential for accurate race prediction
