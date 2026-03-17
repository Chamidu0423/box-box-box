import json

def read_test_docs():
    # If strategy B (Hard then Soft) is FASTER than strategy A (Soft then Hard).
    # Then Total_Time(B) < Total_Time(A)
    #
    # Wait, Soft degrades FASTER than Hard (so S_deg > H_deg).
    #
    # If model is: Deg = Deg_Rate * Laps_on_tire
    # Total degradation A = S_deg*Sum(1..10) + H_deg*Sum(1..10) = 55*S + 55*H
    # Total degradation B = H_deg*Sum(1..10) + S_deg*Sum(1..10) = 55*H + 55*S
    # Exact same.
    # 
    # If model is: Deg_Rate * Laps_on_Tire * Total_Laps_Completed
    # Deg A = S_deg*Sum(i*i) + H_deg*Sum(i*(i+10)) = 385*S + 935*H
    # Deg B = H_deg*Sum(i*i) + S_deg*Sum(i*(i+10)) = 385*H + 935*S
    # Difference B - A = 385(H-S) + 935(S-H) = 550(S-H)
    # Since S > H, S-H is positive, so B - A is positive.
    # This means B is SLOWER than A. 
    # So `Deg_Rate * Laps_on_Tire * Total_Laps_Completed` makes Soft->Hard faster.
    # But observed: Hard->Soft is faster.
    
    # What about: Lap_Time = Base_Lap + Tire_Base + Deg_Rate * Laps_on_Tire - Track_Evo * Total_Laps_Completed
    # Both subtract the exact same track evo sum. Doesn't change order.
    
    # What about Fuel Burn? (Base Lap Time improves as fuel burns weight off)
    # Same thing, it's just Track Evo.
    
    # What about: Temperature affects Tire_Base AND Deg_Rate differently.
    # Still doesn't change relative order sums.
    
    # Wait... What if the pit stop time penalty changes based on lap? No, rules say NO.
    
    # "Fresh tires start at age 0... The first lap on fresh tires is driven at age 1"
    # What if the degradation rate INCREASES exponentially, e.g. Deg = Base_Deg * (1.1 ^ laps_on_tire)?
    # A: Sum(S * 1.1^i) + Sum(H * 1.1^i)
    # B: Sum(H * 1.1^i) + Sum(S * 1.1^i)
    # EXACT SAME SUM.
    
    # Any equation `f(laps_on_tire, compound)` where total race time is the sum of lap times,
    # MUST yield EXACTLY identically sums for any permutation of unchanged stint lengths.
    
    # THE ONLY MATHEMATICAL WAY for B (10H then 10S) to not equal A (10S then 10H) is if:
    # `lap_time_function` takes an input that depends on the current position in the race AND the compound at the same time.
    # i.e., an interaction term.
    
    # Interaction 1: Compound * Total_Laps_Completed
    # Example: lap_time = Base + Tire_Base + Base_Deg*laps_on_tire + (Compound_Weight * Total_Laps)
    # A (Soft then Hard): Sum(S_w * i_1..10) + Sum(H_w * i_11..20)
    # A = S_w*55 + H_w*155
    # B (Hard then Soft): Sum(H_w * i_1..10) + Sum(S_w * i_11..20)
    # B = H_w*55 + S_w*155
    # Difference B - A = H_w*55 + S_w*155 - S_w*55 - H_w*155 = 100*S_w - 100*H_w = 100(S_w - H_w)
    # If S_w < H_w, then B - A is negative, so B is FASTER.
    # This PERFECTLY explains it! 
    # Hard->Soft is faster because the heavier/slower tire (Hard) is run early when the penalty is somehow smaller, 
    # or the Soft tire is run later when it's somehow faster.
    
    # Is there a Compound * Weight interaction mentioned in the rules?
    # "Tire performance changes as tires are used"
    # "All factors combine to determine final lap time"
    # "Temperature interacts with degradation behavior" -> Could track temperature change during the race? "Temperature remains constant throughout each race"
    
    # What if it's fuel weight? 
    # Fuel weight makes the car slower.
    # Heavy car (lap 1) + Soft Tire = Soft Tire degrades FASTER because of the heavy car?
    # Heavy car (lap 1) + Hard Tire = Hard Tire degrades slightly faster but can handle it.
    # This means Deg_Rate = Base_Deg * f(Fuel_Weight).
    # Since Fuel_Weight goes down as Total_Laps_Completed goes up:
    # Deg_Rate = Base_Deg * (Max_Fuel - Total_Laps_Completed * Fuel_Burn)
    
    # Let's test this algebraically.
    # Deg_Term = Base_Deg * laps_on_tire * (Max_Fuel - Total_Laps)
    # Let w(i) = Max_Fuel - i (decreases with lap number i)
    # A (10 S, 10 H):
    # Sum(S * laps_i * w(i)) + Sum(H * laps_i * w(i+10)) for i=1..10
    # B (10 H, 10 S):
    # Sum(H * laps_i * w(i)) + Sum(S * laps_i * w(i+10)) for i=1..10
    
    # Let's take B - A:
    # Sum(H * laps_i * w(i) + S * laps_i * w(i+10) - S * laps_i * w(i) - H * laps_i * w(i+10))
    # = Sum( laps_i * [H*w(i) + S*w(i+10) - S*w(i) - H*w(i+10)] )
    # = Sum( laps_i * [H(w(i) - w(i+10)) - S(w(i) - w(i+10))] )
    # = Sum( laps_i * (H - S) * (w(i) - w(i+10)) )
    
    # Since w(i) decreases with i, w(i) > w(i+10), so (w(i) - w(i+10)) is POSITIVE.
    # Since S degrades faster than H, S > H, so (H - S) is NEGATIVE.
    # Therefore, (H - S) * (w(i) - w(i+10)) is NEGATIVE.
    # Since laps_i is POSITIVE, the whole Sum is NEGATIVE.
    # So B - A is NEGATIVE.
    # B (Hard then Soft) < A (Soft then Hard).
    # Strategy B is FASTER.
    
    # THIS MATCHES OUR OBSERVED DATA: HARD->SOFT IS FASTER THAN SOFT->HARD.
    # The interaction between Degradation and Total Laps (Fuel/Car weight) perfectly explains the nonlinear ordering.
    
    # So the model is: 
    # Lap_Time = Base_Lap + Tire_Base + (Tire_Deg * Laps_on_Tire * (1 - Fuel_Burn_Rate * Total_Laps_Completed)) + Temp_Effect
    pass
    
if __name__ == '__main__':
    read_test_docs()
