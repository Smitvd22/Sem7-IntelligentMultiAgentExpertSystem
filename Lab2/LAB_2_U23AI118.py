import time
import random

class Rule:
    def __init__(self, rule_id, conditions, conclusion):
        self.rule_id = rule_id
        self.conditions = conditions  # list of strings, e.g. ['A', 'NOT B']
        self.conclusion = conclusion

def get_conflict_set(rules, wm):
    conflict_set = []
    for rule in rules:
        match = True
        for cond in rule.conditions:
            if cond.startswith("NOT "):
                fact = cond[4:]
                if fact in wm:
                    match = False
                    break
            else:
                if cond not in wm:
                    match = False
                    break
        
        # Refractoriness: rule is applicable if conclusion is not already in WM
        if match and rule.conclusion not in wm:
            conflict_set.append(rule)
            
    return conflict_set

def exercise_1():
    print("--- Exercise 1 & 2: Matcher with NOT conditions and incremental WM ---")
    rules = [
        Rule("R1", ["A", "B"], "C"),
        Rule("R2", ["C", "NOT D"], "E"),
        Rule("R3", ["A", "E"], "F")
    ]
    
    wm = set()
    # Facts to add one by one
    facts_to_add = ["A", "B", "C", "D", "E"] 
    
    print(f"Initial WM: {wm}")
    cs = get_conflict_set(rules, wm)
    print(f"Conflict set: {[r.rule_id for r in cs]}\n")
    
    for fact in facts_to_add:
        print(f"Adding fact '{fact}' to WM...")
        wm.add(fact)
        print(f"Current WM: {wm}")
        cs = get_conflict_set(rules, wm)
        print(f"Conflict set: {[r.rule_id for r in cs]}\n")

def exercise_3():
    print("--- Exercise 3: Scaling to 1000 auto-generated rules ---")
    # Generate 1000 random rules
    rules = []
    possible_facts = [f"F{i}" for i in range(100)]
    
    for i in range(1000):
        # 2 to 5 conditions per rule
        num_conds = random.randint(2, 5)
        conds = []
        for _ in range(num_conds):
            fact = random.choice(possible_facts)
            if random.random() < 0.2: # 20% chance of NOT condition
                conds.append("NOT " + fact)
            else:
                conds.append(fact)
        
        conclusion = random.choice(possible_facts)
        rules.append(Rule(f"R{i}", conds, conclusion))
        
    wm = set(random.sample(possible_facts, 30))
    
    print("Measuring match time for 1000 rules...")
    start_time = time.perf_counter()
    cs = get_conflict_set(rules, wm)
    end_time = time.perf_counter()
    
    match_time = end_time - start_time
    print(f"Time to match 1000 rules: {match_time:.6f} seconds")
    print(f"Number of rules in conflict set: {len(cs)}")
    print("\nObservation: The time taken to match rules grows linearly with the number of rules. ")
    print("This is inefficient for large rule bases, as every rule is checked against WM every cycle.")
    print("The RETE algorithm exists to optimize this by building a network of conditions to share ")
    print("evaluations and remember partial matches, which avoids re-evaluating unmodified facts and ")
    print("drastically reduces the time required for the match phase.")

if __name__ == "__main__":
    exercise_1()
    print("\n" + "="*60 + "\n")
    exercise_3()
