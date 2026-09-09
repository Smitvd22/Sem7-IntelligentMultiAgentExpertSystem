class Rule:
    def __init__(self, rule_id, conditions, conclusion):
        self.rule_id = rule_id
        self.conditions = conditions
        self.conclusion = conclusion
        
    def __str__(self):
        return f"{self.rule_id}: IF {' AND '.join(self.conditions)} THEN {self.conclusion}"

def get_knowledge_base():
    """
    Defines the animal-identification knowledge base using IF-THEN production rules.
    """
    return [
        Rule("R1", ["has hair"], "is mammal"),
        Rule("R2", ["gives milk"], "is mammal"),
        Rule("R3", ["has feathers"], "is bird"),
        Rule("R4", ["flies", "lays eggs"], "is bird"),
        Rule("R5", ["is mammal", "eats meat"], "is carnivore"),
        Rule("R6", ["is mammal", "has pointed teeth", "has claws", "has forward pointing eyes"], "is carnivore"),
        Rule("R7", ["is mammal", "has hoofs"], "is ungulate"),
        Rule("R8", ["is mammal", "chews cud"], "is ungulate"),
        
        # Exercise 1: Rules for tiger, giraffe, zebra
        Rule("R9", ["is carnivore", "has tawny color", "has black stripes"], "is tiger"),
        Rule("R10", ["is ungulate", "has long neck", "has long legs"], "is giraffe"),
        Rule("R11", ["is mammal", "has black stripes", "not carnivore"], "is zebra")
    ]

def forward_chaining(rules, initial_facts, fire_all=False):
    """
    Forward chaining inference engine.
    """
    wm = set(initial_facts)
    firing_sequence = []
    
    # Target specific animal identifications
    animal_targets = ["is tiger", "is giraffe", "is zebra"]
    
    # Hierarchy to determine most specific class (Exercise 2)
    hierarchy = [
        ["is tiger", "is giraffe", "is zebra"], # Most specific
        ["is carnivore", "is ungulate"],
        ["is bird", "is mammal"]
    ]
    
    identified = None
    
    while True:
        # Construct the conflict set
        conflict_set = []
        for rule in rules:
            # Check if all conditions are satisfied and conclusion is not already in WM
            if all(cond in wm for cond in rule.conditions) and rule.conclusion not in wm:
                conflict_set.append(rule)
                
        # If conflict set is empty, stop
        if not conflict_set:
            break
            
        if fire_all:
            # Exercise 3: Fire ALL applicable rules per cycle
            fired_any_target = False
            for rule in conflict_set:
                wm.add(rule.conclusion)
                firing_sequence.append(rule.rule_id)
                if rule.conclusion in animal_targets:
                    identified = rule.conclusion
                    fired_any_target = True
            
            if fired_any_target:
                break
                
        else:
            # Select and fire the first rule from the conflict set
            rule = conflict_set[0]
            wm.add(rule.conclusion)
            firing_sequence.append(rule.rule_id)
            
            if rule.conclusion in animal_targets:
                identified = rule.conclusion
                break
                
    # If no specific animal was identified, report the most specific class established
    if identified is None:
        most_specific = None
        for level in hierarchy:
            for cls in level:
                if cls in wm:
                    most_specific = cls
                    break
            if most_specific:
                break
        
        if most_specific:
            identified = f"No specific animal identified. Most specific class established: {most_specific}"
        else:
            identified = "No identification can be established from the available facts."
            
    return wm, firing_sequence, identified

def run_exercises():
    rules = get_knowledge_base()
    
    print("=" * 60)
    print("Experiment 4: Forward Chaining Inference Engine")
    print("=" * 60)
    
    print("\n--- Exercise 1: Identify a tiger, a giraffe, and a zebra ---")
    
    # 1. Tiger
    print("\n[Testing Tiger]")
    tiger_facts = ["has hair", "eats meat", "has tawny color", "has black stripes"]
    wm, seq, ident = forward_chaining(rules, tiger_facts)
    print(f"Initial Facts: {tiger_facts}")
    print(f"Firing Sequence: {seq}")
    print(f"Final Working Memory: {wm}")
    print(f"Identification: {ident}")
    
    # 2. Giraffe
    print("\n[Testing Giraffe]")
    giraffe_facts = ["has hair", "has hoofs", "has long neck", "has long legs"]
    wm, seq, ident = forward_chaining(rules, giraffe_facts)
    print(f"Initial Facts: {giraffe_facts}")
    print(f"Firing Sequence: {seq}")
    print(f"Final Working Memory: {wm}")
    print(f"Identification: {ident}")
    
    # 3. Zebra
    print("\n[Testing Zebra]")
    zebra_facts = ["has hair", "has black stripes", "not carnivore"]
    wm, seq, ident = forward_chaining(rules, zebra_facts)
    print(f"Initial Facts: {zebra_facts}")
    print(f"Firing Sequence: {seq}")
    print(f"Final Working Memory: {wm}")
    print(f"Identification: {ident}")
    
    print("\n--- Exercise 2: Facts that satisfy no identification rule ---")
    incomplete_facts = ["has hair", "has pointed teeth", "has claws", "has forward pointing eyes"]
    wm, seq, ident = forward_chaining(rules, incomplete_facts)
    print(f"Initial Facts: {incomplete_facts}")
    print(f"Firing Sequence: {seq}")
    print(f"Final Working Memory: {wm}")
    print(f"Identification: {ident}")
    
    print("\n--- Exercise 3: Modify engine to fire ALL applicable rules per cycle ---")
    # A set of facts that satisfies multiple rules simultaneously at the start (R1 and R2)
    multi_facts = ["has hair", "gives milk", "eats meat", "has tawny color", "has black stripes"]
    print(f"Initial Facts: {multi_facts}")
    
    print("\n[Firing FIRST rule per cycle (Default)]")
    wm1, seq1, ident1 = forward_chaining(rules, multi_facts, fire_all=False)
    print(f"Firing Sequence: {seq1}")
    
    print("\n[Firing ALL rules per cycle]")
    wm2, seq2, ident2 = forward_chaining(rules, multi_facts, fire_all=True)
    print(f"Firing Sequence: {seq2}")
    
    print("\n[Comparison Observations]")
    print("When firing only the first rule, the engine picks one rule per iteration (e.g., R1),")
    print("adds its conclusion to WM, and rebuilds the conflict set. This might make other rules")
    print("like R2 irrelevant if their conclusion ('is mammal') is already in the WM.")
    print("When firing ALL rules per cycle, the engine processes all satisfied rules in the current")
    print("conflict set simultaneously. Thus, both R1 and R2 fire in the first cycle because")
    print("both were satisfied before WM was updated.")

if __name__ == '__main__':
    run_exercises()
