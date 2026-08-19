import random
import math

class Hole:
    def __init__(self, x, y, lifetime):
        self.x = x
        self.y = y
        self.lifetime = lifetime

class BDI_Agent:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.intentions = None
        self.plan = []

def distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def run_simulation(gamma, world_speed, seed, steps=600, exercise=1):
    random.seed(seed)
    
    agent = BDI_Agent(7, 7) # Start in middle
    holes = []
    
    holes_filled = 0
    holes_appeared = 0
    
    # World parameters
    appearance_prob = 0.1
    max_lifetime = 40
    
    battery = 40
    charger_x, charger_y = 7, 7
    stranded_steps = 0
    
    # For bold agent, gamma exceeds max plan length. Max distance is 30.
    if gamma == 'bold':
        gamma = 100
        
    s = 0 # steps since last deliberation
    
    for step in range(steps):
        # 1. World advances (multiple ticks per agent action)
        for _ in range(world_speed):
            # Expire holes
            for h in list(holes):
                h.lifetime -= 1
                if h.lifetime <= 0:
                    holes.remove(h)
                    
            # New hole?
            if random.random() < appearance_prob:
                hx, hy = random.randint(0, 14), random.randint(0, 14)
                # Check if hole already exists there
                if not any(h.x == hx and h.y == hy for h in holes):
                    holes.append(Hole(hx, hy, random.randint(10, max_lifetime)))
                    holes_appeared += 1

        if exercise == 2:
            if battery <= 0 and (agent.x != charger_x or agent.y != charger_y):
                stranded_steps += 1
                continue # Stranded
            
            if agent.x == charger_x and agent.y == charger_y:
                battery = 40 # Recharge
                
        # 2. Re-perceive and revise beliefs (holes array is the belief)
        
        # 3. Deliberation
        deliberate = False
        if agent.intentions is None or not agent.plan:
            deliberate = True
        else:
            s += 1
            if s >= gamma:
                deliberate = True
                
        if deliberate:
            s = 0
            # Charge one agent time step for deliberation (meaning we skip acting this step?)
            # The prompt says "charge one agent time step to deliberation, allowing the world to advance while the agent reasons."
            # So if we deliberate, we don't act this step. We just pick new intention.
            
            # Filter options
            options = []
            for h in holes:
                dist = distance(agent.x, agent.y, h.x, h.y)
                options.append((dist, h))
                
            if exercise == 1:
                options.sort(key=lambda x: x[0]) # Closest first
                
                if options:
                    best_option = options[0][1]
                    agent.intentions = best_option
                    # Plan is path to hole
                    agent.plan = []
                    cx, cy = agent.x, agent.y
                    while cx != best_option.x or cy != best_option.y:
                        if cx < best_option.x: cx += 1
                        elif cx > best_option.x: cx -= 1
                        elif cy < best_option.y: cy += 1
                        elif cy > best_option.y: cy -= 1
                        agent.plan.append((cx, cy))
                else:
                    agent.intentions = None
                    agent.plan = []
            elif exercise == 2:
                # Resource bounded filter
                valid_options = []
                for dist, h in options:
                    dist_to_charger = distance(h.x, h.y, charger_x, charger_y)
                    if dist + dist_to_charger <= battery:
                        valid_options.append((dist, h))
                        
                valid_options.sort(key=lambda x: x[0])
                
                if valid_options:
                    best_option = valid_options[0][1]
                    agent.intentions = best_option
                    # Plan to hole
                    agent.plan = []
                    cx, cy = agent.x, agent.y
                    while cx != best_option.x or cy != best_option.y:
                        if cx < best_option.x: cx += 1
                        elif cx > best_option.x: cx -= 1
                        elif cy < best_option.y: cy += 1
                        elif cy > best_option.y: cy -= 1
                        agent.plan.append((cx, cy))
                else:
                    # Recharge
                    agent.intentions = 'recharge'
                    agent.plan = []
                    cx, cy = agent.x, agent.y
                    while cx != charger_x or cy != charger_y:
                        if cx < charger_x: cx += 1
                        elif cx > charger_x: cx -= 1
                        elif cy < charger_y: cy += 1
                        elif cy > charger_y: cy -= 1
                        agent.plan.append((cx, cy))
            
            # Since we deliberated, we skip action this step
            continue
            
        # 4. Execute action
        if agent.plan:
            next_pos = agent.plan.pop(0)
            agent.x, agent.y = next_pos
            if exercise == 2:
                battery -= 1
                
        # 5. Check if intention achieved or unachievable
        if exercise == 1 or agent.intentions != 'recharge':
            if agent.intentions not in holes: # Unachievable
                agent.intentions = None
                agent.plan = []
            elif agent.x == agent.intentions.x and agent.y == agent.intentions.y: # Achieved
                holes.remove(agent.intentions)
                holes_filled += 1
                agent.intentions = None
                agent.plan = []
        elif exercise == 2 and agent.intentions == 'recharge':
            if agent.x == charger_x and agent.y == charger_y:
                agent.intentions = None
                agent.plan = []
                battery = 40
                
    effectiveness = holes_filled / holes_appeared if holes_appeared > 0 else 0
    return effectiveness, stranded_steps

def solve_exercise_1():
    print("--- Exercise 1: Boldness against dynamism ---")
    gammas = [1, 2, 4, 8, 'bold']
    speeds = [1, 2, 4, 8]
    
    results = {}
    
    for speed in speeds:
        print(f"\nWorld Speed: {speed}")
        best_gamma = None
        max_eff = -1
        
        for g in gammas:
            eff_sum = 0
            for seed in range(25):
                eff, _ = run_simulation(g, speed, seed, exercise=1)
                eff_sum += eff
            avg_eff = eff_sum / 25
            results[(speed, g)] = avg_eff
            print(f"  Gamma: {g}, Avg Effectiveness: {avg_eff:.4f}")
            
            if avg_eff > max_eff:
                max_eff = avg_eff
                best_gamma = g
                
        print(f"  => Best gamma for speed {speed} is {best_gamma}")

def solve_exercise_2():
    print("\n--- Exercise 2: A resource-bounded filter ---")
    # Compare BDI vs battery-blind
    
    # BDI
    eff_sum = 0
    stranded_sum = 0
    for seed in range(25):
        eff, stranded = run_simulation(1, 2, seed, exercise=2)
        eff_sum += eff
        stranded_sum += stranded
    
    print(f"Resource-Bounded Agent (Gamma=1, Speed=2):")
    print(f"  Avg Effectiveness: {eff_sum/25:.4f}")
    print(f"  Avg Stranded Steps: {stranded_sum/25:.1f}")
    print("\nThe check for battery belongs in the filter component (options generation / filtering) because it determines whether a desire (filling a hole) should become an intention based on current beliefs (battery level and distance). The planner assumes the goal is achievable and just finds the shortest path.")

if __name__ == '__main__':
    solve_exercise_1()
    solve_exercise_2()
