"""
Experiment 8: Multi-Agent Task Allocation and Coordination
Aim: To implement a simple multi-agent task allocation system using Partial Global Planning (PGP)
and Contract Net Protocol (CNP).
"""

import matplotlib.pyplot as plt
import os

# ==========================================
# EXERCISE 1: Agent Setup
# ==========================================
class Agent:
    def __init__(self, name):
        self.name = name
        self.assigned_tasks = []
        self.total_cost = 0

class DeliveryAgent(Agent):
    def __init__(self, name, base_cost_multiplier):
        super().__init__(name)
        self.base_cost_multiplier = base_cost_multiplier
        self.local_plan = [] # For PGP

    def calculate_bid(self, task):
        # Bid = base task cost * agent's multiplier
        return round(task['base_cost'] * self.base_cost_multiplier, 2)

class ManagerAgent:
    def __init__(self, name):
        self.name = name
        
    def announce_task(self, task, agents):
        # Step 1 & 2: Task Announcement & Bidding
        bids = {}
        for agent in agents:
            bids[agent.name] = agent.calculate_bid(task)
        return bids
        
    def evaluate_bids_and_assign(self, task, bids, agents):
        # Step 3 & 4: Bid Comparison & Agent Selection
        best_agent_name = min(bids, key=bids.get)
        winning_bid = bids[best_agent_name]
        
        # Step 5: Task Assignment
        for agent in agents:
            if agent.name == best_agent_name:
                agent.assigned_tasks.append((task['name'], winning_bid))
                agent.total_cost += winning_bid
                return best_agent_name, winning_bid
        return None, None

def setup_environment():
    manager = ManagerAgent("Manager_1")
    
    # 3 Delivery Agents
    agents = [
        DeliveryAgent("Agent_A", 1.0),
        DeliveryAgent("Agent_B", 1.2),
        DeliveryAgent("Agent_C", 0.9)
    ]
    
    # 6 Delivery Tasks
    tasks = [
        {'name': 'Task_1_Downtown', 'base_cost': 50},
        {'name': 'Task_2_Suburbs', 'base_cost': 120},
        {'name': 'Task_3_Airport', 'base_cost': 80},
        {'name': 'Task_4_Industrial', 'base_cost': 200},
        {'name': 'Task_5_Mall', 'base_cost': 45},
        {'name': 'Task_6_Harbor', 'base_cost': 150}
    ]
    
    return manager, agents, tasks


# ==========================================
# EXERCISE 2: PGP Coordination
# ==========================================
def run_pgp_coordination():
    print("\n" + "="*60)
    print("EXERCISE 2: PGP Coordination")
    print("="*60)
    
    # Create local plans with a conflict
    # Format: (time_slot, location)
    agent_x = DeliveryAgent("Agent_X", 1.0)
    agent_y = DeliveryAgent("Agent_Y", 1.0)
    
    agent_x.local_plan = [(1, "Depot"), (2, "Loading_Dock_A"), (3, "Zone_B")]
    agent_y.local_plan = [(1, "Zone_C"), (2, "Loading_Dock_A"), (3, "Zone_D")]
    
    print(f"Agent X Initial Plan: {agent_x.local_plan}")
    print(f"Agent Y Initial Plan: {agent_y.local_plan}")
    
    # Detect conflict
    # Conflict: Both at "Loading_Dock_A" at time slot 2. (Assuming capacity 1)
    conflict_found = False
    for i in range(len(agent_x.local_plan)):
        for j in range(len(agent_y.local_plan)):
            time_x, loc_x = agent_x.local_plan[i]
            time_y, loc_y = agent_y.local_plan[j]
            
            if time_x == time_y and loc_x == loc_y:
                print(f"\nCONFLICT DETECTED: Both agents scheduled at '{loc_x}' at time {time_x}.")
                conflict_found = True
                
                # Resolve conflict by delaying Agent Y
                print("Resolving conflict: Delaying Agent Y's plan from time slot 2...")
                
                # Simple resolution: Wait at previous location for 1 time slot, shift rest
                delayed_plan = []
                for t, loc in agent_y.local_plan:
                    if t < time_y:
                        delayed_plan.append((t, loc))
                    elif t == time_y:
                        # Wait at previous location
                        prev_loc = agent_y.local_plan[j-1][1] if j > 0 else "Base"
                        delayed_plan.append((t, prev_loc))
                        delayed_plan.append((t+1, loc))
                    else:
                        delayed_plan.append((t+1, loc))
                        
                agent_y.local_plan = delayed_plan
                break
        if conflict_found:
            break
            
    print(f"\nAgent X Modified Plan: {agent_x.local_plan}")
    print(f"Agent Y Modified Plan: {agent_y.local_plan}")
    print("PGP Conflict successfully resolved by global coordination!")


# ==========================================
# EXERCISE 3 & 4: CNP Task Allocation
# ==========================================
def run_cnp_allocation(manager, agents, tasks):
    print("\n" + "="*60)
    print("EXERCISE 3 & 4: CNP Task Allocation")
    print("="*60)
    
    total_system_cost = 0
    allocation_results = []
    
    print("Starting CNP Process for all tasks...\n")
    
    for task in tasks:
        # Task Announcement & Bidding
        bids = manager.announce_task(task, agents)
        print(f"Task Announcement: {task['name']} (Base Cost/Time: {task['base_cost']})")
        print(f"Received Bids: {bids}")
        
        # Bid Comparison & Agent Selection
        best_agent, winning_bid = manager.evaluate_bids_and_assign(task, bids, agents)
        print(f"-> Assigned to {best_agent} for cost {winning_bid}\n")
        
        allocation_results.append({
            'Task': task['name'],
            'Selected Agent': best_agent,
            'Cost': winning_bid
        })
        total_system_cost += winning_bid
        
    print("-" * 50)
    print(f"{'Task':<20} | {'Selected Agent':<15} | {'Cost/Time'}")
    print("-" * 50)
    for res in allocation_results:
        print(f"{res['Task']:<20} | {res['Selected Agent']:<15} | {res['Cost']:.2f}")
    print("-" * 50)
    print(f"TOTAL SYSTEM COST/TIME: {total_system_cost:.2f}")
    
    return total_system_cost


# ==========================================
# EXERCISE 5: Modification & Visualization
# ==========================================
def run_modified_scenario_and_visualize(manager, original_agents, tasks, orig_total_cost):
    print("\n" + "="*60)
    print("EXERCISE 5: Modification & Visualization")
    print("="*60)
    
    print("Modifying parameters to observe changes in task allocation:")
    print("1. Agent C becomes much more expensive (multiplier 0.9 -> 1.5)")
    print("2. Agent A becomes slightly cheaper (multiplier 1.0 -> 0.8)")
    
    mod_agents = [
        DeliveryAgent("Agent_A", 0.8),
        DeliveryAgent("Agent_B", 1.2),
        DeliveryAgent("Agent_C", 1.5)
    ]
    
    # Run CNP again with modified agents
    print("\nRunning CNP with Modified Agents:")
    mod_total_cost = run_cnp_allocation(manager, mod_agents, tasks)
    
    # Collect data for plotting
    agents_names = [a.name for a in original_agents]
    orig_costs = [a.total_cost for a in original_agents]
    mod_costs = [a.total_cost for a in mod_agents]
    
    # Plotting using Matplotlib
    x = range(len(agents_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], orig_costs, width, label='Original Config', color='lightblue')
    rects2 = ax.bar([i + width/2 for i in x], mod_costs, width, label='Modified Config', color='salmon')
    
    ax.set_ylabel('Total Assigned Cost/Time')
    ax.set_title('Agent vs Total Assigned Cost/Time (CNP Allocation)')
    ax.set_xticks(x)
    ax.set_xticklabels(agents_names)
    ax.legend()
    
    # Annotate bars with values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, 'cnp_allocation_comparison.png')
    plt.savefig(plot_path)
    
    print(f"\nVisualization saved as '{plot_path}'")
    
    print("\nObservation:")
    print("- In the Original configuration, Agent C (multiplier 0.9) won all bids because it was the most efficient (cheapest).")
    print("- After modification, Agent A (multiplier 0.8) became the cheapest, winning all the tasks.")
    print("- Agent C received no tasks after its cost increased.")
    print(f"- The total system cost changed from {orig_total_cost:.2f} to {mod_total_cost:.2f}.")


def main():
    print("*" * 60)
    print("EXPERIMENT 8: Multi-Agent Task Allocation and Coordination")
    print("*" * 60)
    
    # Ex 1
    manager, agents, tasks = setup_environment()
    print("Exercise 1 Setup Complete: 1 Manager, 3 Delivery Agents, 6 Tasks defined.\n")
    
    # Ex 2
    run_pgp_coordination()
    
    # Ex 3 & 4
    orig_total_cost = run_cnp_allocation(manager, agents, tasks)
    
    # Ex 5
    run_modified_scenario_and_visualize(manager, agents, tasks, orig_total_cost)

if __name__ == "__main__":
    main()
