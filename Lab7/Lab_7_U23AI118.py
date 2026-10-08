"""
Experiment 7: Logical and Probabilistic Reasoning
Aim: To implement basic Propositional Logic, First-Order Logic, and probability-based reasoning using Python.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def exercise_1():
    """
    Propositional Logic
    Represent a simple rule such as "If it rains, then the road is wet".
    Generate its truth table and verify the result.
    """
    print("\n" + "=" * 70)
    print("EXERCISE 1 - PROPOSITIONAL LOGIC")
    print("=" * 70)
    
    print("Rule: If it rains (P), then the road is wet (Q).")
    print("Logical implication: P -> Q (Equivalent to 'not P or Q')")
    print()
    print(f"{'Rain (P)':<12} | {'Wet Road (Q)':<15} | {'P -> Q':<10}")
    print("-" * 45)
    
    # Truth values
    truth_values = [True, False]
    
    for P in truth_values:
        for Q in truth_values:
            # Implication P -> Q is logically equivalent to not P or Q
            implication = (not P) or Q
            print(f"{str(P):<12} | {str(Q):<15} | {str(implication):<10}")
            
    print("\nObservation: The rule only evaluates to False when it rains (P=True)")
    print("and the road is not wet (Q=False), which breaks the rule.")


def exercise_2():
    """
    First-Order Logic
    Represent the statements "All humans are mortal" and "Socrates is a human".
    Derive whether Socrates is mortal.
    """
    print("\n" + "=" * 70)
    print("EXERCISE 2 - FIRST-ORDER LOGIC")
    print("=" * 70)
    
    # Facts and rules represented as Python functions/data structures
    humans = ["Socrates", "Plato", "Aristotle"]
    
    def is_human(entity):
        """Predicate: human(x)"""
        return entity in humans
        
    def is_mortal(entity):
        """Rule: For all x, human(x) -> mortal(x)"""
        if is_human(entity):
            return True
        return False
        
    entity = "Socrates"
    print("Statement 1: All humans are mortal.")
    print(f"Statement 2: {entity} is a human.")
    print()
    
    # Inference
    if is_mortal(entity):
        print(f"Conclusion: Therefore, {entity} is mortal.")
    else:
        print(f"Conclusion: Cannot determine if {entity} is mortal.")
        

def exercise_3_and_4():
    """
    Probability of Events & Conditional Probability
    Calculate probabilities based on a dataset.
    """
    print("\n" + "=" * 70)
    print("EXERCISE 3 & 4 - PROBABILITY OF EVENTS & CONDITIONAL PROBABILITY")
    print("=" * 70)
    
    # Dataset representing 100 students
    total_students = 100
    
    # Contingency table (Attended vs Passed)
    attended_and_passed = 60
    attended_and_failed = 10
    not_attended_and_passed = 5
    not_attended_and_failed = 25
    
    # Marginal probabilities
    prob_attended = (attended_and_passed + attended_and_failed) / total_students
    prob_passed = (attended_and_passed + not_attended_and_passed) / total_students
    
    print("--- Exercise 3: Individual, Joint, and Union Probabilities ---")
    print(f"Dataset Size: {total_students} students")
    print(f"P(Attended) = {prob_attended:.2f}")
    print(f"P(Passed) = {prob_passed:.2f}")
    
    # Joint probability: P(Attended AND Passed)
    prob_attended_and_passed = attended_and_passed / total_students
    print(f"\nJoint Probability:")
    print(f"P(Attended AND Passed) = {prob_attended_and_passed:.2f}")
    
    # Union probability: P(Attended OR Passed) = P(Attended) + P(Passed) - P(Attended AND Passed)
    prob_attended_or_passed = prob_attended + prob_passed - prob_attended_and_passed
    print(f"\nUnion Probability:")
    print(f"P(Attended OR Passed) = {prob_attended_or_passed:.2f}")
    
    print("\n--- Exercise 4: Conditional Probabilities ---")
    # P(Passed | Attended) = P(Passed AND Attended) / P(Attended)
    prob_passed_given_attended = prob_attended_and_passed / prob_attended
    print(f"P(Passed | Attended) = {prob_passed_given_attended:.2f} "
          f"(Probability of passing given the student attended)")
    
    # P(Attended | Passed) = P(Attended AND Passed) / P(Passed)
    prob_attended_given_passed = prob_attended_and_passed / prob_passed
    print(f"P(Attended | Passed) = {prob_attended_given_passed:.2f} "
          f"(Probability that a student attended given they passed)")


def exercise_5():
    """
    Bayes' Theorem
    Implement a simple Bayes' theorem example (disease diagnosis).
    """
    print("\n" + "=" * 70)
    print("EXERCISE 5 - BAYES' THEOREM")
    print("=" * 70)
    
    # Disease diagnosis scenario
    # P(D) = Probability of having the disease (prior probability)
    # P(T|D) = True Positive Rate (Sensitivity)
    # P(T|~D) = False Positive Rate
    
    P_D = 0.01  # 1% of the population has the disease
    P_T_given_D = 0.99  # 99% chance of testing positive if you have it
    P_T_given_not_D = 0.05  # 5% chance of testing positive if you don't have it
    P_not_D = 1 - P_D
    
    # Total probability of testing positive P(T)
    P_T = (P_T_given_D * P_D) + (P_T_given_not_D * P_not_D)
    
    # Bayes' Theorem: P(D|T) = (P(T|D) * P(D)) / P(T)
    P_D_given_T = (P_T_given_D * P_D) / P_T
    
    print("Disease Diagnosis Scenario:")
    print(f"Prior probability of disease, P(D) = {P_D}")
    print(f"Sensitivity (True Positive Rate), P(T|D) = {P_T_given_D}")
    print(f"False Positive Rate, P(T|~D) = {P_T_given_not_D}")
    print(f"Probability of a positive test, P(T) = {P_T:.4f}")
    
    print(f"\nBayes' Theorem calculation:")
    print(f"P(D|T) = (P(T|D) * P(D)) / P(T)")
    print(f"Probability of having the disease given a positive test result:")
    print(f"P(D|T) = {P_D_given_T:.4f} ({P_D_given_T*100:.2f}%)")
    
    return P_D, P_D_given_T


def exercise_6(P_D_original, P_D_given_T_original):
    """
    Visualization
    Use Matplotlib to create a graph showing probabilities and modify inputs.
    """
    print("\n" + "=" * 70)
    print("EXERCISE 6 - VISUALIZATION")
    print("=" * 70)
    
    # Constants from the Bayes' theorem example
    P_T_given_D = 0.99
    P_T_given_not_D = 0.05
    
    # Modified scenario: increase prior probability P(D) to 10%
    P_D_modified = 0.10
    P_not_D_modified = 1 - P_D_modified
    P_T_modified = (P_T_given_D * P_D_modified) + (P_T_given_not_D * P_not_D_modified)
    P_D_given_T_modified = (P_T_given_D * P_D_modified) / P_T_modified
    
    print(f"Original Scenario: Prior P(D) = {P_D_original:.2f} -> Posterior P(D|T) = {P_D_given_T_original:.4f}")
    print(f"Modified Scenario: Prior P(D) = {P_D_modified:.2f} -> Posterior P(D|T) = {P_D_given_T_modified:.4f}")
    print("\nObservation:")
    print("When the prevalence of the disease (prior probability) increases,")
    print("the confidence that a positive test means the person actually has")
    print("the disease (posterior probability) increases significantly.")
    
    # Plotting
    labels = ['Prior P(D)', 'Posterior P(D|T)']
    original_values = [P_D_original, P_D_given_T_original]
    modified_values = [P_D_modified, P_D_given_T_modified]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, original_values, width, label='Original (P(D)=0.01)', color='skyblue')
    rects2 = ax.bar(x + width/2, modified_values, width, label='Modified (P(D)=0.10)', color='salmon')
    
    ax.set_ylabel('Probability')
    ax.set_title("Effect of Prior Probability on Posterior Probability\n(Bayes' Theorem)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'bayes_theorem_visualization.png')
    plt.savefig(output_path)
    print(f"\nVisualization saved as '{output_path}'")


def main():
    print("*" * 70)
    print("EXPERIMENT 7: LOGICAL AND PROBABILISTIC REASONING")
    print("*" * 70)
    
    exercise_1()
    exercise_2()
    exercise_3_and_4()
    p_d, p_d_given_t = exercise_5()
    exercise_6(p_d, p_d_given_t)
    
    print("\n" + "*" * 70)
    print("ALL EXERCISES COMPLETED")
    print("*" * 70)


if __name__ == "__main__":
    main()
