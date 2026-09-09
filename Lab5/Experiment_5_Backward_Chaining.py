"""
Experiment 4: Backward Chaining Engine with Askable Facts
Car-Fault Diagnosis

Exercises implemented:
1. Starter motor fault using:
   - clicking sound
   - engine not cranking
2. Change hypothesis order and observe changed questions.
3. Add WHY option showing the rule responsible for the question.

Run:
    python Experiment_4_Backward_Chaining.py
"""

# ------------------------------------------------------------
# Rule representation
# ------------------------------------------------------------

class Rule:
    def __init__(self, rule_id, conclusion, conditions):
        self.rule_id = rule_id
        self.conclusion = conclusion
        self.conditions = conditions

    def __str__(self):
        return (
            f"{self.rule_id}: IF "
            + " AND ".join(self.conditions)
            + f" THEN {self.conclusion}"
        )


# ------------------------------------------------------------
# Knowledge base
# ------------------------------------------------------------

def get_rules():
    """
    Rules for simple car-fault diagnosis.

    The first four faults represent common examples.
    The starter motor rule is the rule required by Exercise 1.
    """
    return [
        Rule("R1", "battery problem",
             ["headlights dim", "battery warning light"]),

        Rule("R2", "alternator problem",
             ["battery warning light", "engine stalls"]),

        Rule("R3", "fuel problem",
             ["engine cranks", "engine does not start"]),

        Rule("R4", "ignition problem",
             ["engine cranks", "no spark"]),

        # Exercise 1: required starter motor fault
        Rule("R5", "starter motor problem",
             ["clicking sound", "engine not cranking"]),
    ]


# ------------------------------------------------------------
# Backward chaining engine
# ------------------------------------------------------------

class BackwardChainingEngine:
    def __init__(self, rules, hypothesis_order=None):
        self.rules = rules

        # Rules/hypotheses are tried in this order.
        self.hypothesis_order = (
            hypothesis_order
            if hypothesis_order is not None
            else [rule.conclusion for rule in rules]
        )

        # Facts entered/confirmed by the user.
        self.facts = {}

        # Facts that the user has explicitly denied.
        self.denied_facts = set()

        # Avoid asking the same question repeatedly.
        self.question_log = []

        # Stores the rule responsible for the current question.
        self.current_question_rule = None

        # Stores WHY explanations.
        self.why_log = []

        # Stores successful reasoning paths.
        self.proof = []

    def find_rules_for_goal(self, goal):
        """Return rules that can prove the given goal."""
        return [
            rule for rule in self.rules
            if rule.conclusion == goal
        ]

    def ask_fact(self, fact, rule):
        """
        Ask the user for an unknown fact.

        The user can enter:
          y / yes
          n / no
          why
          q / quit
        """
        if fact in self.facts:
            return self.facts[fact]

        if fact in self.denied_facts:
            return False

        self.current_question_rule = rule

        while True:
            print()
            print(f"Question: Is it true that '{fact}'?")
            print("Enter: y = Yes, n = No, WHY = why is this asked, q = quit")

            answer = input("Your answer: ").strip().lower()

            if answer in ("y", "yes"):
                self.facts[fact] = True
                self.question_log.append((fact, True, rule.rule_id))
                return True

            if answer in ("n", "no"):
                self.denied_facts.add(fact)
                self.question_log.append((fact, False, rule.rule_id))
                return False

            if answer == "why":
                reason = (
                    f"{rule.rule_id} is being checked because the engine is "
                    f"trying to prove '{rule.conclusion}'. "
                    f"That rule requires: "
                    + " AND ".join(f"'{c}'" for c in rule.conditions)
                    + "."
                )
                print()
                print("WHY:")
                print(reason)
                self.why_log.append((fact, rule.rule_id, reason))
                continue

            if answer == "q":
                raise SystemExit("Diagnosis stopped by the user.")

            print("Please enter y, n, WHY, or q.")

    def prove_goal(self, goal, visited=None):
        """
        Backward chaining:
        Goal -> find rule -> prove conditions -> prove goal.
        """
        if visited is None:
            visited = set()

        # If the goal is already known as true, it is proved.
        if self.facts.get(goal, False):
            return True

        # Prevent circular reasoning.
        if goal in visited:
            return False

        visited = visited | {goal}

        rules = self.find_rules_for_goal(goal)

        for rule in rules:
            all_conditions_true = True

            for condition in rule.conditions:
                # If a condition is itself a conclusion of a rule,
                # try to prove it using backward chaining.
                sub_rules = self.find_rules_for_goal(condition)

                if sub_rules:
                    condition_true = self.prove_goal(condition, visited)
                else:
                    condition_true = self.ask_fact(condition, rule)

                if not condition_true:
                    all_conditions_true = False
                    break

            if all_conditions_true:
                self.facts[goal] = True
                self.proof.append((rule.rule_id, rule.conclusion, rule.conditions))
                return True

        return False

    def diagnose(self):
        """
        Try each fault hypothesis in the selected order.
        """
        for hypothesis in self.hypothesis_order:
            print()
            print("=" * 60)
            print(f"Trying hypothesis: {hypothesis}")
            print("=" * 60)

            if self.prove_goal(hypothesis):
                print()
                print("DIAGNOSIS FOUND")
                print(f"Fault: {hypothesis}")
                return hypothesis

            print(f"Could not establish: {hypothesis}")

        print()
        print("NO FAULT COULD BE IDENTIFIED FROM THE GIVEN FACTS.")
        return None

    def show_summary(self):
        """Display what the engine asked and how it reasoned."""
        print()
        print("=" * 60)
        print("REASONING SUMMARY")
        print("=" * 60)

        print("\nQuestions asked:")
        if not self.question_log:
            print("No questions were asked.")
        else:
            for i, (fact, answer, rule_id) in enumerate(self.question_log, 1):
                result = "YES" if answer else "NO"
                print(f"{i}. {fact} -> {result} (because of {rule_id})")

        print("\nRules that successfully fired:")
        if not self.proof:
            print("None")
        else:
            for rule_id, conclusion, conditions in self.proof:
                print(
                    f"{rule_id}: "
                    f"{' AND '.join(conditions)} -> {conclusion}"
                )

        if self.why_log:
            print("\nWHY explanations used:")
            for fact, rule_id, reason in self.why_log:
                print(f"- Question '{fact}' was caused by {rule_id}.")
                print(f"  {reason}")


# ------------------------------------------------------------
# Exercise 1
# ------------------------------------------------------------

def exercise_1():
    """
    Starter motor diagnosis.

    Expected path:
        starter motor problem
        -> clicking sound
        -> engine not cranking
        -> starter motor problem
    """
    print("\n" + "#" * 60)
    print("EXERCISE 1: STARTER MOTOR FAULT")
    print("#" * 60)

    rules = get_rules()

    # Put starter motor hypothesis first so it is tested immediately.
    order = [
        "starter motor problem",
        "battery problem",
        "alternator problem",
        "fuel problem",
        "ignition problem",
    ]

    engine = BackwardChainingEngine(rules, order)
    diagnosis = engine.diagnose()
    engine.show_summary()

    return diagnosis


# ------------------------------------------------------------
# Exercise 2
# ------------------------------------------------------------

def exercise_2():
    """
    Compare two hypothesis orders.

    The same rules are used, but the order in which possible faults
    are tested is changed. This changes which questions are asked.
    """
    print("\n" + "#" * 60)
    print("EXERCISE 2: CHANGE HYPOTHESIS ORDER")
    print("#" * 60)

    rules = get_rules()

    order_1 = [
        "starter motor problem",
        "battery problem",
        "alternator problem",
        "fuel problem",
        "ignition problem",
    ]

    order_2 = [
        "fuel problem",
        "ignition problem",
        "battery problem",
        "starter motor problem",
        "alternator problem",
    ]

    print("\nORDER 1:")
    print(" -> ".join(order_1))
    print("The engine will try these hypotheses in this order.")

    print("\nORDER 2:")
    print(" -> ".join(order_2))
    print("The engine will try these hypotheses in this new order.")

    print(
        "\nIMPORTANT OBSERVATION:\n"
        "Changing the hypothesis order changes which rule is considered "
        "first, so the user may be asked different questions before the "
        "correct fault is found."
    )

    # Actually run both orders.
    print("\n--- Running Order 1 ---")
    engine1 = BackwardChainingEngine(rules, order_1)
    diagnosis1 = engine1.diagnose()
    engine1.show_summary()

    print("\n--- Running Order 2 ---")
    engine2 = BackwardChainingEngine(rules, order_2)
    diagnosis2 = engine2.diagnose()
    engine2.show_summary()

    return diagnosis1, diagnosis2


# ------------------------------------------------------------
# Exercise 3
# ------------------------------------------------------------

def exercise_3():
    """
    Demonstrate the WHY option.

    The WHY option is already part of ask_fact().
    """
    print("\n" + "#" * 60)
    print("EXERCISE 3: WHY OPTION")
    print("#" * 60)

    print(
        """
When the engine asks a question, type WHY instead of Y/N.

Example:
    Question: Is it true that 'clicking sound'?
    Your answer: WHY

The engine explains which rule caused the question and
what conclusion it is trying to prove.
"""
    )

    rules = get_rules()

    order = [
        "starter motor problem",
        "battery problem",
        "alternator problem",
        "fuel problem",
        "ignition problem",
    ]

    engine = BackwardChainingEngine(rules, order)
    diagnosis = engine.diagnose()
    engine.show_summary()

    return diagnosis


# ------------------------------------------------------------
# Menu
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("EXPERIMENT 4: BACKWARD CHAINING ENGINE")
    print("Car-Fault Diagnosis with Askable Facts")
    print("=" * 60)

    print(
        """
Choose an exercise:

1. Exercise 1 - Add/test starter motor fault
2. Exercise 2 - Change hypothesis order
3. Exercise 3 - Use WHY option
4. Run all exercises
5. Exit
"""
    )

    while True:
        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            exercise_1()
            break

        elif choice == "2":
            exercise_2()
            break

        elif choice == "3":
            exercise_3()
            break

        elif choice == "4":
            exercise_1()
            exercise_2()
            exercise_3()
            break

        elif choice == "5":
            print("Exiting.")
            break

        else:
            print("Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
