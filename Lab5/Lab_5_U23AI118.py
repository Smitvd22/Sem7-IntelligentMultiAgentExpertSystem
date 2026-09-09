"""
Experiment 4: Backward Chaining Engine with Askable Facts
Car-Fault Diagnosis - AUTOMATED VERSION

No user input is required.
The program automatically supplies YES/NO answers for the facts.

Exercises:
1. Starter motor fault using clicking sound + engine not cranking.
2. Change hypothesis order and compare the questions asked.
3. WHY option: automatically displays why each question was asked.
"""


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


def get_rules():
    """Knowledge base containing car-fault IF-THEN rules."""
    return [
        Rule("R1", "battery problem",
             ["headlights dim", "battery warning light"]),

        Rule("R2", "alternator problem",
             ["battery warning light", "engine stalls"]),

        Rule("R3", "fuel problem",
             ["engine cranks", "engine does not start"]),

        Rule("R4", "ignition problem",
             ["engine cranks", "no spark"]),

        # Exercise 1
        Rule("R5", "starter motor problem",
             ["clicking sound", "engine not cranking"]),
    ]


class BackwardChainingEngine:
    def __init__(self, rules, known_answers, hypothesis_order):
        self.rules = rules
        self.known_answers = known_answers
        self.hypothesis_order = hypothesis_order

        # Facts already confirmed or rejected.
        self.facts = {}
        self.denied_facts = set()

        # For displaying what happened.
        self.question_log = []
        self.why_log = []
        self.proof = []

    def find_rules_for_goal(self, goal):
        return [r for r in self.rules if r.conclusion == goal]

    def ask_fact(self, fact, rule):
        """
        Automated replacement for asking the user.

        The answer comes from known_answers instead of input().
        The WHY explanation is also displayed automatically.
        """
        if fact in self.facts:
            return self.facts[fact]

        if fact in self.denied_facts:
            return False

        # Automated answer
        answer = self.known_answers.get(fact, False)

        print(f"Question: Is it true that '{fact}'?")
        print(f"Automated answer: {'YES' if answer else 'NO'}")

        # Exercise 3: WHY explanation
        reason = (
            f"WHY -> {rule.rule_id} is asking about '{fact}' "
            f"because the engine is trying to prove "
            f"'{rule.conclusion}'. The rule requires: "
            + " AND ".join(f"'{c}'" for c in rule.conditions)
            + "."
        )
        print(reason)
        self.why_log.append((fact, rule.rule_id, reason))

        self.question_log.append((fact, answer, rule.rule_id))

        if answer:
            self.facts[fact] = True
        else:
            self.denied_facts.add(fact)

        return answer

    def prove_goal(self, goal, visited=None):
        """Backward chaining: Goal -> rule -> conditions -> goal."""
        if visited is None:
            visited = set()

        if self.facts.get(goal, False):
            return True

        if goal in visited:
            return False

        visited = visited | {goal}

        rules = self.find_rules_for_goal(goal)

        for rule in rules:
            conditions_satisfied = True

            for condition in rule.conditions:
                # If another rule can prove this condition, recursively
                # use backward chaining.
                if self.find_rules_for_goal(condition):
                    condition_true = self.prove_goal(condition, visited)
                else:
                    condition_true = self.ask_fact(condition, rule)

                if not condition_true:
                    conditions_satisfied = False
                    break

            if conditions_satisfied:
                self.facts[goal] = True
                self.proof.append(
                    (rule.rule_id, rule.conclusion, rule.conditions)
                )
                return True

        return False

    def diagnose(self):
        """Try possible faults in the selected hypothesis order."""
        for hypothesis in self.hypothesis_order:
            print(f"\nTrying hypothesis: {hypothesis}")

            if self.prove_goal(hypothesis):
                print(f"DIAGNOSIS: {hypothesis}")
                return hypothesis

        print("DIAGNOSIS: No fault could be identified.")
        return None

    def show_summary(self):
        print("\nQuestions asked:")
        if not self.question_log:
            print("None")
        else:
            for i, (fact, answer, rule_id) in enumerate(
                self.question_log, 1
            ):
                print(
                    f"{i}. {fact} -> "
                    f"{'YES' if answer else 'NO'} "
                    f"(Rule {rule_id})"
                )

        print("\nSuccessful rule chain:")
        if not self.proof:
            print("None")
        else:
            for rule_id, conclusion, conditions in self.proof:
                print(
                    f"{rule_id}: "
                    f"{' AND '.join(conditions)} "
                    f"-> {conclusion}"
                )


def create_starter_motor_answers():
    """
    Facts for Exercise 1.

    These are exactly the two required observations from the question:
    clicking sound = YES
    engine not cranking = YES
    """
    return {
        "clicking sound": True,
        "engine not cranking": True,

        # Other possible facts are not needed for this diagnosis.
        "headlights dim": False,
        "battery warning light": False,
        "engine stalls": False,
        "engine cranks": False,
        "engine does not start": False,
        "no spark": False,
    }


def exercise_1():
    print("\n" + "=" * 70)
    print("EXERCISE 1 - STARTER MOTOR FAULT")
    print("=" * 70)

    rules = get_rules()
    answers = create_starter_motor_answers()

    order = [
        "starter motor problem",
        "battery problem",
        "alternator problem",
        "fuel problem",
        "ignition problem",
    ]

    print("\nGiven facts are automatically supplied:")
    print("- clicking sound = YES")
    print("- engine not cranking = YES")

    engine = BackwardChainingEngine(rules, answers, order)
    diagnosis = engine.diagnose()
    engine.show_summary()

    print("\nExpected result: starter motor problem")
    return diagnosis


def run_with_order(order, answers):
    """Run one automated diagnosis and return its question sequence."""
    rules = get_rules()
    engine = BackwardChainingEngine(rules, answers, order)
    diagnosis = engine.diagnose()

    questions = [q[0] for q in engine.question_log]

    return diagnosis, questions


def exercise_2():
    print("\n" + "=" * 70)
    print("EXERCISE 2 - CHANGE HYPOTHESIS ORDER")
    print("=" * 70)

    rules = get_rules()
    answers = create_starter_motor_answers()

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

    print("\n--- ORDER 1 ---")
    print(" -> ".join(order_1))
    diagnosis1, questions1 = run_with_order(order_1, answers)

    print("\nQuestions asked in Order 1:")
    for q in questions1:
        print(f"- {q}")

    print("\n--- ORDER 2 ---")
    print(" -> ".join(order_2))
    diagnosis2, questions2 = run_with_order(order_2, answers)

    print("\nQuestions asked in Order 2:")
    for q in questions2:
        print(f"- {q}")

    print("\nOBSERVATION:")
    print(
        "Changing the hypothesis order changes which fault is checked first. "
        "Therefore, the sequence of questions can change."
    )
    print(
        "Both orders can still identify the correct starter motor fault "
        "once the required facts are available."
    )

    return diagnosis1, diagnosis2


def exercise_3():
    print("\n" + "=" * 70)
    print("EXERCISE 3 - AUTOMATIC WHY OPTION")
    print("=" * 70)

    print(
        "\nThe program automatically prints WHY for every question.\n"
        "This shows which rule caused the question and what goal is being "
        "proved."
    )

    rules = get_rules()
    answers = create_starter_motor_answers()

    order = [
        "starter motor problem",
        "battery problem",
        "alternator problem",
        "fuel problem",
        "ignition problem",
    ]

    engine = BackwardChainingEngine(rules, answers, order)
    diagnosis = engine.diagnose()

    print("\nWHY SUMMARY:")
    for fact, rule_id, reason in engine.why_log:
        print(f"- {fact}: asked because of {rule_id}")

    engine.show_summary()

    return diagnosis


def main():
    print("=" * 70)
    print("EXPERIMENT 4: BACKWARD CHAINING ENGINE")
    print("AUTOMATED CAR-FAULT DIAGNOSIS")
    print("=" * 70)

    print(
        """
This version needs NO user input.

The program automatically:
1. Provides the required facts.
2. Performs backward chaining.
3. Identifies the starter motor fault.
4. Changes hypothesis order for Exercise 2.
5. Displays WHY explanations for Exercise 3.
"""
    )

    exercise_1()
    exercise_2()
    exercise_3()

    print("\n" + "=" * 70)
    print("ALL 3 EXERCISES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
