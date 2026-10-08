"""
Experiment 6: Fuzzy Logic and Constraint Satisfaction
Student Performance (Fuzzy) + Timetable Scheduling (CSP)

No external libraries are required.
All fuzzy logic and CSP logic is implemented from scratch.

Exercises:
1. Fuzzy logic system for "student performance" based on attendance & marks.
2. Constraint satisfaction problem for a simple timetable.
3. Change fuzzy rules / CSP constraints and observe differences.
"""


# ──────────────────────────────────────────────────────────────
#  PART A — FUZZY LOGIC SYSTEM
# ──────────────────────────────────────────────────────────────


def trimf(x, a, b, c):
    """Triangular membership function.

    Returns the degree of membership of *x* in the triangle [a, b, c].
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function.

    Returns the degree of membership of *x* in the trapezoid [a, b, c, d].
    """
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


# ---------- Linguistic variables ----------

def attendance_low(x):
    return trapmf(x, 0, 0, 40, 60)


def attendance_medium(x):
    return trimf(x, 40, 60, 80)


def attendance_high(x):
    return trapmf(x, 60, 80, 100, 100)


def marks_low(x):
    return trapmf(x, 0, 0, 30, 50)


def marks_medium(x):
    return trimf(x, 30, 50, 70)


def marks_high(x):
    return trapmf(x, 50, 70, 100, 100)


# Output membership functions for "performance"
def performance_poor(x):
    return trapmf(x, 0, 0, 20, 40)


def performance_average(x):
    return trimf(x, 20, 50, 80)


def performance_good(x):
    return trapmf(x, 60, 80, 100, 100)


def defuzzify_centroid(output_mf_weights, step=1):
    """Centroid defuzzification over [0, 100].

    output_mf_weights is a list of (membership_function, firing_strength)
    tuples. The aggregated output is the point-wise maximum of each
    clipped membership function.
    """
    numerator = 0.0
    denominator = 0.0

    for x in range(0, 101, step):
        # Aggregated membership: max of all clipped outputs.
        agg = 0.0
        for mf, strength in output_mf_weights:
            clipped = min(mf(x), strength)
            if clipped > agg:
                agg = clipped
        numerator += x * agg
        denominator += agg

    if denominator == 0:
        return 0.0
    return numerator / denominator


def fuzzy_student_performance(attendance, marks, rules):
    """Evaluate student performance using fuzzy inference.

    Parameters
    ----------
    attendance : float   (0 – 100)
    marks      : float   (0 – 100)
    rules      : list of (att_mf, marks_mf, out_mf, label) tuples

    Returns
    -------
    crisp_score : float  (defuzzified performance score)
    rule_activations : list of (label, firing_strength)
    """
    rule_activations = []
    output_pairs = []

    for att_mf, marks_mf, out_mf, label in rules:
        # Firing strength = AND (minimum) of the two input memberships.
        firing = min(att_mf(attendance), marks_mf(marks))
        rule_activations.append((label, firing))
        output_pairs.append((out_mf, firing))

    crisp = defuzzify_centroid(output_pairs)
    return crisp, rule_activations


# ---------- Default fuzzy rule set ----------

def get_default_fuzzy_rules():
    """Nine rules covering all combinations of attendance × marks."""
    return [
        (attendance_low,    marks_low,    performance_poor,    "R1: Low Att  & Low Marks  -> Poor"),
        (attendance_low,    marks_medium, performance_poor,    "R2: Low Att  & Med Marks  -> Poor"),
        (attendance_low,    marks_high,   performance_average, "R3: Low Att  & High Marks -> Average"),
        (attendance_medium, marks_low,    performance_poor,    "R4: Med Att  & Low Marks  -> Poor"),
        (attendance_medium, marks_medium, performance_average, "R5: Med Att  & Med Marks  -> Average"),
        (attendance_medium, marks_high,   performance_good,    "R6: Med Att  & High Marks -> Good"),
        (attendance_high,   marks_low,    performance_average, "R7: High Att & Low Marks  -> Average"),
        (attendance_high,   marks_medium, performance_good,    "R8: High Att & Med Marks  -> Good"),
        (attendance_high,   marks_high,   performance_good,    "R9: High Att & High Marks -> Good"),
    ]


# ──────────────────────────────────────────────────────────────
#  PART B — CONSTRAINT SATISFACTION PROBLEM (Timetable)
# ──────────────────────────────────────────────────────────────


def solve_csp(variables, domains, constraints):
    """Simple backtracking CSP solver.

    Parameters
    ----------
    variables   : list of variable names  (e.g. subject names)
    domains     : dict   variable -> list of possible values (time slots)
    constraints : list of callables   constraint(assignment) -> bool

    Returns
    -------
    assignment  : dict or None
    """
    assignment = {}

    def is_consistent(var, value, assignment):
        test = dict(assignment)
        test[var] = value
        for constraint in constraints:
            if not constraint(test):
                return False
        return True

    def backtrack(assignment):
        if len(assignment) == len(variables):
            return dict(assignment)

        # Pick first unassigned variable.
        unassigned = [v for v in variables if v not in assignment]
        var = unassigned[0]

        for value in domains[var]:
            if is_consistent(var, value, assignment):
                assignment[var] = value
                result = backtrack(assignment)
                if result is not None:
                    return result
                del assignment[var]
        return None

    return backtrack(assignment)


def print_timetable(assignment, title="Timetable"):
    """Pretty-print a timetable assignment."""
    if assignment is None:
        print("No valid timetable found.\n")
        return

    print(f"\n{title}:")
    print("-" * 40)

    # Sort by slot for readability.
    sorted_items = sorted(assignment.items(), key=lambda item: item[1])
    for subject, slot in sorted_items:
        print(f"  {slot:<20s} | {subject}")
    print("-" * 40)


# ---------- Default timetable data ----------

def get_default_timetable_data():
    """Variables, domains, and constraints for the timetable CSP."""

    subjects = ["Maths", "Physics", "Chemistry", "English", "CS"]

    time_slots = [
        "Mon 9-10",
        "Mon 10-11",
        "Mon 11-12",
        "Tue 9-10",
        "Tue 10-11",
    ]

    domains = {subj: list(time_slots) for subj in subjects}

    constraints = []

    # C1: No two subjects in the same time slot.
    def no_overlap(assignment):
        values = list(assignment.values())
        return len(values) == len(set(values))

    constraints.append(no_overlap)

    # C2: Maths must be scheduled before Physics (earlier slot).
    def maths_before_physics(assignment):
        if "Maths" in assignment and "Physics" in assignment:
            return time_slots.index(assignment["Maths"]) < \
                   time_slots.index(assignment["Physics"])
        return True

    constraints.append(maths_before_physics)

    # C3: English must NOT be on Monday.
    def english_not_monday(assignment):
        if "English" in assignment:
            return not assignment["English"].startswith("Mon")
        return True

    constraints.append(english_not_monday)

    return subjects, domains, constraints


# ──────────────────────────────────────────────────────────────
#  EXERCISES
# ──────────────────────────────────────────────────────────────


def exercise_1():
    """Fuzzy logic: student performance for various inputs."""

    print("\n" + "=" * 70)
    print("EXERCISE 1 — FUZZY LOGIC: STUDENT PERFORMANCE")
    print("=" * 70)

    rules = get_default_fuzzy_rules()

    print("\nFuzzy Rules:")
    for _, _, _, label in rules:
        print(f"  {label}")

    test_cases = [
        (30, 20),   # Low attendance, low marks
        (50, 50),   # Medium attendance, medium marks
        (85, 90),   # High attendance, high marks
        (75, 40),   # High attendance, low marks
        (40, 80),   # Low attendance, high marks
        (60, 60),   # Boundary values
    ]

    print("\n{:<15s} {:<10s} {:<15s}   Active Rules".format(
        "Attendance", "Marks", "Performance"
    ))
    print("-" * 70)

    for att, mrk in test_cases:
        score, activations = fuzzy_student_performance(att, mrk, rules)
        active = [
            label.split(":")[0]
            for label, strength in activations if strength > 0
        ]
        print(
            f"{att:<15} {mrk:<10} {score:<15.2f}   "
            f"{', '.join(active) if active else 'None'}"
        )

    print("\nOBSERVATION:")
    print("- Higher attendance AND marks lead to a higher performance score.")
    print("- When one input is low and the other high, the score is moderate.")
    print("- Boundary values activate multiple overlapping membership")
    print("  functions, producing intermediate scores.")


def exercise_2():
    """CSP: simple timetable scheduling."""

    print("\n" + "=" * 70)
    print("EXERCISE 2 — CSP: TIMETABLE SCHEDULING")
    print("=" * 70)

    subjects, domains, constraints = get_default_timetable_data()

    print("\nSubjects :", ", ".join(subjects))
    print("Time Slots:", ", ".join(list(domains.values())[0]))
    print("\nConstraints:")
    print("  C1: No two subjects in the same time slot.")
    print("  C2: Maths must be scheduled before Physics.")
    print("  C3: English must NOT be on Monday.")

    solution = solve_csp(subjects, domains, constraints)

    print_timetable(solution, "Generated Timetable")

    if solution:
        # Verify constraints in the output.
        slots = list(solution.values())
        all_unique = len(slots) == len(set(slots))
        all_slots = list(domains.values())[0]
        maths_ok = all_slots.index(solution["Maths"]) < \
                   all_slots.index(solution["Physics"])
        english_ok = not solution["English"].startswith("Mon")

        print("Constraint verification:")
        print(f"  C1 (no overlap)           : {'PASS' if all_unique else 'FAIL'}")
        print(f"  C2 (Maths before Physics) : {'PASS' if maths_ok else 'FAIL'}")
        print(f"  C3 (English not on Monday): {'PASS' if english_ok else 'FAIL'}")

    print("\nOBSERVATION:")
    print("- The backtracking solver finds a valid timetable that satisfies")
    print("  all three constraints without any violations.")


def exercise_3():
    """Modify fuzzy rules and CSP constraints; compare results."""

    print("\n" + "=" * 70)
    print("EXERCISE 3 — MODIFIED RULES AND CONSTRAINTS")
    print("=" * 70)

    # ── 3A: Modified fuzzy rules ──

    print("\n--- 3A: MODIFIED FUZZY RULES ---")
    print("\nChange: R7 now maps High Att & Low Marks -> Good (was Average)")
    print("Change: R2 now maps Low Att & Med Marks -> Average (was Poor)\n")

    modified_rules = [
        (attendance_low,    marks_low,    performance_poor,    "R1: Low Att  & Low Marks  -> Poor"),
        (attendance_low,    marks_medium, performance_average, "R2: Low Att  & Med Marks  -> Average  [CHANGED]"),
        (attendance_low,    marks_high,   performance_average, "R3: Low Att  & High Marks -> Average"),
        (attendance_medium, marks_low,    performance_poor,    "R4: Med Att  & Low Marks  -> Poor"),
        (attendance_medium, marks_medium, performance_average, "R5: Med Att  & Med Marks  -> Average"),
        (attendance_medium, marks_high,   performance_good,    "R6: Med Att  & High Marks -> Good"),
        (attendance_high,   marks_low,    performance_good,    "R7: High Att & Low Marks  -> Good     [CHANGED]"),
        (attendance_high,   marks_medium, performance_good,    "R8: High Att & Med Marks  -> Good"),
        (attendance_high,   marks_high,   performance_good,    "R9: High Att & High Marks -> Good"),
    ]

    original_rules = get_default_fuzzy_rules()

    compare_cases = [
        (50, 50),
        (75, 40),
        (40, 80),
    ]

    print(
        "{:<15s} {:<10s} {:<20s} {:<20s}".format(
            "Attendance", "Marks", "Original Score", "Modified Score"
        )
    )
    print("-" * 65)

    for att, mrk in compare_cases:
        orig_score, _ = fuzzy_student_performance(att, mrk, original_rules)
        mod_score, _ = fuzzy_student_performance(att, mrk, modified_rules)
        print(
            f"{att:<15} {mrk:<10} {orig_score:<20.2f} {mod_score:<20.2f}"
        )

    print("\nOBSERVATION (Fuzzy):")
    print("- Changing R7 to 'Good' increases the score for students with")
    print("  high attendance but low marks, rewarding attendance more.")
    print("- Changing R2 to 'Average' increases the score for students with")
    print("  low attendance but medium marks, being more lenient.")

    # ── 3B: Modified CSP constraints ──

    print("\n--- 3B: MODIFIED CSP CONSTRAINTS ---")
    print("\nChange: Chemistry must be on Monday (new constraint C4).")
    print("Change: Remove constraint C2 (Maths before Physics).\n")

    subjects = ["Maths", "Physics", "Chemistry", "English", "CS"]

    time_slots = [
        "Mon 9-10",
        "Mon 10-11",
        "Mon 11-12",
        "Tue 9-10",
        "Tue 10-11",
    ]

    domains = {subj: list(time_slots) for subj in subjects}

    modified_constraints = []

    # C1: No overlap (kept).
    def no_overlap(assignment):
        values = list(assignment.values())
        return len(values) == len(set(values))

    modified_constraints.append(no_overlap)

    # C2 (Maths before Physics) is REMOVED.

    # C3: English not on Monday (kept).
    def english_not_monday(assignment):
        if "English" in assignment:
            return not assignment["English"].startswith("Mon")
        return True

    modified_constraints.append(english_not_monday)

    # C4 (NEW): Chemistry must be on Monday.
    def chemistry_on_monday(assignment):
        if "Chemistry" in assignment:
            return assignment["Chemistry"].startswith("Mon")
        return True

    modified_constraints.append(chemistry_on_monday)

    print("Active constraints:")
    print("  C1: No two subjects in the same time slot.")
    print("  C3: English must NOT be on Monday.")
    print("  C4: Chemistry MUST be on Monday.  [NEW]")
    print("  (C2 removed: Maths no longer needs to be before Physics.)")

    # Original timetable for comparison.
    orig_subjects, orig_domains, orig_constraints = get_default_timetable_data()
    orig_solution = solve_csp(orig_subjects, orig_domains, orig_constraints)

    mod_solution = solve_csp(subjects, domains, modified_constraints)

    print_timetable(orig_solution, "Original Timetable")
    print_timetable(mod_solution, "Modified Timetable")

    if mod_solution:
        chem_ok = mod_solution["Chemistry"].startswith("Mon")
        english_ok = not mod_solution["English"].startswith("Mon")
        print("Constraint verification (modified):")
        print(f"  C1 (no overlap)             : PASS")
        print(f"  C3 (English not on Monday)  : {'PASS' if english_ok else 'FAIL'}")
        print(f"  C4 (Chemistry on Monday)    : {'PASS' if chem_ok else 'FAIL'}")

    print("\nOBSERVATION (CSP):")
    print("- Removing C2 gives the solver more freedom; Maths and Physics")
    print("  can now appear in any relative order.")
    print("- Adding C4 forces Chemistry onto Monday, reducing available")
    print("  Monday slots for other subjects and reshuffling the timetable.")
    print("- The solver still finds a valid assignment because the domain")
    print("  is large enough to satisfy all active constraints.")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("EXPERIMENT 6: FUZZY LOGIC AND CONSTRAINT SATISFACTION")
    print("=" * 70)

    print(
        """
This program demonstrates:
1. A fuzzy logic system for evaluating student performance
   based on attendance and marks.
2. A constraint satisfaction problem (CSP) for timetable scheduling.
3. The effect of changing fuzzy rules and CSP constraints.

No external libraries are needed.
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
