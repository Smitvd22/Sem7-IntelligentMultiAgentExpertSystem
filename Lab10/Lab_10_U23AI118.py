"""
Experiment 10: Intelligent Decision Support System
Student ID: U23AI118

Aim:
To develop an intelligent decision-support system using rule-based reasoning,
learning from examples, and ethical AI evaluation.

Application Domain: Loan / Credit Approval System

Exercises:
1. Expert System         : Rule-based knowledge base + inference engine for loan approval.
2. Imitation Learning    : Train a Decision-Tree model on human-expert examples; evaluate predictions.
3. Explanation Facility  : Display rules / feature importances responsible for each recommendation.
4. Ethical AI            : Test across demographic groups to detect bias; measure fairness metrics.
5. Testing & Comparison  : Modify rules / training examples; compare before-vs-after recommendations.
"""

import os
import random
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] scikit-learn not found - Exercise 2 will use a hand-crafted model.\n")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# SHARED UTILITIES
# ============================================================
def section(title):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def subsection(title):
    print(f"\n  {'-' * 60}\n  {title}\n  {'-' * 60}")

def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {filename}")
    return path


# ============================================================
# EXERCISE 1 - EXPERT SYSTEM
# ============================================================
LOAN_RULES = [
    {
        "id": "R1",
        "name": "Hard Reject - Very High DTI",
        "description": "Debt-to-Income ratio > 60%",
        "condition": lambda a: a["dti"] > 60,
        "action": "REJECT",
        "reason": "DTI exceeds 60% - repayment capacity is critically low.",
    },
    {
        "id": "R2",
        "name": "Hard Reject - Very Low Credit Score",
        "description": "Credit score < 500",
        "condition": lambda a: a["credit_score"] < 500,
        "action": "REJECT",
        "reason": "Credit score below 500 indicates very high default risk.",
    },
    {
        "id": "R3",
        "name": "Hard Reject - Recent Default",
        "description": "Defaulted in last 2 years",
        "condition": lambda a: a["recent_default"] is True,
        "action": "REJECT",
        "reason": "Applicant has defaulted on a loan within the past 2 years.",
    },
    {
        "id": "R4",
        "name": "Strong Approve - Excellent Profile",
        "description": "Credit >= 750, income >= 60k, DTI <= 30, employed",
        "condition": lambda a: (
            a["credit_score"] >= 750
            and a["annual_income"] >= 60_000
            and a["dti"] <= 30
            and a["employment_status"] == "employed"
        ),
        "action": "APPROVE",
        "reason": "Excellent credit score, sufficient income, low DTI, and stable employment.",
    },
    {
        "id": "R5",
        "name": "Conditional Approve - Good Profile with Collateral",
        "description": "Credit >= 650, DTI <= 45, collateral provided",
        "condition": lambda a: (
            a["credit_score"] >= 650
            and a["dti"] <= 45
            and a["collateral"] is True
        ),
        "action": "APPROVE (with conditions)",
        "reason": "Good credit score and acceptable DTI; collateral mitigates risk.",
    },
    {
        "id": "R6",
        "name": "Approve - Good Credit, Stable Income",
        "description": "Credit >= 700, income >= 40k, employed",
        "condition": lambda a: (
            a["credit_score"] >= 700
            and a["annual_income"] >= 40_000
            and a["employment_status"] == "employed"
        ),
        "action": "APPROVE",
        "reason": "Good credit score, sufficient income, and stable employment.",
    },
    {
        "id": "R7",
        "name": "Manual Review - Borderline",
        "description": "Credit 580-699, DTI 30-55",
        "condition": lambda a: (
            580 <= a["credit_score"] <= 699
            and 30 <= a["dti"] <= 55
        ),
        "action": "MANUAL REVIEW",
        "reason": "Borderline credit score and moderate DTI; needs underwriter review.",
    },
    {
        "id": "R8",
        "name": "Reject - Low Income and Poor Credit",
        "description": "Income < 20k and credit < 600",
        "condition": lambda a: (
            a["annual_income"] < 20_000
            and a["credit_score"] < 600
        ),
        "action": "REJECT",
        "reason": "Insufficient income combined with poor credit score.",
    },
    {
        "id": "R9",
        "name": "Default Rule - Reject",
        "description": "No other rule matched",
        "condition": lambda a: True,
        "action": "REJECT",
        "reason": "Profile does not meet minimum lending criteria.",
    },
]


def run_inference_engine(applicant, rules=None, verbose=True):
    """Forward-chaining inference engine. Returns decision dict."""
    if rules is None:
        rules = LOAN_RULES
    fired_rules_checked = []
    for rule in rules:
        match = rule["condition"](applicant)
        fired_rules_checked.append((rule["id"], rule["name"], match))
        if match:
            result = {
                "action": rule["action"],
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "reason": rule["reason"],
                "rules_checked": fired_rules_checked,
            }
            if verbose:
                print(f"    Fired Rule  : [{rule['id']}] {rule['name']}")
                print(f"    Decision    : {rule['action']}")
                print(f"    Explanation : {rule['reason']}")
            return result
    return {"action": "REJECT", "rule_id": "DEFAULT", "rule_name": "Default",
            "reason": "No rule matched.", "rules_checked": fired_rules_checked}


def exercise1_expert_system():
    section("EXERCISE 1 - Expert System: Rule-Based Inference Engine")

    print("\n  Knowledge Base: Loan Approval Rules")
    print(f"  {'ID':<4} {'Rule Name':<45} {'Description'}")
    print("  " + "-" * 90)
    for r in LOAN_RULES:
        print(f"  {r['id']:<4} {r['name']:<45} {r['description']}")

    test_applicants = [
        {"name": "Alice (Excellent)",
         "credit_score": 780, "annual_income": 85_000, "dti": 22,
         "employment_status": "employed", "collateral": True, "recent_default": False},
        {"name": "Bob (Good + Collateral)",
         "credit_score": 660, "annual_income": 45_000, "dti": 38,
         "employment_status": "self-employed", "collateral": True, "recent_default": False},
        {"name": "Carol (Borderline)",
         "credit_score": 620, "annual_income": 35_000, "dti": 44,
         "employment_status": "employed", "collateral": False, "recent_default": False},
        {"name": "Dave (High DTI)",
         "credit_score": 710, "annual_income": 50_000, "dti": 65,
         "employment_status": "employed", "collateral": False, "recent_default": False},
        {"name": "Eve (Recent Default)",
         "credit_score": 590, "annual_income": 32_000, "dti": 40,
         "employment_status": "employed", "collateral": False, "recent_default": True},
        {"name": "Frank (Low Income & Poor Credit)",
         "credit_score": 540, "annual_income": 18_000, "dti": 55,
         "employment_status": "unemployed", "collateral": False, "recent_default": False},
    ]

    results = []
    for app in test_applicants:
        subsection(f"Applicant: {app['name']}")
        print(f"    Credit Score      : {app['credit_score']}")
        print(f"    Annual Income     : ${app['annual_income']:,}")
        print(f"    Debt-to-Income %  : {app['dti']}")
        print(f"    Employment Status : {app['employment_status']}")
        print(f"    Collateral        : {app['collateral']}")
        print(f"    Recent Default    : {app['recent_default']}")
        print()
        result = run_inference_engine(app, verbose=True)
        results.append((app["name"], result["action"], result["rule_id"]))

    subsection("Summary Table")
    print(f"  {'Applicant':<30} {'Decision':<25} {'Rule Fired'}")
    print("  " + "-" * 70)
    for name, decision, rule_id in results:
        print(f"  {name:<30} {decision:<25} {rule_id}")

    decision_counts = defaultdict(int)
    for _, d, _ in results:
        decision_counts[d] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"APPROVE": "#2ecc71", "APPROVE (with conditions)": "#f39c12",
              "MANUAL REVIEW": "#3498db", "REJECT": "#e74c3c"}
    labels = list(decision_counts.keys())
    vals = [decision_counts[l] for l in labels]
    bar_colors = [colors.get(l, "#95a5a6") for l in labels]
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, padding=4, fontsize=12, fontweight="bold")
    ax.set_title("Exercise 1 - Expert System: Decision Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Applicants")
    ax.set_ylim(0, max(vals) + 2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_fig(fig, "ex1_expert_system_decisions.png")
    return results


# ============================================================
# EXERCISE 2 - IMITATION LEARNING
# ============================================================
def _generate_expert_dataset(n=300):
    """Synthesise labelled examples that mirror the expert rules."""
    rng = np.random.default_rng(RANDOM_SEED)
    credit_scores  = rng.integers(450, 850, n)
    annual_incomes = rng.integers(15_000, 120_000, n)
    dtis           = rng.integers(10, 70, n)
    employment     = rng.choice([0, 1], n, p=[0.3, 0.7])
    collateral     = rng.choice([0, 1], n, p=[0.4, 0.6])
    recent_default = rng.choice([0, 1], n, p=[0.85, 0.15])
    X = np.column_stack([credit_scores, annual_incomes, dtis,
                         employment, collateral, recent_default])
    y = []
    for i in range(n):
        cs, inc, dti, emp, col, rd = X[i]
        app = {
            "credit_score": cs, "annual_income": inc, "dti": dti,
            "employment_status": "employed" if emp else "unemployed",
            "collateral": bool(col), "recent_default": bool(rd),
        }
        result = run_inference_engine(app, verbose=False)
        label = 1 if "APPROVE" in result["action"] else 0
        y.append(label)
    feature_names = ["Credit Score", "Annual Income", "DTI (%)",
                     "Employed", "Collateral", "Recent Default"]
    return X, np.array(y), feature_names


def exercise2_imitation_learning():
    section("EXERCISE 2 - Imitation Learning: Decision-Tree on Expert Examples")
    X, y, feature_names = _generate_expert_dataset(n=300)
    print(f"\n  Dataset: {len(X)} examples | Approved: {y.sum()} | Rejected: {(y==0).sum()}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    if SKLEARN_AVAILABLE:
        clf = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        subsection("Model Evaluation")
        print(f"  Accuracy : {acc * 100:.2f} %\n")
        print(classification_report(y_test, y_pred,
                                    target_names=["REJECT", "APPROVE"],
                                    zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im = axes[0].imshow(cm, cmap="Blues")
        axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
        axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(["REJECT", "APPROVE"])
        axes[0].set_yticklabels(["REJECT", "APPROVE"])
        for r in range(2):
            for c in range(2):
                axes[0].text(c, r, str(cm[r, c]), ha="center", va="center",
                             fontsize=16, fontweight="bold",
                             color="white" if cm[r, c] > cm.max() / 2 else "black")
        importances = clf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        axes[1].barh([feature_names[i] for i in sorted_idx[::-1]],
                     importances[sorted_idx[::-1]],
                     color="#3498db", edgecolor="white")
        axes[1].set_title("Feature Importances (Imitation Model)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Importance")
        axes[1].spines[["top", "right"]].set_visible(False)
        plt.suptitle("Exercise 2 - Imitation Learning Results", fontsize=15, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, "ex2_imitation_learning.png")
        return clf, X_test, y_test, y_pred, feature_names, X, y
    else:
        subsection("Fallback Threshold Model (sklearn unavailable)")
        correct = 0
        predictions = []
        for i, xi in enumerate(X_test):
            cs, inc, dti, emp, col, rd = xi
            pred = 1 if (cs >= 650 and dti <= 45 and not rd) else 0
            predictions.append(pred)
            if pred == y_test[i]:
                correct += 1
        acc = correct / len(y_test)
        print(f"  Threshold Model Accuracy: {acc * 100:.2f} %")
        return None, X_test, y_test, np.array(predictions), feature_names, X, y


# ============================================================
# EXERCISE 3 - EXPLANATION FACILITY
# ============================================================
def exercise3_explanation_facility(clf, feature_names, X, y):
    section("EXERCISE 3 - Explanation Facility: Rules & Feature Importances")

    subsection("3a. Rule-Based Explanation for Individual Applicants")
    demo_apps = [
        {"name": "Grace", "credit_score": 760, "annual_income": 70_000, "dti": 25,
         "employment_status": "employed", "collateral": True, "recent_default": False},
        {"name": "Hank", "credit_score": 580, "annual_income": 28_000, "dti": 50,
         "employment_status": "employed", "collateral": False, "recent_default": False},
        {"name": "Iris", "credit_score": 490, "annual_income": 22_000, "dti": 35,
         "employment_status": "unemployed", "collateral": False, "recent_default": False},
    ]
    for app in demo_apps:
        print(f"\n  Applicant : {app['name']}")
        result = run_inference_engine(app, verbose=False)
        print(f"  Decision  : {result['action']}")
        print(f"  Rule Fired: [{result['rule_id']}] {result['rule_name']}")
        print(f"  Reason    : {result['reason']}")
        print(f"  Rules checked before firing ({len(result['rules_checked'])}):")
        for rid, rname, matched in result["rules_checked"]:
            status = "FIRED" if matched else "skip"
            print(f"    [{rid}] {rname:<45} - {status}")

    if SKLEARN_AVAILABLE and clf is not None:
        subsection("3b. Decision-Tree Textual Structure (top 3 levels)")
        tree_text = export_text(clf, feature_names=feature_names, max_depth=3)
        print(textwrap.indent(tree_text, "    "))
        subsection("3c. Feature Importance Ranking")
        importances = clf.feature_importances_
        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        for rank, (fname, imp) in enumerate(ranked, 1):
            bar = "#" * int(imp * 40)
            print(f"  {rank}. {fname:<20} {bar:<42} {imp:.4f}")

    subsection("3d. Rule-Check Heatmap for Demo Applicants")
    rule_ids  = [r["id"] for r in LOAN_RULES]
    app_names = [a["name"] for a in demo_apps]
    heatmap   = np.zeros((len(demo_apps), len(LOAN_RULES)))
    decisions = []
    for i, app in enumerate(demo_apps):
        res = run_inference_engine(app, verbose=False)
        decisions.append(res["action"])
        for j, (rid, _, matched) in enumerate(res["rules_checked"]):
            heatmap[i, j] = 1 if matched else 0.2

    fig, ax = plt.subplots(figsize=(12, 3.5))
    cax = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(rule_ids)))
    ax.set_xticklabels(rule_ids, fontsize=10)
    ax.set_yticks(range(len(app_names)))
    ax.set_yticklabels([f"{n}\n({d})" for n, d in zip(app_names, decisions)], fontsize=9)
    ax.set_xlabel("Rules (left to right = priority order)")
    ax.set_title("Exercise 3 - Rule Activation Heatmap\n"
                 "(bright = rule FIRED, medium = checked but not fired)",
                 fontsize=13, fontweight="bold")
    plt.colorbar(cax, ax=ax, shrink=0.8, label="Activation (1=fired)")
    plt.tight_layout()
    save_fig(fig, "ex3_explanation_heatmap.png")


# ============================================================
# EXERCISE 4 - ETHICAL AI: FAIRNESS & BIAS DETECTION
# ============================================================
def exercise4_ethical_ai():
    section("EXERCISE 4 - Ethical AI: Fairness & Bias Detection")

    subsection("Scenario A - Identical Financial Profile Across Groups")
    base_profile = {
        "credit_score": 650, "annual_income": 40_000, "dti": 38,
        "employment_status": "employed", "collateral": False, "recent_default": False,
    }
    groups_a = ["Group A", "Group B", "Group C", "Group D"]
    print(f"  Base Profile: credit={base_profile['credit_score']}, "
          f"income=${base_profile['annual_income']:,}, DTI={base_profile['dti']}%\n")
    for g in groups_a:
        profile = dict(base_profile, name=g)
        result = run_inference_engine(profile, verbose=False)
        print(f"  {g}: Decision = {result['action']}  (Rule: {result['rule_id']})")
    print("\n  All groups with identical financials receive the SAME decision.")
    print("  -> No rule-based demographic bias detected in Scenario A.\n")

    subsection("Scenario B - Realistic Demographic Group Profiles")
    rng = np.random.default_rng(RANDOM_SEED + 1)
    groups_b = {
        "Group X (High-Income)" : {"income_mean": 70_000, "cs_mean": 710, "dti_mean": 28},
        "Group Y (Mid-Income)"  : {"income_mean": 45_000, "cs_mean": 660, "dti_mean": 38},
        "Group Z (Low-Income)"  : {"income_mean": 28_000, "cs_mean": 610, "dti_mean": 50},
        "Group W (Mixed)"       : {"income_mean": 50_000, "cs_mean": 690, "dti_mean": 35},
    }
    approval_rates = {}
    group_details  = {}
    for g_name, params in groups_b.items():
        n_samples = 100
        cs  = rng.normal(params["cs_mean"],  50, n_samples).clip(400, 850).astype(int)
        inc = rng.normal(params["income_mean"], 10_000, n_samples).clip(12_000, 150_000).astype(int)
        dti = rng.normal(params["dti_mean"], 10, n_samples).clip(5, 70).astype(int)
        emp = rng.choice(["employed", "unemployed"], n_samples, p=[0.75, 0.25])
        col = rng.choice([True, False], n_samples, p=[0.5, 0.5])
        rd  = rng.choice([True, False], n_samples, p=[0.1, 0.9])
        approved = 0
        for i in range(n_samples):
            app = {"credit_score": cs[i], "annual_income": inc[i], "dti": dti[i],
                   "employment_status": emp[i], "collateral": col[i],
                   "recent_default": rd[i]}
            result = run_inference_engine(app, verbose=False)
            if "APPROVE" in result["action"]:
                approved += 1
        rate = approved / n_samples
        approval_rates[g_name] = rate
        group_details[g_name] = params
        print(f"  {g_name:<30} Approval Rate: {rate * 100:.1f} %  "
              f"(avg income ${params['income_mean']:,}, cs {params['cs_mean']}, dti {params['dti_mean']}%)")

    subsection("Fairness Metrics")
    rates     = list(approval_rates.values())
    max_rate  = max(rates)
    min_rate  = min(rates)
    disparity = max_rate - min_rate
    print(f"  Maximum Approval Rate : {max_rate * 100:.1f} %")
    print(f"  Minimum Approval Rate : {min_rate * 100:.1f} %")
    print(f"  Disparity (max-min)   : {disparity * 100:.1f} pp")
    ratio_80 = min_rate / max_rate if max_rate > 0 else 1
    print(f"  80% Rule Ratio        : {ratio_80:.3f}  "
          f"({'PASS' if ratio_80 >= 0.8 else 'FAIL - Potential Disparate Impact'})")
    print("\n  Analysis:")
    print("  Disparity is driven by FINANCIAL features (income, credit score, DTI).")
    print("  The rule-based system is TRANSPARENT: every decision has an auditable rule.")
    print("  Recommendation: Monitor that income/credit-score proxies do not")
    print("  inadvertently encode protected attributes in real-world deployment.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    g_labels = list(approval_rates.keys())
    g_rates  = [approval_rates[g] * 100 for g in g_labels]
    bar_cols = ["#2ecc71" if r >= 50 else "#e74c3c" for r in g_rates]
    bars = axes[0].bar(g_labels, g_rates, color=bar_cols, edgecolor="white", linewidth=1.5)
    axes[0].axhline(80, color="#e67e22", linestyle="--", linewidth=1.5, label="80% threshold")
    axes[0].bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
    axes[0].set_title("Approval Rate by Demographic Group", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Approval Rate (%)")
    axes[0].set_ylim(0, 110)
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)
    plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right", fontsize=9)
    incomes = [groups_b[g]["income_mean"] for g in g_labels]
    axes[1].scatter(incomes, g_rates, s=200, c=bar_cols, edgecolors="black", zorder=3)
    for i, g in enumerate(g_labels):
        axes[1].annotate(g, (incomes[i], g_rates[i]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes[1].set_xlabel("Group Mean Annual Income ($)")
    axes[1].set_ylabel("Approval Rate (%)")
    axes[1].set_title("Approval Rate vs. Mean Income", fontsize=13, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    plt.suptitle("Exercise 4 - Ethical AI: Fairness Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "ex4_ethical_ai_fairness.png")
    return approval_rates


# ============================================================
# EXERCISE 5 - TESTING & COMPARISON (STRICTER RULE SET)
# ============================================================
LOAN_RULES_STRICT = [
    {
        "id": "R1", "name": "Hard Reject - Very High DTI",
        "description": "DTI > 50% (tightened from 60)",
        "condition": lambda a: a["dti"] > 50,
        "action": "REJECT",
        "reason": "DTI exceeds 50% under revised policy.",
    },
    {
        "id": "R2", "name": "Hard Reject - Very Low Credit Score",
        "description": "Credit < 550 (raised from 500)",
        "condition": lambda a: a["credit_score"] < 550,
        "action": "REJECT",
        "reason": "Credit below 550 - very high risk (policy updated).",
    },
    {
        "id": "R3", "name": "Hard Reject - Recent Default",
        "description": "Defaulted in last 2 years",
        "condition": lambda a: a["recent_default"] is True,
        "action": "REJECT",
        "reason": "Applicant defaulted within past 2 years.",
    },
    {
        "id": "R4", "name": "Strong Approve - Excellent Profile (Stricter)",
        "description": "Credit >= 800, income >= 80k, DTI <= 25, employed",
        "condition": lambda a: (
            a["credit_score"] >= 800
            and a["annual_income"] >= 80_000
            and a["dti"] <= 25
            and a["employment_status"] == "employed"
        ),
        "action": "APPROVE",
        "reason": "Exceptional profile meets strict revised criteria.",
    },
    {
        "id": "R5", "name": "Conditional Approve - With Collateral",
        "description": "Credit >= 700, DTI <= 35, collateral",
        "condition": lambda a: (
            a["credit_score"] >= 700
            and a["dti"] <= 35
            and a["collateral"] is True
        ),
        "action": "APPROVE (with conditions)",
        "reason": "Meets revised threshold; collateral required.",
    },
    {
        "id": "R6", "name": "Approve - Good Credit, Stable Income (Stricter)",
        "description": "Credit >= 730, income >= 55k, employed",
        "condition": lambda a: (
            a["credit_score"] >= 730
            and a["annual_income"] >= 55_000
            and a["employment_status"] == "employed"
        ),
        "action": "APPROVE",
        "reason": "Meets revised income and credit thresholds.",
    },
    {
        "id": "R7", "name": "Manual Review - Borderline",
        "description": "Credit 600-729, DTI 25-45",
        "condition": lambda a: (
            600 <= a["credit_score"] <= 729
            and 25 <= a["dti"] <= 45
        ),
        "action": "MANUAL REVIEW",
        "reason": "Borderline under revised policy; manual review required.",
    },
    {
        "id": "R8", "name": "Reject - Low Income and Poor Credit",
        "description": "Income < 25k and credit < 650",
        "condition": lambda a: (
            a["annual_income"] < 25_000
            and a["credit_score"] < 650
        ),
        "action": "REJECT",
        "reason": "Insufficient income and poor credit under revised thresholds.",
    },
    {
        "id": "R9", "name": "Default Rule - Reject",
        "description": "No other rule matched",
        "condition": lambda a: True,
        "action": "REJECT",
        "reason": "Does not meet revised minimum criteria.",
    },
]


def exercise5_testing_and_comparison(ex1_results):
    section("EXERCISE 5 - Testing & Comparison: Original vs. Modified Rules")

    comparison_apps = [
        {"name": "Alice (Excellent)",
         "credit_score": 780, "annual_income": 85_000, "dti": 22,
         "employment_status": "employed", "collateral": True, "recent_default": False},
        {"name": "Bob (Good + Collateral)",
         "credit_score": 660, "annual_income": 45_000, "dti": 38,
         "employment_status": "self-employed", "collateral": True, "recent_default": False},
        {"name": "Carol (Borderline)",
         "credit_score": 620, "annual_income": 35_000, "dti": 44,
         "employment_status": "employed", "collateral": False, "recent_default": False},
        {"name": "Dave (High DTI)",
         "credit_score": 710, "annual_income": 50_000, "dti": 65,
         "employment_status": "employed", "collateral": False, "recent_default": False},
        {"name": "Eve (Recent Default)",
         "credit_score": 590, "annual_income": 32_000, "dti": 40,
         "employment_status": "employed", "collateral": False, "recent_default": True},
        {"name": "Frank (Low Income & Poor Credit)",
         "credit_score": 540, "annual_income": 18_000, "dti": 55,
         "employment_status": "unemployed", "collateral": False, "recent_default": False},
        {"name": "Grace (Borderline-Strict)",
         "credit_score": 720, "annual_income": 52_000, "dti": 30,
         "employment_status": "employed", "collateral": False, "recent_default": False},
        {"name": "Hank (Mid-Tier)",
         "credit_score": 680, "annual_income": 48_000, "dti": 36,
         "employment_status": "employed", "collateral": True, "recent_default": False},
    ]

    print(f"\n  {'Applicant':<32} {'Original Decision':<28} {'Modified Decision':<28} {'Changed?'}")
    print("  " + "-" * 105)
    before_counts = defaultdict(int)
    after_counts  = defaultdict(int)
    comparison_rows = []
    for app in comparison_apps:
        orig   = run_inference_engine(app, rules=LOAN_RULES,        verbose=False)
        strict = run_inference_engine(app, rules=LOAN_RULES_STRICT, verbose=False)
        changed = "CHANGED" if orig["action"] != strict["action"] else "same"
        before_counts[orig["action"]]   += 1
        after_counts[strict["action"]]  += 1
        comparison_rows.append((app["name"], orig["action"], strict["action"], changed))
        print(f"  {app['name']:<32} {orig['action']:<28} {strict['action']:<28} {changed}")

    all_decisions = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
    x = np.arange(len(all_decisions))
    width = 0.35
    before_vals = [before_counts.get(d, 0) for d in all_decisions]
    after_vals  = [after_counts.get(d, 0)  for d in all_decisions]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    bars1 = axes[0].bar(x - width/2, before_vals, width, label="Original Rules",
                        color="#3498db", edgecolor="white", linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, after_vals,  width, label="Stricter Rules",
                        color="#e74c3c", edgecolor="white", linewidth=1.5)
    axes[0].bar_label(bars1, padding=3, fontsize=11)
    axes[0].bar_label(bars2, padding=3, fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_decisions, rotation=15, ha="right", fontsize=9)
    axes[0].set_title("Decision Distribution: Original vs. Stricter Rules",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].axis("off")
    table_data = [["Applicant", "Original", "Modified", "Change?"]]
    for row in comparison_rows:
        table_data.append([row[0][:26], row[1][:18], row[2][:18], row[3]])
    tbl = axes[1].table(cellText=table_data[1:], colLabels=table_data[0],
                        loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width([0, 1, 2, 3])
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 0 and r <= len(comparison_rows) and comparison_rows[r-1][3] != "same":
            cell.set_facecolor("#fdecea")
    axes[1].set_title("Per-Applicant Decision Comparison", fontsize=13, fontweight="bold")
    plt.suptitle("Exercise 5 - Testing & Comparison: Rule Modification Impact",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "ex5_testing_comparison.png")

    changed_count = sum(1 for row in comparison_rows if row[3] != "same")
    subsection("Key Observations")
    print(f"  Total applicants tested            : {len(comparison_apps)}")
    print(f"  Decisions changed (Original->Strict): {changed_count}")
    print(f"  Unchanged decisions                : {len(comparison_apps) - changed_count}")
    print()
    print("  Stricter rules have the following effects:")
    print("  * DTI limit: 60% -> 50%  => more high-debt applicants rejected")
    print("  * Credit floor: 500->550, 750->800 => fewer automatic approvals")
    print("  * Income requirement raised => mid-tier shifts to MANUAL REVIEW or REJECT")
    print("  * Transparency preserved: each changed decision has a clear auditable rule.")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("   EXPERIMENT 10 - INTELLIGENT DECISION SUPPORT SYSTEM")
    print("   Application Domain: Loan / Credit Approval")
    print("   Student ID: U23AI118")
    print("=" * 70)

    ex1_results = exercise1_expert_system()

    if SKLEARN_AVAILABLE:
        clf, X_test, y_test, y_pred, feature_names, X, y = exercise2_imitation_learning()
    else:
        clf, X_test, y_test, y_pred, feature_names, X, y = exercise2_imitation_learning()

    exercise3_explanation_facility(clf, feature_names, X, y)
    exercise4_ethical_ai()
    exercise5_testing_and_comparison(ex1_results)

    section("EXPERIMENT 10 - COMPLETE")
    print(f"\n  Output files in: {OUTPUT_DIR}")
    for p in ["ex1_expert_system_decisions.png", "ex2_imitation_learning.png",
              "ex3_explanation_heatmap.png", "ex4_ethical_ai_fairness.png",
              "ex5_testing_comparison.png"]:
        print(f"    * {p}")
    print()
    print("  Exercise Summary:")
    print("  1. Expert System      -> Rule-based inference engine with 9 loan-approval rules")
    print("  2. Imitation Learning -> Decision-Tree trained on 300 expert-labelled examples")
    print("  3. Explanation        -> Rule-trace + tree structure + feature importance ranking")
    print("  4. Ethical AI         -> 80%-rule fairness check across 4 demographic groups")
    print("  5. Testing            -> Compared 8 applicants before and after rule tightening")


if __name__ == "__main__":
    main()
