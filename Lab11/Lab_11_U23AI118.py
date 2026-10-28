"""
Experiment 11: Project LAST EXIT — Intelligent Disaster Evacuation System
Student ID: U23AI118

Aim:
To develop an intelligent system that coordinates rescue teams and helps people
evacuate safely during a disaster.

Exercises:
1. Knowledge Representation : Rescue teams, affected areas, shelters and their
                               relationships using frames / semantic-network style dicts.
2. Rule-Based Reasoning      : IF-THEN forward-chaining rules to identify evacuation
                               priorities and suggest rescue actions.
3. Handling Uncertainty      : Fuzzy-logic membership functions + centroid defuzzification
                               for uncertain flood levels and road conditions.
4. Resource Allocation       : Greedy priority-based assignment of teams -> areas -> shelters.
5. Multi-Agent Coordination  : Contract-Net Protocol (CNP) -- manager broadcasts tasks,
                               agents bid, manager awards, conflicts resolved by re-bidding.
6. Simulation & Evaluation   : Three scenarios (normal + unexpected events) compared on
                               evacuation time, resource usage, and people saved.
"""

import os
import random
import math
import textwrap
from collections import defaultdict
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "red":    "#E63946",
    "orange": "#F4A261",
    "yellow": "#E9C46A",
    "green":  "#2A9D8F",
    "blue":   "#457B9D",
    "purple": "#6D6875",
    "dark":   "#1D3557",
    "light":  "#A8DADC",
    "white":  "#F1FAEE",
}

def section(title):
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")

def subsection(title):
    bar = "-" * 62
    print(f"\n  {bar}\n  {title}\n  {bar}")

def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {filename}")
    return path


KNOWLEDGE_BASE = {
    "areas": {
        "A1": {"name": "Downtown Core",     "population": 3500, "flood_level": 8.2,
               "road_condition": 0.3,  "priority": None, "evacuated": 0},
        "A2": {"name": "North Riverside",   "population": 1800, "flood_level": 6.5,
               "road_condition": 0.6,  "priority": None, "evacuated": 0},
        "A3": {"name": "East Industrial",   "population": 900,  "flood_level": 4.1,
               "road_condition": 0.8,  "priority": None, "evacuated": 0},
        "A4": {"name": "South Suburbs",     "population": 2100, "flood_level": 7.8,
               "road_condition": 0.4,  "priority": None, "evacuated": 0},
        "A5": {"name": "West Hills",        "population": 600,  "flood_level": 2.3,
               "road_condition": 0.9,  "priority": None, "evacuated": 0},
    },
    "shelters": {
        "S1": {"name": "City Sports Arena",  "capacity": 2000, "occupied": 0,
               "location": "Central",  "available": True},
        "S2": {"name": "North High School",  "capacity": 800,  "occupied": 0,
               "location": "North",    "available": True},
        "S3": {"name": "West Community Hall","capacity": 500,  "occupied": 0,
               "location": "West",     "available": True},
        "S4": {"name": "East Trade Centre",  "capacity": 1200, "occupied": 0,
               "location": "East",     "available": True},
    },
    "rescue_teams": {
        "T1": {"name": "Alpha Squad",   "type": "boat",       "capacity": 150,
               "speed": 0.9, "available": True, "assigned_area": None, "assigned_shelter": None},
        "T2": {"name": "Bravo Unit",    "type": "truck",      "capacity": 200,
               "speed": 0.7, "available": True, "assigned_area": None, "assigned_shelter": None},
        "T3": {"name": "Charlie Corps", "type": "helicopter", "capacity": 80,
               "speed": 1.0, "available": True, "assigned_area": None, "assigned_shelter": None},
        "T4": {"name": "Delta Force",   "type": "boat",       "capacity": 120,
               "speed": 0.85,"available": True, "assigned_area": None, "assigned_shelter": None},
        "T5": {"name": "Echo Rescue",   "type": "truck",      "capacity": 180,
               "speed": 0.65,"available": True, "assigned_area": None, "assigned_shelter": None},
    },
    "road_network": {
        ("A1", "S1"): 0.5, ("A1", "S2"): 0.3, ("A1", "S3"): 0.2, ("A1", "S4"): 0.6,
        ("A2", "S1"): 0.7, ("A2", "S2"): 0.9, ("A2", "S3"): 0.4, ("A2", "S4"): 0.5,
        ("A3", "S1"): 0.8, ("A3", "S2"): 0.6, ("A3", "S3"): 0.5, ("A3", "S4"): 0.95,
        ("A4", "S1"): 0.6, ("A4", "S2"): 0.4, ("A4", "S3"): 0.7, ("A4", "S4"): 0.5,
        ("A5", "S1"): 0.5, ("A5", "S2"): 0.6, ("A5", "S3"): 0.9, ("A5", "S4"): 0.4,
    },
}


def exercise_1(kb):
    section("EXERCISE 1 — KNOWLEDGE REPRESENTATION")
    print("\n  Technique: Frame-Based Representation (semantic-network style dicts)")
    print("  Each entity (area / shelter / team) is a frame with named slots.\n")

    subsection("Affected Areas")
    print(f"  {'ID':<4} {'Name':<22} {'Pop':>6} {'Flood(m)':>9} {'Road Cond':>10}")
    print(f"  {'-'*4} {'-'*22} {'-'*6} {'-'*9} {'-'*10}")
    for aid, a in kb["areas"].items():
        print(f"  {aid:<4} {a['name']:<22} {a['population']:>6} "
              f"{a['flood_level']:>9.1f} {a['road_condition']:>10.2f}")

    subsection("Shelters")
    print(f"  {'ID':<4} {'Name':<24} {'Capacity':>9} {'Location':<12} {'Available'}")
    print(f"  {'-'*4} {'-'*24} {'-'*9} {'-'*12} {'-'*9}")
    for sid, s in kb["shelters"].items():
        print(f"  {sid:<4} {s['name']:<24} {s['capacity']:>9} "
              f"{s['location']:<12} {str(s['available'])}")

    subsection("Rescue Teams")
    print(f"  {'ID':<4} {'Name':<16} {'Type':<12} {'Cap':>5} {'Speed':>6}")
    print(f"  {'-'*4} {'-'*16} {'-'*12} {'-'*5} {'-'*6}")
    for tid, t in kb["rescue_teams"].items():
        print(f"  {tid:<4} {t['name']:<16} {t['type']:<12} {t['capacity']:>5} {t['speed']:>6.2f}")

    subsection("Road-Network Passability (area -> shelter)")
    area_groups = defaultdict(dict)
    for (aid, sid), score in kb["road_network"].items():
        area_groups[aid][sid] = score
    header = f"  {'Route':<12}" + "".join(f"{sid:>8}" for sid in kb["shelters"])
    print(header)
    print(f"  {'-'*12}" + "-"*8*len(kb["shelters"]))
    for aid, routes in sorted(area_groups.items()):
        row = f"  {aid:<12}"
        for sid in kb["shelters"]:
            row += f"{routes.get(sid, 0.0):>8.2f}"
        print(row)

    _plot_knowledge_overview(kb)


def _plot_knowledge_overview(kb):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Exercise 1 — Knowledge Base Overview", fontsize=14, fontweight="bold",
                 color=COLORS["dark"])
    fig.patch.set_facecolor(COLORS["white"])

    ax = axes[0]
    areas = list(kb["areas"].keys())
    flood = [kb["areas"][a]["flood_level"] for a in areas]
    road  = [kb["areas"][a]["road_condition"] * 10 for a in areas]
    pop   = [kb["areas"][a]["population"] / 500 for a in areas]
    x = np.arange(len(areas)); w = 0.25
    ax.bar(x - w, flood, w, label="Flood level (m)", color=COLORS["blue"])
    ax.bar(x,     road,  w, label="Road cond x10",  color=COLORS["green"])
    ax.bar(x + w, pop,   w, label="Pop /500",        color=COLORS["orange"])
    ax.set_xticks(x); ax.set_xticklabels(areas)
    ax.set_title("Affected Areas — Risk Factors", color=COLORS["dark"])
    ax.legend(fontsize=8); ax.set_facecolor("#EEF4F7"); ax.set_ylabel("Scaled value")

    ax = axes[1]
    shelters = list(kb["shelters"].keys())
    cap = [kb["shelters"][s]["capacity"] for s in shelters]
    bars = ax.bar(shelters, cap, color=[COLORS["green"], COLORS["blue"], COLORS["purple"], COLORS["light"]])
    ax.set_title("Shelter Capacities", color=COLORS["dark"])
    ax.set_ylabel("Capacity (people)"); ax.set_facecolor("#EEF4F7")
    for bar, c in zip(bars, cap):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(c), ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    teams = list(kb["rescue_teams"].keys())
    scores = [kb["rescue_teams"][t]["capacity"] * kb["rescue_teams"][t]["speed"] / 100 for t in teams]
    palette = [COLORS["red"], COLORS["orange"], COLORS["yellow"], COLORS["green"], COLORS["blue"]]
    bars2 = ax.barh(teams, scores, color=palette)
    ax.set_title("Team Capability (Cap x Speed / 100)", color=COLORS["dark"])
    ax.set_xlabel("Capability Score"); ax.set_facecolor("#EEF4F7")
    for bar, sc in zip(bars2, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{sc:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    save_fig(fig, "ex1_knowledge_representation.png")


EVACUATION_RULES = [
    {"id": "R1", "name": "Critical Flood Priority",
     "description": "Flood level > 7 m => CRITICAL priority",
     "condition": lambda a: a["flood_level"] > 7.0,
     "action": "priority = CRITICAL",
     "apply": lambda a: a.update({"priority": "CRITICAL"})},
    {"id": "R2", "name": "High Flood Priority",
     "description": "5 m < flood <= 7 m => HIGH priority",
     "condition": lambda a: 5.0 < a["flood_level"] <= 7.0,
     "action": "priority = HIGH",
     "apply": lambda a: a.update({"priority": "HIGH"})},
    {"id": "R3", "name": "Medium Flood Priority",
     "description": "3 m < flood <= 5 m => MEDIUM priority",
     "condition": lambda a: 3.0 < a["flood_level"] <= 5.0,
     "action": "priority = MEDIUM",
     "apply": lambda a: a.update({"priority": "MEDIUM"})},
    {"id": "R4", "name": "Low Flood Priority",
     "description": "Flood <= 3 m => LOW priority",
     "condition": lambda a: a["flood_level"] <= 3.0,
     "action": "priority = LOW",
     "apply": lambda a: a.update({"priority": "LOW"})},
    {"id": "R5", "name": "Road Blocked Escalation",
     "description": "Road condition < 0.4 => escalate priority one level",
     "condition": lambda a: a.get("road_condition", 1.0) < 0.4 and a.get("priority") in ("HIGH", "MEDIUM"),
     "action": "priority escalated",
     "apply": lambda a: a.update({"priority": {"HIGH": "CRITICAL", "MEDIUM": "HIGH"}.get(a.get("priority"), a.get("priority"))})},
    {"id": "R6", "name": "Dense Population Boost",
     "description": "Population > 2000 => boost to at least HIGH",
     "condition": lambda a: a["population"] > 2000 and a.get("priority") in ("MEDIUM", "LOW"),
     "action": "priority boosted",
     "apply": lambda a: a.update({"priority": "HIGH"})},
    {"id": "R7", "name": "Helicopter Required",
     "description": "Road condition < 0.35 => recommend helicopter",
     "condition": lambda a: a.get("road_condition", 1.0) < 0.35,
     "action": "recommend_helicopter = True",
     "apply": lambda a: a.update({"recommend_helicopter": True})},
    {"id": "R8", "name": "Boat Recommended",
     "description": "Flood > 6 m AND road < 0.6 => recommend boat",
     "condition": lambda a: a["flood_level"] > 6.0 and a.get("road_condition", 1.0) < 0.6,
     "action": "recommend_boat = True",
     "apply": lambda a: a.update({"recommend_boat": True})},
]

PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def forward_chain(areas):
    fired_log = []
    changed = True
    passes = 0
    while changed:
        changed = False
        passes += 1
        for aid, area in areas.items():
            for rule in EVACUATION_RULES:
                before = {k: area.get(k) for k in ("priority", "recommend_helicopter", "recommend_boat")}
                if rule["condition"](area):
                    rule["apply"](area)
                    after = {k: area.get(k) for k in ("priority", "recommend_helicopter", "recommend_boat")}
                    if before != after:
                        fired_log.append({"area": aid, "rule": rule["id"],
                                          "name": rule["name"], "action": rule["action"], "pass": passes})
                        changed = True
    return fired_log


def exercise_2(kb):
    section("EXERCISE 2 — RULE-BASED REASONING (Forward Chaining)")

    subsection("Inference Rules Defined")
    for r in EVACUATION_RULES:
        print(f"  [{r['id']}] {r['name']}")
        print(f"       IF: {r['description']}")
        print(f"       THEN: {r['action']}\n")

    log = forward_chain(kb["areas"])

    subsection("Forward-Chaining Execution Trace")
    for entry in log:
        print(f"  Pass {entry['pass']} | Area {entry['area']} | {entry['rule']}: "
              f"{entry['name']} => {entry['action']}")

    subsection("Derived Priorities & Recommendations")
    print(f"  {'ID':<4} {'Name':<22} {'Priority':<10} {'Helicopter':>11} {'Boat':>6}")
    print(f"  {'-'*4} {'-'*22} {'-'*10} {'-'*11} {'-'*6}")
    for aid, a in kb["areas"].items():
        heli = "Yes" if a.get("recommend_helicopter") else "-"
        boat = "Yes" if a.get("recommend_boat")        else "-"
        print(f"  {aid:<4} {a['name']:<22} {a.get('priority','?'):<10} {heli:>11} {boat:>6}")

    sorted_areas = sorted(kb["areas"].items(),
                          key=lambda x: PRIORITY_ORDER.get(x[1].get("priority", "LOW"), 9))
    subsection("Evacuation Priority Order")
    for rank, (aid, a) in enumerate(sorted_areas, 1):
        print(f"  {rank}. [{a.get('priority','?'):<8}] {a['name']} (pop={a['population']})")

    _plot_rules(kb, log)
    return [aid for aid, _ in sorted_areas]


def _plot_rules(kb, log):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exercise 2 — Rule-Based Reasoning Results", fontsize=14,
                 fontweight="bold", color=COLORS["dark"])
    fig.patch.set_facecolor(COLORS["white"])

    ax = axes[0]
    areas = list(kb["areas"].keys())
    p_colors = {"CRITICAL": COLORS["red"], "HIGH": COLORS["orange"],
                "MEDIUM": COLORS["yellow"], "LOW": COLORS["green"]}
    pops = [kb["areas"][a]["population"] for a in areas]
    cols = [p_colors.get(kb["areas"][a].get("priority", "LOW"), COLORS["blue"]) for a in areas]
    bars = ax.bar(areas, pops, color=cols)
    ax.set_title("Population by Evacuation Priority", color=COLORS["dark"])
    ax.set_ylabel("Population"); ax.set_facecolor("#EEF4F7")
    patches = [mpatches.Patch(color=c, label=l) for l, c in p_colors.items()]
    ax.legend(handles=patches, fontsize=8)
    for bar, a in zip(bars, areas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                kb["areas"][a].get("priority", "?"), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax = axes[1]
    rule_counts = defaultdict(int)
    for e in log:
        rule_counts[e["rule"]] += 1
    rids   = list(rule_counts.keys())
    counts = [rule_counts[r] for r in rids]
    ax.barh(rids, counts, color=COLORS["blue"])
    ax.set_title("Rule Firing Frequency", color=COLORS["dark"])
    ax.set_xlabel("Times Fired"); ax.set_facecolor("#EEF4F7")

    plt.tight_layout()
    save_fig(fig, "ex2_rule_based_reasoning.png")


def _trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x <= b: return (x - a) / (b - a)
    return (c - x) / (c - b)

def _trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if x <= b: return (x - a) / (b - a)
    if x <= c: return 1.0
    return (d - x) / (d - c)

FLOOD_MF = {
    "low":      lambda f: _trapmf(f, 0, 0, 2, 4),
    "moderate": lambda f: _trimf(f, 2, 5, 8),
    "severe":   lambda f: _trapmf(f, 6, 8, 10, 10),
}
ROAD_MF = {
    "blocked":  lambda r: _trapmf(r, 0, 0, 0.2, 0.4),
    "damaged":  lambda r: _trimf(r, 0.2, 0.5, 0.8),
    "passable": lambda r: _trapmf(r, 0.6, 0.8, 1.0, 1.0),
}
URGENCY_RULES = [
    ("severe",   "blocked",  9.5), ("severe",   "damaged",  8.5),
    ("severe",   "passable", 7.0), ("moderate", "blocked",  7.5),
    ("moderate", "damaged",  6.0), ("moderate", "passable", 4.5),
    ("low",      "blocked",  5.0), ("low",      "damaged",  3.0),
    ("low",      "passable", 1.5),
]


def fuzzy_urgency(flood_level, road_condition):
    flood_mem = {k: FLOOD_MF[k](flood_level)   for k in FLOOD_MF}
    road_mem  = {k: ROAD_MF[k](road_condition) for k in ROAD_MF}
    fired = []
    numerator = denominator = 0.0
    for (ft, rt, centre) in URGENCY_RULES:
        strength = min(flood_mem[ft], road_mem[rt])
        if strength > 0:
            fired.append((ft, rt, centre, strength))
            numerator   += strength * centre
            denominator += strength
    crisp = numerator / denominator if denominator > 0 else 0.0
    return crisp, fired


def urgency_label(score):
    if score >= 8.0: return "EXTREME"
    if score >= 6.5: return "SEVERE"
    if score >= 5.0: return "HIGH"
    if score >= 3.5: return "MODERATE"
    return "LOW"


def exercise_3(kb):
    section("EXERCISE 3 — HANDLING UNCERTAINTY (Fuzzy Logic)")
    print("\n  MF: Flood(0-10m): low[0-4], moderate[2-8], severe[6-10]")
    print("  MF: Road(0-1): blocked[0-0.4], damaged[0.2-0.8], passable[0.6-1.0]")
    print("  Output: 9 Mamdani rules => centroid defuzzification\n")

    subsection("Fuzzy Urgency per Area")
    print(f"  {'Area':<4} {'Name':<22} {'Flood':>6} {'Road':>5} {'Urgency':>8} {'Label':<10}")
    print(f"  {'-'*4} {'-'*22} {'-'*6} {'-'*5} {'-'*8} {'-'*10}")
    area_urgency = {}
    for aid, a in kb["areas"].items():
        score, fired = fuzzy_urgency(a["flood_level"], a["road_condition"])
        label = urgency_label(score)
        area_urgency[aid] = {"score": score, "label": label, "fired": fired}
        print(f"  {aid:<4} {a['name']:<22} {a['flood_level']:>6.1f} "
              f"{a['road_condition']:>5.2f} {score:>8.2f} {label:<10}")

    subsection("Fired Rules for A1 (Downtown Core)")
    for ft, rt, centre, strength in area_urgency["A1"]["fired"]:
        print(f"  IF flood={ft:>8} AND road={rt:>9} => urgency~{centre:>4.1f}  (strength={strength:.3f})")
    print(f"\n  Crisp urgency A1 = {area_urgency['A1']['score']:.3f} => {area_urgency['A1']['label']}")

    _plot_fuzzy(kb, area_urgency)


def _plot_fuzzy(kb, urgencies):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Exercise 3 — Fuzzy Logic for Uncertainty Handling",
                 fontsize=14, fontweight="bold", color=COLORS["dark"])
    fig.patch.set_facecolor(COLORS["white"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 10, 300)
    ax1.plot(x, [FLOOD_MF["low"](v)      for v in x], color=COLORS["green"],  label="Low")
    ax1.plot(x, [FLOOD_MF["moderate"](v) for v in x], color=COLORS["orange"], label="Moderate")
    ax1.plot(x, [FLOOD_MF["severe"](v)   for v in x], color=COLORS["red"],    label="Severe")
    ax1.set_title("Flood Level MF", color=COLORS["dark"])
    ax1.set_xlabel("Flood level (m)"); ax1.set_ylabel("Membership")
    ax1.legend(fontsize=8); ax1.set_facecolor("#EEF4F7")

    ax2 = fig.add_subplot(gs[0, 1])
    r = np.linspace(0, 1, 300)
    ax2.plot(r, [ROAD_MF["blocked"](v)  for v in r], color=COLORS["red"],    label="Blocked")
    ax2.plot(r, [ROAD_MF["damaged"](v)  for v in r], color=COLORS["orange"], label="Damaged")
    ax2.plot(r, [ROAD_MF["passable"](v) for v in r], color=COLORS["green"],  label="Passable")
    ax2.set_title("Road Condition MF", color=COLORS["dark"])
    ax2.set_xlabel("Road condition (0-1)"); ax2.set_ylabel("Membership")
    ax2.legend(fontsize=8); ax2.set_facecolor("#EEF4F7")

    ax3 = fig.add_subplot(gs[0, 2])
    aids   = list(urgencies.keys())
    scores = [urgencies[a]["score"] for a in aids]
    u_col  = {"EXTREME": COLORS["red"], "SEVERE": COLORS["orange"],
               "HIGH": COLORS["yellow"], "MODERATE": COLORS["green"], "LOW": COLORS["blue"]}
    cols   = [u_col[urgencies[a]["label"]] for a in aids]
    bars   = ax3.bar(aids, scores, color=cols)
    ax3.axhline(8.0, ls="--", color=COLORS["red"],    lw=1, label="EXTREME >= 8")
    ax3.axhline(6.5, ls="--", color=COLORS["orange"], lw=1, label="SEVERE  >= 6.5")
    ax3.axhline(5.0, ls="--", color=COLORS["yellow"], lw=1, label="HIGH    >= 5")
    ax3.set_title("Fuzzy Urgency Score per Area", color=COLORS["dark"])
    ax3.set_ylabel("Urgency (0-10)"); ax3.set_ylim(0, 11)
    ax3.legend(fontsize=7); ax3.set_facecolor("#EEF4F7")
    for bar, sc in zip(bars, scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{sc:.1f}", ha="center", va="bottom", fontsize=9)

    ax4 = fig.add_subplot(gs[1, 0])
    fl_range = np.linspace(0, 10, 50)
    rd_range = np.linspace(0, 1,  50)
    Z = np.array([[fuzzy_urgency(f, rr)[0] for rr in rd_range] for f in fl_range])
    im = ax4.contourf(rd_range, fl_range, Z, levels=20, cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax4, label="Urgency score")
    ax4.set_title("Urgency Surface (Flood x Road)", color=COLORS["dark"])
    ax4.set_xlabel("Road condition"); ax4.set_ylabel("Flood level (m)")

    ax5 = fig.add_subplot(gs[1, 1])
    for aid, a in kb["areas"].items():
        sc = urgencies[aid]["score"]
        ax5.scatter(a["road_condition"], a["flood_level"], s=200,
                    c=[sc], cmap="RdYlGn_r", vmin=0, vmax=10, zorder=3, edgecolors="k")
        ax5.annotate(aid, (a["road_condition"], a["flood_level"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 10)
    ax5.set_title("Areas on Risk Plane", color=COLORS["dark"])
    ax5.set_xlabel("Road condition"); ax5.set_ylabel("Flood level (m)")
    ax5.set_facecolor("#EEF4F7")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 10))
    sm.set_array([])
    plt.colorbar(sm, ax=ax5, label="Urgency")

    ax6 = fig.add_subplot(gs[1, 2])
    label_cnt = defaultdict(int)
    for v in urgencies.values():
        label_cnt[v["label"]] += 1
    order = ["EXTREME", "SEVERE", "HIGH", "MODERATE", "LOW"]
    non_zero = [(l, label_cnt[l]) for l in order if label_cnt[l] > 0]
    if non_zero:
        lbs, vs = zip(*non_zero)
        explode = [0.07] * len(lbs)
        ax6.pie(vs, labels=lbs, explode=explode,
                colors=[u_col[l] for l in lbs], autopct="%1.0f%%", startangle=90)
    ax6.set_title("Urgency Label Distribution", color=COLORS["dark"])

    save_fig(fig, "ex3_fuzzy_logic_uncertainty.png")


def allocate_resources(kb, priority_order):
    assignments = []
    area_teams   = {}
    area_shelter = {}

    for aid in priority_order:
        area = kb["areas"][aid]
        need_heli = area.get("recommend_helicopter", False)
        need_boat = area.get("recommend_boat", False)
        best_team = None; best_score = -1
        for tid, team in kb["rescue_teams"].items():
            if not team["available"]: continue
            cap_score = team["capacity"] * team["speed"]
            if need_heli and team["type"] == "helicopter": cap_score *= 1.5
            elif need_boat and team["type"] == "boat":     cap_score *= 1.3
            if cap_score > best_score:
                best_score = cap_score; best_team = tid
        if best_team:
            kb["rescue_teams"][best_team]["available"] = False
            kb["rescue_teams"][best_team]["assigned_area"] = aid
            area_teams[aid] = best_team

    for aid in priority_order:
        area = kb["areas"][aid]
        if aid not in area_teams:
            area_shelter[aid] = None; continue
        best_shlt = None; best_score = -1
        for sid, shelter in kb["shelters"].items():
            if not shelter["available"]: continue
            rem = shelter["capacity"] - shelter["occupied"]
            if rem <= 0: continue
            road_pass = kb["road_network"].get((aid, sid), 0.0)
            score = road_pass * min(rem / area["population"], 1.0)
            if score > best_score:
                best_score = score; best_shlt = sid
        if best_shlt:
            team = kb["rescue_teams"][area_teams[aid]]
            evacuees = min(team["capacity"], area["population"],
                           kb["shelters"][best_shlt]["capacity"] - kb["shelters"][best_shlt]["occupied"])
            kb["shelters"][best_shlt]["occupied"] += evacuees
            area["evacuated"] = evacuees
            area_shelter[aid] = best_shlt
            if kb["shelters"][best_shlt]["occupied"] >= kb["shelters"][best_shlt]["capacity"]:
                kb["shelters"][best_shlt]["available"] = False
        else:
            area_shelter[aid] = None

    for aid in priority_order:
        area  = kb["areas"][aid]
        tid   = area_teams.get(aid)
        sid   = area_shelter.get(aid)
        team  = kb["rescue_teams"][tid] if tid else None
        shelt = kb["shelters"][sid]     if sid else None
        road_pass = kb["road_network"].get((aid, sid), 0.5) if sid else 0.2
        speed     = team["speed"] if team else 0.3
        evac_time = round(10 / (speed * road_pass + 0.01), 1)
        assignments.append({
            "area_id":      aid,
            "area_name":    area["name"],
            "priority":     area.get("priority", "?"),
            "population":   area["population"],
            "evacuated":    area.get("evacuated", 0),
            "team_id":      tid,
            "team_name":    team["name"]  if team  else "NONE",
            "team_type":    team["type"]  if team  else "-",
            "shelter_id":   sid,
            "shelter_name": shelt["name"] if shelt else "NONE",
            "evac_time":    evac_time,
            "road_pass":    road_pass,
        })
    return assignments


def exercise_4(kb, priority_order):
    section("EXERCISE 4 — RESOURCE ALLOCATION")
    assignments = allocate_resources(kb, priority_order)

    subsection("Assignment Results")
    print(f"  {'Area':<4} {'Priority':<9} {'Team':<16} {'Type':<12} {'Shelter':<26} {'Evac.':>6} {'Pop':>5} {'Time':>6}")
    print(f"  {'-'*90}")
    for a in assignments:
        print(f"  {a['area_id']:<4} {a['priority']:<9} {a['team_name']:<16} "
              f"{a['team_type']:<12} {a['shelter_name']:<26} "
              f"{a['evacuated']:>6} {a['population']:>5} {a['evac_time']:>6.1f}")

    subsection("Shelter Utilisation")
    print(f"  {'Shelter':<26} {'Capacity':>9} {'Occupied':>9} {'Remaining':>10} {'Fill%':>7}")
    print(f"  {'-'*65}")
    for sid, s in kb["shelters"].items():
        rem  = s["capacity"] - s["occupied"]
        fill = (s["occupied"] / s["capacity"]) * 100
        print(f"  {s['name']:<26} {s['capacity']:>9} {s['occupied']:>9} {rem:>10} {fill:>7.1f}%")

    total_pop  = sum(a["population"] for a in kb["areas"].values())
    total_evac = sum(a["evacuated"]  for a in kb["areas"].values())
    print(f"\n  Total population: {total_pop}  |  Evacuated: {total_evac}  "
          f"({total_evac/total_pop*100:.1f}%)")

    _plot_allocation(kb, assignments)
    return assignments


def _plot_allocation(kb, assignments):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Exercise 4 — Resource Allocation Results", fontsize=14,
                 fontweight="bold", color=COLORS["dark"])
    fig.patch.set_facecolor(COLORS["white"])

    ax = axes[0]
    names = [a["area_name"].split()[0] for a in assignments]
    pops  = [a["population"] for a in assignments]
    evacs = [a["evacuated"]  for a in assignments]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, pops,  w, label="Population", color=COLORS["blue"])
    ax.bar(x + w/2, evacs, w, label="Evacuated",  color=COLORS["green"])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("Population vs Evacuated per Area", color=COLORS["dark"])
    ax.set_ylabel("People"); ax.legend(fontsize=8); ax.set_facecolor("#EEF4F7")

    ax = axes[1]
    p_col = {"CRITICAL": COLORS["red"], "HIGH": COLORS["orange"],
              "MEDIUM": COLORS["yellow"], "LOW": COLORS["green"]}
    times = [a["evac_time"] for a in assignments]
    cols  = [p_col.get(a["priority"], COLORS["blue"]) for a in assignments]
    bars  = ax.bar(names, times, color=cols)
    ax.set_title("Estimated Evacuation Time per Area", color=COLORS["dark"])
    ax.set_ylabel("Time (units)"); ax.set_facecolor("#EEF4F7")
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                f"{t:.1f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis='x', rotation=15)

    ax = axes[2]
    snames = [s["name"].split()[0] for s in kb["shelters"].values()]
    caps   = [s["capacity"] for s in kb["shelters"].values()]
    occ    = [s["occupied"]  for s in kb["shelters"].values()]
    x2 = np.arange(len(snames)); w2 = 0.35
    ax.bar(x2 - w2/2, caps, w2, label="Capacity",  color=COLORS["light"])
    ax.bar(x2 + w2/2, occ,  w2, label="Occupied",  color=COLORS["purple"])
    ax.set_xticks(x2); ax.set_xticklabels(snames, rotation=15, ha="right")
    ax.set_title("Shelter Capacity vs Occupied", color=COLORS["dark"])
    ax.set_ylabel("People"); ax.legend(fontsize=8); ax.set_facecolor("#EEF4F7")

    plt.tight_layout()
    save_fig(fig, "ex4_resource_allocation.png")


class RescueAgent:
    def __init__(self, team_id, team_data):
        self.id    = team_id
        self.name  = team_data["name"]
        self.type  = team_data["type"]
        self.cap   = team_data["capacity"]
        self.speed = team_data["speed"]
        self.busy  = False
        self.task  = None
        self.log   = []

    def evaluate_task(self, task):
        if self.busy: return -1.0
        base  = self.cap * self.speed
        road  = task.get("road_passability", 0.5)
        bonus = 1.0
        if task.get("need_heli") and self.type == "helicopter": bonus = 1.5
        if task.get("need_boat") and self.type == "boat":       bonus = 1.4
        return round(base * road * bonus, 3)

    def accept_award(self, task):
        self.busy = True; self.task = task
        self.log.append(f"Accepted task: {task['task_id']}")

    def report_complete(self, people_moved):
        self.busy = False
        result = {"agent": self.id, "task_id": self.task["task_id"], "people_moved": people_moved}
        self.log.append(f"Completed {self.task['task_id']} | moved={people_moved}")
        self.task = None
        return result


class CNPManager:
    def __init__(self, agents, kb):
        self.agents  = agents
        self.kb      = kb
        self.history = []

    def _announce(self, task):
        bids = []
        for agent in self.agents:
            bid_val = agent.evaluate_task(task)
            if bid_val >= 0:
                bids.append({"agent": agent, "bid": bid_val})
        return sorted(bids, key=lambda b: b["bid"], reverse=True)

    def _award(self, task, bids):
        if not bids: return None
        winner = bids[0]["agent"]
        winner.accept_award(task)
        self.history.append({
            "task_id":    task["task_id"],
            "area":       task["area_id"],
            "winner":     winner.id,
            "winner_name": winner.name,
            "bid":        bids[0]["bid"],
            "runners_up": [(b["agent"].id, b["bid"]) for b in bids[1:3]],
        })
        return winner

    def run(self, tasks):
        results = []
        for task in tasks:
            bids   = self._announce(task)
            winner = self._award(task, bids)
            if winner is None:
                results.append({"task_id": task["task_id"], "area": task["area_id"],
                                 "agent": "NONE", "people_moved": 0, "status": "NO_BID"})
                continue
            shelter = task.get("shelter_id")
            shlt_cap = 0
            if shelter and shelter in self.kb["shelters"]:
                s = self.kb["shelters"][shelter]
                shlt_cap = s["capacity"] - s["occupied"]
            people_moved = min(winner.cap, task["population"], max(shlt_cap, 0))
            outcome = winner.report_complete(people_moved)
            outcome["status"] = "COMPLETED"; outcome["area"] = task["area_id"]
            results.append(outcome)
        return results

    def print_history(self):
        subsection("CNP Contract Award Log")
        for h in self.history:
            ru = ", ".join(f"{aid}({bid:.1f})" for aid, bid in h.get("runners_up", []))
            print(f"  Task {h['task_id']} [{h['area']}] => Winner: {h['winner_name']} "
                  f"(bid={h['bid']:.1f})  |  Runners-up: {ru or '-'}")


def exercise_5(kb, assignments):
    section("EXERCISE 5 — MULTI-AGENT COORDINATION (Contract-Net Protocol)")
    print("\n  Protocol: Contract-Net Protocol (CNP)")
    print("  Steps: Manager ANNOUNCES task => Agents BID => Manager AWARDS")
    print("         => Winner EXECUTES => Reports BACK\n")

    agents = [RescueAgent(tid, tdata) for tid, tdata in kb["rescue_teams"].items()]
    tasks  = []
    for idx, a in enumerate(assignments):
        tasks.append({
            "task_id":         f"TSK-{idx+1:02d}",
            "area_id":         a["area_id"],
            "population":      a["population"],
            "road_passability":a["road_pass"],
            "shelter_id":      a["shelter_id"],
            "need_boat":       kb["areas"][a["area_id"]].get("recommend_boat", False),
            "need_heli":       kb["areas"][a["area_id"]].get("recommend_helicopter", False),
        })

    manager = CNPManager(agents, kb)
    results = manager.run(tasks)
    manager.print_history()

    subsection("Task Execution Results")
    print(f"  {'Task':<10} {'Area':<4} {'Agent':<5} {'Moved':>7} {'Status':<12}")
    print(f"  {'-'*40}")
    for r in results:
        print(f"  {r['task_id']:<10} {r['area']:<4} {r['agent']:<5} "
              f"{r['people_moved']:>7} {r['status']:<12}")

    subsection("Agent Logs")
    for agent in agents:
        if agent.log:
            print(f"  [{agent.id}] {agent.name}:")
            for entry in agent.log:
                print(f"       -> {entry}")

    subsection("Conflict Resolution Note")
    print("  When multiple CRITICAL areas bid for the same helicopter/boat team:")
    print("  CNP resolves the conflict by awarding the contract to the task with the")
    print("  highest bid score (capability x road passability x type-fit bonus).")
    print("  Losing areas are assigned the next-best available team automatically.")

    _plot_cnp(results, manager.history)
    return results


def _plot_cnp(results, history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Exercise 5 — Multi-Agent CNP Coordination", fontsize=14,
                 fontweight="bold", color=COLORS["dark"])
    fig.patch.set_facecolor(COLORS["white"])

    ax = axes[0]
    task_ids = [h["task_id"] for h in history]
    bids     = [h["bid"]     for h in history]
    ax.bar(task_ids, bids, color=COLORS["blue"])
    ax.set_title("Winning Bid per Task", color=COLORS["dark"])
    ax.set_ylabel("Bid Score"); ax.set_facecolor("#EEF4F7")
    ax.tick_params(axis='x', rotation=30)

    ax = axes[1]
    agent_cnt = defaultdict(int)
    for h in history:
        agent_cnt[h["winner"]] += 1
    if agent_cnt:
        ax.pie(list(agent_cnt.values()), labels=list(agent_cnt.keys()),
               autopct="%1.0f%%", startangle=90,
               colors=[COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"]])
    ax.set_title("Task Distribution per Agent", color=COLORS["dark"])

    ax = axes[2]
    moved_ids  = [r["task_id"]      for r in results]
    moved_vals = [r["people_moved"] for r in results]
    cols = [COLORS["green"] if r["status"] == "COMPLETED" else COLORS["red"] for r in results]
    ax.bar(moved_ids, moved_vals, color=cols)
    ax.set_title("People Moved per Task", color=COLORS["dark"])
    ax.set_ylabel("People"); ax.set_facecolor("#EEF4F7")
    ax.tick_params(axis='x', rotation=30)
    ax.legend(handles=[mpatches.Patch(color=COLORS["green"], label="Completed"),
                        mpatches.Patch(color=COLORS["red"],   label="No Bid")], fontsize=8)

    plt.tight_layout()
    save_fig(fig, "ex5_cnp_coordination.png")


def _run_scenario(scenario_name, kb_snapshot):
    forward_chain(kb_snapshot["areas"])
    sorted_areas   = sorted(kb_snapshot["areas"].items(),
                            key=lambda x: PRIORITY_ORDER.get(x[1].get("priority", "LOW"), 9))
    priority_order = [aid for aid, _ in sorted_areas]
    assignments    = allocate_resources(kb_snapshot, priority_order)
    total_pop  = sum(a["population"]        for a in kb_snapshot["areas"].values())
    total_evac = sum(a.get("evacuated", 0)  for a in kb_snapshot["areas"].values())
    teams_used = sum(1 for t in kb_snapshot["rescue_teams"].values() if not t["available"])
    avg_time   = sum(a["evac_time"] for a in assignments) / max(len(assignments), 1)
    fill_rate  = (sum(s["occupied"] for s in kb_snapshot["shelters"].values()) /
                  max(sum(s["capacity"] for s in kb_snapshot["shelters"].values()), 1))
    return {
        "scenario": scenario_name, "total_pop": total_pop, "evacuated": total_evac,
        "evac_pct": total_evac / total_pop * 100, "teams_used": teams_used,
        "avg_time": avg_time, "fill_rate": fill_rate * 100, "assignments": assignments,
    }


def _print_scenario_summary(res):
    print(f"  Population   : {res['total_pop']}")
    print(f"  Evacuated    : {res['evacuated']} ({res['evac_pct']:.1f}%)")
    print(f"  Teams Used   : {res['teams_used']}")
    print(f"  Avg Evac Time: {res['avg_time']:.2f} units")
    print(f"  Shelter Fill : {res['fill_rate']:.1f}%")
    print(f"\n  {'Area':<4} {'Priority':<9} {'Team':<16} {'Shelter':<26} {'Evac':>6} {'Pop':>5}")
    print(f"  {'-'*70}")
    for a in res["assignments"]:
        print(f"  {a['area_id']:<4} {a['priority']:<9} {a['team_name']:<16} "
              f"{a['shelter_name']:<26} {a['evacuated']:>6} {a['population']:>5}")


def exercise_6(kb_original):
    section("EXERCISE 6 — SIMULATION & EVALUATION")

    subsection("Scenario A: Normal Conditions")
    kb_a = deepcopy(kb_original)
    res_a = _run_scenario("Normal Conditions", kb_a)
    _print_scenario_summary(res_a)

    subsection("Scenario B: Unexpected Events")
    kb_b = deepcopy(kb_original)
    print("  [Event 1] Blocked route: A1 -> S1 passability drops to 0.05")
    kb_b["road_network"][("A1", "S1")] = 0.05
    print("  [Event 2] S2 (North High School) UNAVAILABLE (structural damage)")
    kb_b["shelters"]["S2"]["available"] = False; kb_b["shelters"]["S2"]["capacity"] = 0
    print("  [Event 3] T3 (Charlie Corps helicopter) UNAVAILABLE (mechanical fault)\n")
    kb_b["rescue_teams"]["T3"]["available"] = False
    res_b = _run_scenario("Disrupted Scenario", kb_b)
    _print_scenario_summary(res_b)

    subsection("Scenario C: Worst-Case Escalation")
    kb_c = deepcopy(kb_original)
    print("  [Event 1] Flash floods: all flood levels +2 m")
    for a in kb_c["areas"].values():
        a["flood_level"] = min(a["flood_level"] + 2.0, 10.0)
    print("  [Event 2] Road deterioration: all conditions -0.2\n")
    for a in kb_c["areas"].values():
        a["road_condition"] = max(a["road_condition"] - 0.2, 0.05)
    for k in kb_c["road_network"]:
        kb_c["road_network"][k] = max(kb_c["road_network"][k] - 0.2, 0.05)
    res_c = _run_scenario("Worst Case", kb_c)
    _print_scenario_summary(res_c)

    subsection("Comparative Analysis")
    scenarios = [res_a, res_b, res_c]
    print(f"  {'Scenario':<26} {'Evacuated':>10} {'Evac%':>7} {'Teams':>7} {'Avg Time':>10} {'Shelter Fill':>13}")
    print(f"  {'-'*80}")
    for s in scenarios:
        print(f"  {s['scenario']:<26} {s['evacuated']:>10} {s['evac_pct']:>7.1f}% "
              f"{s['teams_used']:>7} {s['avg_time']:>10.2f} {s['fill_rate']:>12.1f}%")

    _plot_simulation(scenarios)


def _plot_simulation(scenarios):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Exercise 6 — Simulation & Evaluation: Scenario Comparison",
                 fontsize=14, fontweight="bold", color=COLORS["dark"])
    fig.patch.set_facecolor(COLORS["white"])

    labels  = [s["scenario"] for s in scenarios]
    x       = np.arange(len(labels))
    sc_cols = [COLORS["green"], COLORS["orange"], COLORS["red"]]

    def bar_plot(ax, values, title, ylabel, fmt_fn=None):
        bars = ax.bar(x, values, color=sc_cols)
        ax.set_title(title, color=COLORS["dark"]); ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
        ax.set_facecolor("#EEF4F7")
        for b, v in zip(bars, values):
            txt = fmt_fn(v) if fmt_fn else str(v)
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(values)*0.01,
                    txt, ha="center", va="bottom", fontsize=9)

    bar_plot(axes[0][0], [s["evacuated"]  for s in scenarios], "People Evacuated",   "Count")
    bar_plot(axes[0][1], [s["evac_pct"]   for s in scenarios], "Evacuation Rate (%)", "%",    lambda v: f"{v:.1f}%")
    bar_plot(axes[0][2], [s["teams_used"] for s in scenarios], "Rescue Teams Used",  "Teams")
    bar_plot(axes[1][0], [s["avg_time"]   for s in scenarios], "Avg Evacuation Time","Units", lambda v: f"{v:.2f}")
    bar_plot(axes[1][1], [s["fill_rate"]  for s in scenarios], "Shelter Fill Rate",  "%",    lambda v: f"{v:.1f}%")

    ax_r = fig.add_subplot(2, 3, 6, polar=True)
    axes[1][2].set_visible(False)
    metrics = ["Evac%", "Teams\nUsed", "Speed\n(inv)", "Shelter\nFill"]
    max_vals = [100, 5, 30, 100]
    scenario_data = [
        [s["evac_pct"], s["teams_used"], max(0, 30 - s["avg_time"]), s["fill_rate"]]
        for s in scenarios
    ]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    for s, dat, col in zip(scenarios, scenario_data, sc_cols):
        normalized = [v / m * 10 for v, m in zip(dat, max_vals)]
        normalized += normalized[:1]
        ax_r.plot(angles, normalized, color=col, linewidth=2, label=s["scenario"])
        ax_r.fill(angles, normalized, color=col, alpha=0.1)
    ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(metrics, fontsize=9)
    ax_r.set_ylim(0, 10)
    ax_r.set_title("Overall Resilience Radar", pad=20, color=COLORS["dark"])
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax_r.set_facecolor("#EEF4F7")

    plt.tight_layout()
    save_fig(fig, "ex6_simulation_evaluation.png")


def main():
    print("*" * 72)
    print("  EXPERIMENT 11 — PROJECT LAST EXIT")
    print("  Intelligent Disaster Evacuation System")
    print("  Student ID: U23AI118")
    print("*" * 72)

    kb_fresh = deepcopy(KNOWLEDGE_BASE)
    exercise_1(kb_fresh)

    kb_ex2 = deepcopy(KNOWLEDGE_BASE)
    priority_order = exercise_2(kb_ex2)

    exercise_3(kb_ex2)

    kb_ex4 = deepcopy(kb_ex2)
    assignments = exercise_4(kb_ex4, priority_order)

    kb_ex5 = deepcopy(kb_ex2)
    exercise_5(kb_ex5, assignments)

    kb_ex6 = deepcopy(KNOWLEDGE_BASE)
    exercise_6(kb_ex6)

    print("\n" + "*" * 72)
    print("  ALL EXERCISES COMPLETED — PNG charts saved to Lab11 folder.")
    print("*" * 72)


if __name__ == "__main__":
    main()
