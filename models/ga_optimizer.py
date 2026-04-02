import numpy as np
import pandas as pd
import joblib
import random
from deap import base, creator, tools, algorithms

# ── Load RF model and scaler ───────────────────────────────────────────────────
rf     = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/rf_scaler.pkl")

# ── Latest known city state (Dec 2024 — last row of dataset) ──────────────────
df = pd.read_csv("data/carbon_emission_data.csv")
latest = df.iloc[-1]

BASE_STATE = {
    "month_index":         latest["month_index"],
    "temperature_c":       latest["temperature_c"],
    "gdp_index":           latest["gdp_index"],
    "transport_activity":  latest["transport_activity"],
    "industry_activity":   latest["industry_activity"],
    "renewable_share_pct": latest["renewable_share_pct"],
}

BASELINE_EMISSION = latest["total_emission"]

FEATURES = [
    "month_index",
    "temperature_c",
    "gdp_index",
    "transport_activity",
    "industry_activity",
    "renewable_share_pct",
]

# ── Prediction helper ──────────────────────────────────────────────────────────
def predict_emission(transport_red, renewable_inc, industry_eff):
    """
    Apply policy adjustments to base state and predict emission via RF model.
    transport_red  : % reduction in transport activity  (0–20)
    renewable_inc  : % increase in renewable share       (0–20)
    industry_eff   : % reduction in industry activity   (0–10)
    """
    state = BASE_STATE.copy()
    state["transport_activity"]  *= (1 - transport_red  / 100)
    state["renewable_share_pct"] += renewable_inc
    state["industry_activity"]   *= (1 - industry_eff   / 100)

    X = np.array([[state[f] for f in FEATURES]])
    X_sc = scaler.transform(X)
    return rf.predict(X_sc)[0]

# ── GA Setup ───────────────────────────────────────────────────────────────────
# Chromosome: [transport_reduction%, renewable_increase%, industry_efficiency%]
# Bounds:      transport: 0–20,  renewable: 0–20,  industry: 0–10

BOUNDS = [(0, 20), (0, 20), (0, 10)]

# Minimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def random_individual():
    return creator.Individual([
        random.uniform(BOUNDS[0][0], BOUNDS[0][1]),
        random.uniform(BOUNDS[1][0], BOUNDS[1][1]),
        random.uniform(BOUNDS[2][0], BOUNDS[2][1]),
    ])

def evaluate(individual):
    t, r, i = individual
    # Clamp to bounds before evaluating
    t = np.clip(t, *BOUNDS[0])
    r = np.clip(r, *BOUNDS[1])
    i = np.clip(i, *BOUNDS[2])
    emission = predict_emission(t, r, i)
    return (emission,)

toolbox.register("individual",  tools.initIterate, creator.Individual, random_individual)
toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",    evaluate)
toolbox.register("mate",        tools.cxBlend, alpha=0.3)
toolbox.register("mutate",      tools.mutGaussian, mu=0, sigma=1.5, indpb=0.3)
toolbox.register("select",      tools.selTournament, tournsize=3)

# ── Run GA ─────────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

POP_SIZE    = 100
GENERATIONS = 80
CXPB        = 0.7    # crossover probability
MUTPB       = 0.3    # mutation probability

population = toolbox.population(n=POP_SIZE)

print("Running Genetic Algorithm...")
print(f"Population: {POP_SIZE}  |  Generations: {GENERATIONS}")
print(f"Baseline emission: {BASELINE_EMISSION:.4f} M tons\n")

stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("min",  np.min)
stats.register("mean", np.mean)

hof = tools.HallOfFame(1)   # keep best individual ever seen
convergence_history = []    # best RF-predicted emission (Mt) after each generation

for gen in range(1, GENERATIONS + 1):
    # Evaluate fitness
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    hof.update(population)
    convergence_history.append(float(hof[0].fitness.values[0]))

    # Selection
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Mutation
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Clamp to bounds after mutation
    for ind in offspring:
        ind[0] = np.clip(ind[0], *BOUNDS[0])
        ind[1] = np.clip(ind[1], *BOUNDS[1])
        ind[2] = np.clip(ind[2], *BOUNDS[2])

    # Replace population
    population[:] = offspring

    if gen % 10 == 0:
        best_fit = hof[0].fitness.values[0]
        print(f"  Gen {gen:>3}/{GENERATIONS}  |  Best emission: {best_fit:.4f} M tons")

# ── Results ────────────────────────────────────────────────────────────────────
best = hof[0]
best_transport  = np.clip(best[0], *BOUNDS[0])
best_renewable  = np.clip(best[1], *BOUNDS[1])
best_industry   = np.clip(best[2], *BOUNDS[2])
optimized_emission = predict_emission(best_transport, best_renewable, best_industry)
reduction_pct      = (BASELINE_EMISSION - optimized_emission) / BASELINE_EMISSION * 100

print("\n" + "=" * 45)
print("        Genetic Algorithm Results")
print("=" * 45)
print(f"  Reduce transport activity by : {best_transport:.1f}%")
print(f"  Increase renewable share by  : {best_renewable:.1f}%")
print(f"  Improve industrial efficiency: {best_industry:.1f}%")
print("-" * 45)
print(f"  Baseline emission  : {BASELINE_EMISSION:.4f} M tons")
print(f"  Optimized emission : {optimized_emission:.4f} M tons")
print(f"  Emission reduction : {reduction_pct:.1f}%")
print("=" * 45)

# ── Save best strategy ─────────────────────────────────────────────────────────
result = {
    "transport_reduction": round(best_transport, 2),
    "renewable_increase":  round(best_renewable, 2),
    "industry_efficiency": round(best_industry, 2),
    "baseline_emission":   round(BASELINE_EMISSION, 4),
    "optimized_emission":  round(optimized_emission, 4),
    "reduction_pct":       round(reduction_pct, 2),
    "population_size":    POP_SIZE,
    "generations":         GENERATIONS,
    "fitness_evaluations": POP_SIZE * GENERATIONS,
}

import json, os
os.makedirs("models", exist_ok=True)
with open("models/ga_result.json", "w") as f:
    json.dump(result, f, indent=2)

conv_payload = {
    "generation": list(range(1, len(convergence_history) + 1)),
    "best_emission_mt": [round(x, 6) for x in convergence_history],
    "note": "Best RF-predicted emission (Mt) in hall of fame after each generation.",
}
with open("models/ga_convergence.json", "w") as f:
    json.dump(conv_payload, f, indent=2)

print("\nBest strategy saved to models/ga_result.json")
print(f"Convergence trace saved to models/ga_convergence.json ({len(convergence_history)} points)")
