import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# ==============================
# ZDT1 FUNCTION
# ==============================
def zdt1(individual):
    n = len(individual)
    f1 = individual[0]
    g = 1 + 9 * sum(individual[1:]) / (n - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2

# ==============================
# MANUAL HYPERVOLUME (2D)
# ==============================
def compute_hv(points, ref=(1.1, 1.1)):
    points = sorted(points, key=lambda x: x[0])
    hv = 0.0
    prev_f1 = ref[0]

    for f1, f2 in reversed(points):
        width = prev_f1 - f1
        height = ref[1] - f2
        hv += width * height
        prev_f1 = f1

    return hv

# ==============================
# PARAMETERS
# ==============================
POP_SIZE = 100
GEN = 200
NDIM = 30
CXPB = 0.9
MUTPB = 1.0 / NDIM

# ==============================
# DEAP SETUP
# ==============================
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=15)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20, indpb=1.0/NDIM)

# ==============================
# NSGA-II
# ==============================
def run_nsga2():
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POP_SIZE)

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    pop = toolbox.select(pop, len(pop))

    hv_values = []

    for gen in range(GEN):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, POP_SIZE)

        fits = [ind.fitness.values for ind in pop]
        hv_values.append(compute_hv(fits))

    return pop, hv_values

# ==============================
# SPEA-II
# ==============================
def run_spea2():
    toolbox.register("select_spea", tools.selSPEA2)

    pop = toolbox.population(n=POP_SIZE)

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hv_values = []

    for gen in range(GEN):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select_spea(pop + offspring, k=POP_SIZE)

        fits = [ind.fitness.values for ind in pop]
        hv_values.append(compute_hv(fits))

    return pop, hv_values

# ==============================
# SPLIT POPULATION
# ==============================
def split_population(pop):
    fronts = tools.sortNondominated(pop, len(pop))
    pareto = fronts[0]
    dominated = [ind for front in fronts[1:] for ind in front]
    return pareto, dominated

# ==============================
# PLOT PARETO
# ==============================
def plot_pareto(pareto, dominated, title):
    f1 = np.linspace(0, 1, 200)
    f2 = 1 - np.sqrt(f1)

    plt.figure()

    plt.plot(f1, f2, label="True Pareto Front")

    pf = np.array([ind.fitness.values for ind in pareto])
    plt.scatter(pf[:, 0], pf[:, 1], label="Non-dominated")

    if dominated:
        dom = np.array([ind.fitness.values for ind in dominated])
        plt.scatter(dom[:, 0], dom[:, 1], label="Dominated")

    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# ==============================
# PLOT HV
# ==============================
def plot_hv(nsga_hv, spea_hv):
    plt.figure()
    plt.plot(nsga_hv, label="NSGA-II")
    plt.plot(spea_hv, label="SPEA-II")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume Curve")
    plt.legend()
    plt.grid()
    plt.show()

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("Running NSGA-II...")
    nsga_pop, nsga_hv = run_nsga2()
    nsga_pf, nsga_dom = split_population(nsga_pop)

    print("Running SPEA-II...")
    spea_pop, spea_hv = run_spea2()
    spea_pf, spea_dom = split_population(spea_pop)

    plot_pareto(nsga_pf, nsga_dom, "NSGA-II")
    plot_pareto(spea_pf, spea_dom, "SPEA-II")

    plot_hv(nsga_hv, spea_hv)