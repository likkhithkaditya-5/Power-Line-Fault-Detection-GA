import random
import numpy as np
from deap import base, creator, tools
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config

NUM_FEATURE_BITS = 6

# DEAP's creator.create() must only run once per process, or it raises
# an error on re-import / re-run within the same session.
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def set_ga_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)


def make_random_gene(gene_spec):
    gene_type = gene_spec[0]
    low = gene_spec[1]
    high = gene_spec[2]
    if gene_type == "float":
        return random.uniform(low, high)
    if gene_type == "int":
        return random.randint(low, high)
    raise ValueError("Unknown gene type: " + gene_type)


def make_individual(gene_specs):
    mask_bits = []
    for i in range(NUM_FEATURE_BITS):
        mask_bits.append(random.randint(0, 1))

    hyperparam_genes = []
    for i in range(len(gene_specs)):
        hyperparam_genes.append(make_random_gene(gene_specs[i]))

    combined = mask_bits + hyperparam_genes
    return creator.Individual(combined)


def decode_individual(individual, gene_specs, gene_names):
    feature_mask = []
    for i in range(NUM_FEATURE_BITS):
        feature_mask.append(int(round(individual[i])))

    hyperparams = {}
    for i in range(len(gene_specs)):
        gene_type = gene_specs[i][0]
        low = gene_specs[i][1]
        high = gene_specs[i][2]
        raw_value = individual[NUM_FEATURE_BITS + i]
        raw_value = min(max(raw_value, low), high)  # clip defensively
        if gene_type == "int":
            raw_value = int(round(raw_value))
        hyperparams[gene_names[i]] = raw_value

    return feature_mask, hyperparams


def mutate_individual(individual, gene_specs, bit_flip_prob, gene_mutate_prob):
    for i in range(NUM_FEATURE_BITS):
        if random.random() < bit_flip_prob:
            individual[i] = 1 - individual[i]

    for i in range(len(gene_specs)):
        gene_index = NUM_FEATURE_BITS + i
        if random.random() >= gene_mutate_prob:
            continue

        gene_type = gene_specs[i][0]
        low = gene_specs[i][1]
        high = gene_specs[i][2]
        if gene_type == "float":
            span = high - low
            new_value = individual[gene_index] + random.gauss(0, span * 0.1)
            individual[gene_index] = min(max(new_value, low), high)
        else:  # int
            step = random.choice([-3, -2, -1, 1, 2, 3])
            new_value = int(round(individual[gene_index])) + step
            individual[gene_index] = min(max(new_value, low), high)

    return (individual,)


def build_fitness_function(model_builder, gene_specs, gene_names, x_train_full, y_train,
                           cv_folds, penalty_lambda=None):
    if penalty_lambda is None:
        penalty_lambda = config.GA_FEATURE_PENALTY_LAMBDA

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_SEED)

    def evaluate(individual):
        feature_mask, hyperparams = decode_individual(individual, gene_specs, gene_names)

        num_selected = 0
        for bit in feature_mask:
            num_selected = num_selected + bit

        if num_selected == 0:
            return (-1.0,)  # invalid chromosome, worse than any real macro-F1

        selected_columns = []
        for i in range(NUM_FEATURE_BITS):
            if feature_mask[i] == 1:
                selected_columns.append(i)
        x_subset = x_train_full[:, selected_columns]

        model = model_builder(hyperparams)
        cv_scores = cross_val_score(model, x_subset, y_train, cv=splitter, scoring="f1_macro")
        mean_macro_f1 = float(np.mean(cv_scores))

        feature_penalty = penalty_lambda * (num_selected / NUM_FEATURE_BITS)
        fitness = mean_macro_f1 - feature_penalty
        return (fitness,)

    return evaluate


def build_toolbox(gene_specs, evaluate_function):
    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, gene_specs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_function)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register(
        "mutate",
        mutate_individual,
        gene_specs=gene_specs,
        bit_flip_prob=0.15,
        gene_mutate_prob=0.3,
    )
    toolbox.register("select", tools.selTournament, tournsize=config.GA_TOURNAMENT_SIZE)
    return toolbox


def run_ga(toolbox):
    population = toolbox.population(n=config.GA_POPULATION_SIZE)

    fitnesses = []
    for individual in population:
        fitnesses.append(toolbox.evaluate(individual))
    for i in range(len(population)):
        population[i].fitness.values = fitnesses[i]

    best_fitness_per_generation = []
    best_individual = tools.selBest(population, 1)[0]

    for generation in range(config.GA_NUM_GENERATIONS):
        offspring = toolbox.select(population, len(population) - 1)

        cloned_offspring = []
        for individual in offspring:
            cloned_offspring.append(toolbox.clone(individual))
        offspring = cloned_offspring

        for i in range(0, len(offspring) - 1, 2):
            child_1 = offspring[i]
            child_2 = offspring[i + 1]
            if random.random() < config.GA_CROSSOVER_PROB:
                toolbox.mate(child_1, child_2)
                del child_1.fitness.values
                del child_2.fitness.values

        for mutant in offspring:
            if random.random() < config.GA_MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_individuals = []
        for individual in offspring:
            if not individual.fitness.valid:
                invalid_individuals.append(individual)

        invalid_fitnesses = []
        for individual in invalid_individuals:
            invalid_fitnesses.append(toolbox.evaluate(individual))
        for i in range(len(invalid_individuals)):
            invalid_individuals[i].fitness.values = invalid_fitnesses[i]

        # Elitism: keep the single best individual seen so far in the population.
        population = offspring + [best_individual]
        current_best = tools.selBest(population, 1)[0]
        if current_best.fitness.values[0] > best_individual.fitness.values[0]:
            best_individual = toolbox.clone(current_best)

        best_fitness_per_generation.append(best_individual.fitness.values[0])

    return best_individual, best_fitness_per_generation
