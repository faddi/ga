import random
import numpy as np

random.seed(100)


def main():
    population_size = 100
    target = 2000

    # create population
    population = [abs(random.random() * 10) for _ in range(population_size)]

    def eval_individual(ind):
        return abs(ind - target)

    generation = 1

    # run main loop
    while True:

        ## eval individuals
        evals = [eval_individual(ind) for ind in population]

        ## select
        selected_indexes = np.argsort(evals)[0 : len(evals) // 2]

        best = population[selected_indexes[0]]
        best_eval = evals[selected_indexes[0]]

        print(f"{generation} - Best: {best}")

        if best_eval < 1e-4:
            print(best)
            break

        new_population = [population[k] for k in selected_indexes]

        ## mutate
        mutated_individuals = [ind + random.random() for ind in new_population]

        population = [] + new_population + mutated_individuals

        generation += 1


if __name__ == "__main__":
    main()
