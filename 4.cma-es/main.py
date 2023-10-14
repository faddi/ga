import copy
import random
import numpy as np
import gym
import ray
import cma

random.seed(100)


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


class Individual:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.h = 32
        self.l1 = np.random.randn(input_size, self.h)
        self.b1 = np.random.randn(self.h)

        self.l2 = np.random.randn(self.h, self.h)
        self.b2 = np.random.randn(self.h)

        self.l3 = np.random.randn(self.h, output_size)

        self.output_size = output_size

    def select_action(self, input: np.ndarray):
        x = np.dot(input, self.l1) + self.b1
        x = np.tanh(x)

        x = np.dot(x, self.l2) + self.b2
        x = np.tanh(x)

        x = np.dot(x, self.l3)

        out = np.random.choice(self.output_size, p=softmax(x))

        return out

    def to_vector(self):
        return np.concatenate(
            [
                self.l1.flatten(),
                self.b1.flatten(),
                self.l2.flatten(),
                self.b2.flatten(),
                self.l3.flatten(),
            ]
        )

    def from_vector(self, v):
        vector = np.array(v, copy=True)

        self.l1 = vector[: self.l1.size].reshape(self.l1.shape)
        vector = vector[self.l1.size :]
        self.b1 = vector[: self.b1.size].reshape(self.b1.shape)
        vector = vector[self.b1.size :]
        self.l2 = vector[: self.l2.size].reshape(self.l2.shape)
        vector = vector[self.l2.size :]
        self.b2 = vector[: self.b2.size].reshape(self.b2.shape)
        vector = vector[self.b2.size :]
        self.l3 = vector[: self.l3.size].reshape(self.l3.shape)
        return self


def eval_individual(ind: Individual, env_name: str, render=False) -> float:
    env = gym.make(env_name)
    observation = env.reset()
    total = 0.0
    for _ in range(1000):
        action = ind.select_action(observation)

        if render:
            env.render()

        observation, reward, done, info = env.step(action)
        total += reward

        if done:
            env.close()
            return total

    return float("-inf")


remote_eval_individual = ray.remote(eval_individual)


def main():
    population_size = 100

    # env_name = "CartPole-v1"
    # state_space = 4
    # action_space = 2

    env_name = "LunarLander-v2"
    state_space = 8
    action_space = 4

    render = False

    # create population
    population = [Individual(state_space, action_space) for _ in range(population_size)]

    generation = 1

    seed_ind = population[0]
    sigma0 = 1

    es = cma.CMAEvolutionStrategy(
        seed_ind.to_vector(),
        sigma0,
        {
            "popsize": population_size,
        },
    )

    # run main loop
    while True:
        solutions = es.ask()
        for i, solution in enumerate(solutions):
            population[i].from_vector(solution)

        ## eval individuals
        evals = ray.get(
            [remote_eval_individual.remote(ind, env_name) for ind in population]
        )

        evals = np.array(evals)

        es.tell(solutions, evals * -1)
        es.logger.add()
        es.disp()

        ## select
        selected_indexes = np.argsort(evals)[::-1]
        sorted_evals = evals[selected_indexes]

        best = population[selected_indexes[0]]
        best_eval = sorted_evals[0]
        mean_eval = np.mean(sorted_evals)

        print(f"{generation} - Best: {best_eval:.2f} Mean: {mean_eval:.2f}")

        if render:
            print(eval_individual(best, env_name, True))

        generation += 1


if __name__ == "__main__":
    main()
