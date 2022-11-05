import copy
import random
import numpy as np
import gym
import ray

random.seed(100)


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


class Individual:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.lr = 1e-1
        h = 32
        self.l1 = np.random.randn(input_size, h)
        self.b1 = np.random.randn(h)

        self.l2 = np.random.randn(h, h)
        self.b2 = np.random.randn(h)

        self.l3 = np.random.randn(h, output_size)

        self.output_size = output_size

    def eval(self, input: np.ndarray):

        x = np.dot(input, self.l1) + self.b1
        x = np.tanh(x)

        x = np.dot(x, self.l2) + self.b2
        x = np.tanh(x)

        x = np.dot(x, self.l3)

        out = np.random.choice(self.output_size, p=softmax(x))

        return out

    def mutate(self):

        self.lr = self.lr * np.exp(np.random.randn() * 0.1)

        self.l1 = self.l1 + np.random.randn(*self.l1.shape) * self.lr
        self.b1 = self.b1 + np.random.randn(*self.b1.shape) * self.lr

        self.l2 = self.l2 + np.random.randn(*self.l2.shape) * self.lr
        self.b2 = self.b2 + np.random.randn(*self.b2.shape) * self.lr

        self.l3 = self.l3 + np.random.randn(*self.l3.shape) * self.lr

        return self


def eval_individual(ind: Individual, env_name: str, render=False) -> float:
    env = gym.make(env_name)
    observation = env.reset()
    total = 0.0
    for _ in range(1000):
        action = ind.eval(observation)

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

    env_name = "CartPole-v1"
    state_space = 4
    action_space = 2

    # env_name = "LunarLander-v2"
    # state_space = 8
    # action_space = 4

    render = False

    # create population
    population = [Individual(state_space, action_space) for _ in range(population_size)]

    generation = 1

    # run main loop
    while True:

        ## eval individuals
        evals = ray.get(
            [remote_eval_individual.remote(ind, env_name) for ind in population]
        )

        ## select
        selected_indexes = np.argsort(evals)[::-1][0 : len(evals) // 2]

        best = population[selected_indexes[0]]
        best_eval = evals[selected_indexes[0]]

        print(f"{generation} - Best: {best_eval} - Mean: {np.mean(evals)}")

        if render:
            print(eval_individual(best, env_name, True))

        new_population = [population[k] for k in selected_indexes]

        ## mutate
        mutated_individuals = [copy.deepcopy(ind).mutate() for ind in new_population]

        population = [] + new_population + mutated_individuals

        generation += 1


if __name__ == "__main__":
    main()
