import copy
import random
import numpy as np
import gymnasium as gym

random.seed(100)


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


class Individual:
    def __init__(self, input_size: int, output_size: int) -> None:
        h = 32
        self.l1 = np.random.randn(input_size, h)
        self.b1 = np.random.randn(h)

        self.l2 = np.random.randn(h, h)
        self.b2 = np.random.randn(h)

        self.l3 = np.random.randn(h, output_size)

        self.output_size = output_size

    def select_action(self, input: np.ndarray):
        x = np.dot(input, self.l1) + self.b1
        x = np.tanh(x)

        x = np.dot(x, self.l2) + self.b2
        x = np.tanh(x)

        x = np.dot(x, self.l3)

        out = np.random.choice(self.output_size, p=softmax(x))

        return out

    def mutate(self):
        lr = 1e-1

        self.l1 = self.l1 + np.random.randn(*self.l1.shape) * lr
        self.b1 = self.b1 + np.random.randn(*self.b1.shape) * lr

        self.l2 = self.l2 + np.random.randn(*self.l2.shape) * lr
        self.b2 = self.b2 + np.random.randn(*self.b2.shape) * lr

        self.l3 = self.l3 + np.random.randn(*self.l3.shape) * lr

        return self


def main():
    population_size = 100

    env_name = "CartPole-v1"
    state_space = 4
    action_space = 2

    # env_name = "LunarLander-v2"
    # state_space = 8
    # action_space = 4

    # create population
    population = [Individual(state_space, action_space) for _ in range(population_size)]

    def eval_individual(ind: Individual) -> float:
        env = gym.make(env_name)
        observation, _ = env.reset()
        total = 0.0
        for _ in range(1000):
            action = ind.select_action(observation)

            observation, reward, done, trunc, info = env.step(action)
            total += float(reward)

            if done or trunc:
                return total

        return float("-inf")

    generation = 1

    # run main loop
    while True:
        ## eval individuals
        evals = [eval_individual(ind) for ind in population]

        ## select
        selected_indexes = np.argsort(evals)[::-1][0 : len(evals) // 2]

        best_eval = evals[selected_indexes[0]]

        print(f"{generation} - Best: {best_eval}")

        new_population = [population[k] for k in selected_indexes]

        ## mutate
        mutated_individuals = [copy.deepcopy(ind).mutate() for ind in new_population]

        population = [] + new_population + mutated_individuals

        generation += 1


if __name__ == "__main__":
    main()
