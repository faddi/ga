Various genetic algorithm experiments.

**1.base**

Fitness: Match fixed number.  
Genes: single float.  
Mutation: random.random.

**2.gym**

Fitness: Openai gym.  
Genes: nn weights, implemented in numpy.  
Mutation: sample nn weight shifts from normal distribution.

**3.parallelized**

Same as 2.gym parallelized with ray.

Fitness: Openai gym.  
Genes: nn weights, implemented in numpy.  
Mutation: sample nn weight shifts from normal distribution.
