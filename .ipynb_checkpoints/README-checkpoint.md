## Price Optimization Simulations
### finding optimum individual prices with multi-arm contextual bandit policies

### Price Sensitivity Model

The following curve can be interpreted as a depiction of the likelihood of a given client closing a deal. There are two client groups (orange and blue) with different price sensitivities. Prices range between 0 and 1. 

Each point in the diagram is the average percentage of clients who will accept a deal at price x. The blue client group is more price sensitive.

Each client has a stochastic price threshold (the threshold is normally distributed with mean and standard deviation being fixed per client group). The client accepts the deal if the price falls below this threshold.

![png](plt/price_sensitivity_data_simulation.png)
    

The scatterplot above approximates the curve defined by:
    $$1 - CDF(\mu, \sigma)$$
Where CDF denotes the cumulative distribution function of the normal distribution.

    
![png](plt/price_sensitivity_curve.png)
    

#### Optimal Price

The price which we optimize for is the price with the highest expected return, i.e. $$ max \mathbb{E}[reward] = argmax_{price} [ price* \mathbb{P}(deal_{price}) ]$$
The plot below depicts the expected return per price and client group and marks the optimal with a vertical line.


    
![png](plt/expected_reward_curve.png)
    


### Simulation of Online Price Optimization JAX

See the white paper for a description of multi-arm contextual bandits and the policies defined below.

Policies compared include:

- Thomspon Sampling
- Epsilon Greedy
- Upper Confidence Bound
- EXP3

All base-algorithms implement **cross-over learning**, i.e.

- if a deal is lost at price p, lost deals are simulated for all prices >p, ceteris paribus
- if a deal is won at price p, won deals are simulated for all prices <p, ceteris paribus

The first set of simulations implement simple **Q-Learning** and are performed with JAX. 
Developed by Google, Jax is a Python library that is specifically designed for ML research. Unlike Tensorflow and PyTorch, it is built with the functional programming (FP) paradigm, making it highly composable and promoting the concept of pure functions that have no side effects. This means that all state changes, such as parameter updates or splitting of random generators, must be done explicitly. Although this may require more lines of code than their object-oriented programming (OOP) equivalent, it gives developers full control over state changes, leading to an increased understanding of the code and fewer surprises.

Ideas and algorithms behind the simulations can be found here:
https://github.com/xanmendelsohn/Dynamic-Pricing-with-Reinforcement-Learning

The second set of simulations are performed with PyTorch and includes agents which have a base model that maps the context to the value function. This is analogous to **Deep-Q-Learning**.

### Evaluation

#### Reward Evaluation
  
![png](plt/policy_reward_comparison_q_model.png)

#### Regret Evaluation
    
![png](plt/policy_regret_comparison_q_model.png)

### Simulation of Online Price Optimization PyTorch

The second set of simulations is performed with PyTorch and includes agents which have a base model that maps the context to the value function. This is analogous to **Deep-Q-Learning**.

Different exploration policies, optimizers, and learning rates are tested, including:

-  **Langevin Monte Carlo Thompson Sampling (LMC-TS)**, which uses **Markov Chain Monte Carlo (MCMC)** methods to directly sample from the posterior distribution in contextual bandits. (https://arxiv.org/abs/2206.11254)
- Epsilon Greedy Policies

Since the model is updated after every new action-reward pair is obtained, the model training is by construction a Stochastic Gradient Descent. Stochastic gradient descent will always be an approximation of the Thompson Sampling policy because of the stochastic nature of parameter updates.

Model initialization, which is the default random weight and bias initialization implemented by PyTorch, leads to an **optimistic start**. The model assigns all actions an approximately equal success probability so that high prices are allocated the highest expected return. The agent will thus explore high prices first, leading to a larger number of failed deals in the beginning and also a higher exploration rate in the beginning.

All model updates implement **cross-over learning**, i.e.

- if a deal is lost at price p, lost deals are simulated for all prices >p, ceteris paribus
- if a deal is won at price p, won deals are simulated for all prices <p, ceteris paribus

### Evaluation

#### Reward Evaluation

    
![png](plt/policy_reward_comparison_q_model.png)
    

#### Regret Evaluation

    
![png](plt/policy_regret_comparison_q_model.png)
    

