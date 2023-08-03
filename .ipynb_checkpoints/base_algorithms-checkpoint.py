from jax import jit, grad, vmap, device_put, random
import jax.numpy as jnp
import jax
# Low level operators
from jax import lax
from functools import partial
import flax.linen as nn
import optax

# Function to initiate value function (Q Table)
def action_value_init(feature_list, price_list):
    """
    Initiates value function (Q Table)
    """
    n_dict = jnp.ones(shape=[len(feature_list), len(price_list)], dtype=jnp.float32)
    q_dict = jnp.ones(shape=[len(feature_list), len(price_list)], dtype=jnp.float32)

    return (n_dict, q_dict)

# Function to update value function (Q Table)
def action_value_update(features, price_list, client_group, params, price, sold):
    """
    Updates value function (Q Table)
    """

    n_dict, q_dict = params

    # Function to return prices above 'price'
    def range_above(price):
        return jnp.where(price_list >= price, 1., 0.)

    # Function to return prices below 'price'
    def range_below(price):
        return jnp.where(price_list <= price, 1., 0.)

    # Determine the condition for 'price_ind'
    price_ind = lax.cond(sold, range_below, range_above, price)

    # Incrementing the counter of the taken action by one
    n_dict = n_dict.at[client_group].add(price_ind)

    # Converting the boolean sold variable to a float value
    r = sold.astype(jnp.float32)

    # Incrementally updating the action-value estimate
    q_dict = q_dict.at[client_group].add(price_ind * ((r - q_dict[client_group]) / n_dict[client_group]))

    return (n_dict, q_dict)

# Function to initiate hyperparameters of the beta distribution
def beta_init(feature_list, price_list):
    """
    Returns the initial hyperparameters of the beta distribution
    """
    a = jnp.ones(shape=[len(feature_list), len(price_list)], dtype=jnp.float32)
    b = jnp.ones(shape=[len(feature_list), len(price_list)], dtype=jnp.float32)

    return (a, b)

# Function to update hyperparameters of the beta distribution
def beta_update(features, price_list, client_group, params, price, sold):
    """
    Calculates the updated hyperparameters of the beta distribution
    """

    a, b = params

    # Function to increment beta by one
    def range_above(price, a, b):
        price_ind = jnp.where(price_list >= price, 1., 0.)
        b = b.at[client_group].add(price_ind)
        return (a, b)

    # Function to increment alpha by one
    def range_below(price, a, b):
        price_ind = jnp.where(price_list <= price, 1., 0.)
        a = a.at[client_group].add(price_ind)
        return (a, b)

    # Determine whether to increment alpha or beta based on 'sold' condition
    return lax.cond(
        sold,
        range_below,
        range_above,
        price,
        a, b
    )

# Function to initiate parameters for EXP3 algorithm
def exp3_init(feature_list, price_list):
    """
    Returns the initial parameters for EXP3 algorithm
    """
    w = jnp.ones(shape=[len(feature_list), len(price_list)], dtype=jnp.float32)

    return w

# Function to update parameters for EXP3 algorithm
def exp3_update(features, price_list, client_group, params, price, sold, gamma=0.1):
    """
    Calculates the updated parameters for EXP3 algorithm
    """

    w = params[client_group]
    p = (1 - gamma) * w / jnp.sum(w) + gamma / len(price_list)

    # Function to update S for lost deal at given price
    def range_above(price, client_group, params, gamma):
        return params

    # Function to update S for lost deal at given price
    def range_below(price, client_group, params, gamma):
        price_ind = jnp.where(price_list <= price, 1., 0.)
        x = price_list * price_ind * p * (gamma / len(price_list))
        params = params.at[client_group].multiply(jnp.exp(x))
        return params

    # Determine whether to update S for lost deal above or below 'price' based on 'sold' condition
    return lax.cond(
        sold,
        range_below,
        range_above,
        price,
        client_group,
        params, gamma
    )

# Function to simulate a dataset
def simulate_salesdata(N, features, price_list, price_sensitivity_parms, seed):
    """
    Simulates a dataset.

    Args:
    - N (int): Number of samples.
    - features (ndarray): Array of features.
    - price_list (list): List of price values.
    - price_sensitivity_parms (list): List of price sensitivity parameters.
    - seed (int): Random seed.

    Returns:
    - ndarray: Simulated dataset.
    """

    # Splitting the random key
    rng_f, rng_p, rng_s = random.split(jax.random.PRNGKey(seed), num=3)

    # Randomly selecting features and prices from the provided lists
    random_indices_feature = jax.random.randint(rng_f, shape=(N,), minval=0, maxval=features.shape[0])
    array_feature = jnp.asarray([features[i] for i in random_indices_feature])
    array_feature1 = jnp.asarray([i[0] for i in array_feature])
    array_feature2 = jnp.asarray([i[1] for i in array_feature])

    random_indices_price = jax.random.randint(rng_p, shape=(N,), minval=0, maxval=len(price_list))
    array_price = jnp.asarray([price_list[i] for i in random_indices_price])

    # Generating random normal values and calculating client price limits
    random_normal = random.normal(rng_s, shape=(N,))
    client_price_limit = jnp.asarray([price_sensitivity_parms[random_indices_feature[i]][0] +
                                      price_sensitivity_parms[random_indices_feature[i]][1] * random_normal[i] for i in range(len(random_indices_feature))])

    # Determining whether the deal is sold or not based on client price limit
    DK = client_price_limit >= array_price
    f1 = lambda: 1
    f2 = lambda: 0
    cond = lambda dk: jax.lax.cond(dk, f1, f2)
    vcond = jax.vmap(cond)
    array_sale = vcond(DK)

    # Stacking the arrays to create the dataset
    ds = jnp.stack([array_feature1, array_feature2, array_price, array_sale], axis=0)

    return ds
