import numpy as np
import torch

def action_value_init(feature_list, price_list):
    """
    Initiates value function (Q Table)
    """
    n_dict = torch.ones(size=(len(feature_list), len(price_list)), dtype=torch.float32)
    q_dict = torch.ones(size=(len(feature_list), len(price_list)), dtype=torch.float32)

    return (n_dict, q_dict)

def action_value_update(features, price_list, client_group, params, price, sold):
    """
    Updates value function (Q Table)
    """
    n_dict, q_dict = params

    # Function to return prices above 'price'
    def range_above(price):
        return torch.where(price_list >= price, torch.tensor(1., dtype=torch.float32), torch.tensor(0., dtype=torch.float32))

    # Function to return prices below 'price'
    def range_below(price):
        return torch.where(price_list <= price, torch.tensor(1., dtype=torch.float32), torch.tensor(0., dtype=torch.float32))

    # Determine the condition for 'price_ind'
    price_ind = torch.where(sold, range_below(price), range_above(price))

    # Incrementing the counter of the taken action by one
    n_dict[client_group] += price_ind

    # Converting the boolean sold variable to a float value
    r = sold.float()

    # Incrementally updating the action-value estimate
    q_dict[client_group] += price_ind * ((r - q_dict[client_group]) / n_dict[client_group])

    return (n_dict, q_dict)

def beta_init(feature_list, price_list):
    """
    Returns the initial hyperparameters of the beta distribution
    """
    a = torch.ones(size=(len(feature_list), len(price_list)), dtype=torch.float32)
    b = torch.ones(size=(len(feature_list), len(price_list)), dtype=torch.float32)

    return (a, b)

def beta_update(features, price_list, client_group, params, price, sold):
    """
    Calculates the updated hyperparameters of the beta distribution
    """

    a, b = params

    # Function to increment beta by one
    def range_above(price, a, b):
        price_ind = torch.where(price_list >= price, torch.tensor(1., dtype=torch.float32), torch.tensor(0., dtype=torch.float32))
        b[client_group] += price_ind
        return (a, b)

    # Function to increment alpha by one
    def range_below(price, a, b):
        price_ind = torch.where(price_list <= price, torch.tensor(1., dtype=torch.float32), torch.tensor(0., dtype=torch.float32))
        a[client_group] += price_ind
        return (a, b)

    # Determine whether to increment alpha or beta based on 'sold' condition
    if sold:
        return range_below(price, a, b)
    else:
        return range_above(price, a, b)


def simulate_salesdata(N, features, price_list, price_sensitivity_params, seed):
    """
    Simulates a dataset.

    Args:
    - N (int): Number of samples.
    - features (ndarray): Array of features.
    - price_list (list): List of price values.
    - price_sensitivity_params (list): List of price sensitivity parameters.
    - seed (int): Random seed.

    Returns:
    - ndarray: Simulated dataset.
    """

    # Set seed
    torch.manual_seed(seed)

    # Randomly selecting features and prices from the provided lists
    random_indices_feature = torch.randint(features.shape[0], (N,))
    array_feature = features[random_indices_feature]
    array_feature1 = array_feature[:, 0]
    array_feature2 = array_feature[:, 1]

    random_indices_price = torch.randint(len(price_list), (N,))
    array_price = torch.tensor([price_list[i] for i in random_indices_price])

    # Generating random normal values and calculating client price limits
    random_normal = torch.randn(N)
    client_price_limit = torch.tensor([price_sensitivity_params[random_indices_feature[i]][0] +
                                      price_sensitivity_params[random_indices_feature[i]][1] * random_normal[i] for i in range(N)])

    # Determining whether the deal is sold or not based on client price limit
    DK = client_price_limit >= array_price
    array_sale = torch.where(DK, torch.tensor(1), torch.tensor(0))

    # Stacking the arrays to create the dataset
    ds = torch.stack((array_feature1, array_feature2, array_price, array_sale), dim=0)

    return ds

