import numpy as np
import torch

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

