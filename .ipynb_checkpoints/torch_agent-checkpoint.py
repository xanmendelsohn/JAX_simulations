import torch
import torch.distributions.uniform as uniform
import torch.distributions.beta as beta

# Define a base class '_agent' with common methods and placeholders for subclasses
class _agent(object):
    def __init__(self, name):
        self.name = name

    def clear(self):
        raise NotImplementedError

    def choose_price(self, context, price_list):
        raise NotImplementedError

    def update_model(self, X, y, num_iter):
        raise NotImplementedError

# Define a class 'Agent' that inherits from '_agent'
# This class is for agents implementing Deep Q learning
class Agent(_agent):
    def __init__(self,
                 base_model,             # neural network model
                 optimizer,              # optimizer
                 criterion,              # loss function
                 epsilon = 0,            # set eps>0 for epsilon geedy policy
                 explore_first = 0,      # set explore_first >0 to explore in first rounds
                 #decay_lr = False,      # turn on learning rate decay
                 #decay_step=20,         # learning rate decay step
                 name='default'):
        super(Agent, self).__init__(name)

        # Initialize attributes
        self.base_model = base_model
        self.optimizer = optimizer(self.base_model.parameters())
        self.criterion = criterion
        self.epsilon = epsilon
        self.explore_first = explore_first
        #self.decay_lr = decay_lr
        #self.decay_step = decay_step
        self.step = 0

    def clear(self):
        # Clear/reset the model's weights and step count
        self.base_model.init_weights()
        self.step = 0

    def choose_price(self, context, client_group, price_list):
        # Choose a price using the model and exploration strategy
        stacked_context = torch.stack([context] * len(price_list), dim=0)
        X_tmp = torch.cat((stacked_context, price_list.unsqueeze(1)), dim=1)
        with torch.no_grad():
            preds = self.base_model(X_tmp)
            price_index = torch.argmax(preds.squeeze(dim=1)*price_list)

        # Apply epsilon-greedy exploration strategy
        if torch.rand(1) < self.epsilon or self.explore_first > self.step:
            return price_list[torch.randint(len(price_list), (1,)).item()]
        else:
            return price_list[price_index]

    def update_model(self, client_group, price, sold, price_list, X, y, num_iter=5):
        self.step += 1
        self.base_model.train()
        # update using full batch
        # if self.decay_lr:
        #     if self.step % self.decay_step == 0:
        #         self.optimizer.lr = self.base_lr / self.step

        for i in range(num_iter):
            self.base_model.zero_grad()
            pred = self.base_model(X) 
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
        assert not torch.isnan(loss), "Loss is Nan!"
        
# Define another class 'QAgent' inheriting from '_agent'
# This class is for agents implementing Q learning
class QAgent(_agent):
    def __init__(self,
                 context_space,             # neural network model
                 price_list,                # optimizer
                 epsilon = 0,               # set eps>0 for epsilon geedy policy
                 explore_first = 0,         # set explore_first >0 to explore in first rounds
                 policy = "Epsilon Greedy", # ["Epsilon Greedy", "Thompson Sampling"]
                 name='default'):
        super(QAgent, self).__init__(name)

        # Initialize attributes
        self.context_space = context_space
        self.price_list = price_list
        self.epsilon = epsilon
        self.explore_first = explore_first
        self.policy = policy
        self.step = 0
        self.q_dict = torch.ones(size=(len(self.context_space), len(self.price_list)), dtype=torch.float32)
        self.n_dict = torch.ones(size=(len(self.context_space), len(self.price_list)), dtype=torch.float32)
        self.beta_a = torch.ones(size=(len(self.context_space), len(self.price_list)), dtype=torch.float32)
        self.beta_b = torch.ones(size=(len(self.context_space), len(self.price_list)), dtype=torch.float32)


    def clear(self):
    # Clear/reset Q-values and exploration parameters
        self.q_dict = torch.ones(size=(len(self.context_space), len(self.price_list)), dtype=torch.float32)
        self.n_dict = torch.ones(size=(len(self.context_space), len(self.price_list)), dtype=torch.float32)
        self.beta_a = torch.ones(size=(len(self.context_space), len(self.price_list)), dtype=torch.float32)
        self.beta_b = torch.ones(size=(len(self.context_space), len(self.price_list)), dtype=torch.float32)
        self.step = 0
        
    def epsilon_greedy_policy(self, client_group, price_list):
        """
        Randomly selects either the variant with highest action-value,
        or an arbitrary variant.
        """

        # Selecting a random variant
        def explore(client_group, price_list):
            i = torch.randint(len(price_list), (1,)).item()
            return price_list[i]

        # Selecting the variant with the highest action-value estimate
        def exploit(client_group, price_list):
            # maximum expected reward
            exp_reward = self.q_dict[client_group] * price_list
            argmax = price_list[torch.argmax(exp_reward)]
            return argmax

        # Deciding randomly whether to explore or to exploit
        if torch.rand(1) < self.epsilon:
            return explore(client_group, price_list)
        else:
            return exploit(client_group, price_list)
        
    def thompson_sampling_policy(self, client_group, price_list):
        """ 
        Thompson sampling by drawing from conjugate prior beta distribution
        """

        dist = beta.Beta(self.beta_a[client_group], self.beta_b[client_group])
        sampled_price_index = torch.argmax(dist.sample() * price_list)
        price = price_list[sampled_price_index]

        return price

    def choose_price(self, context, client_group, price_list):
        if self.explore_first > self.step:
            return price_list[torch.randint(len(price_list), (1,)).item()]
        elif self.policy == "Thompson Sampling":
            return self.thompson_sampling_policy(client_group, price_list)
        else: 
            return self.epsilon_greedy_policy(client_group, price_list)
        
    # Function to return prices above 'price'    
    def range_above(self, price_list, price):
        return torch.where(price_list >= price, torch.tensor(1., dtype=torch.float32), torch.tensor(0., dtype=torch.float32))

    # Function to return prices below 'price'
    def range_below(self,price_list, price):
        return torch.where(price_list <= price, torch.tensor(1., dtype=torch.float32), torch.tensor(0., dtype=torch.float32))
        
    def update_model(self, client_group, price, sold, price_list, X, y, num_iter=5):
        self.step += 1
        
        # Determine the condition for 'price_ind'
        price_ind = torch.where(sold, self.range_below(price_list, price), self.range_above(price_list, price))

        # Incrementing the counter of the taken action by one
        self.n_dict[client_group] += price_ind

        # Converting the boolean sold variable to a float value
        r = sold.float()

        # Incrementally updating the q- values
        self.q_dict[client_group] += price_ind * ((r - self.q_dict[client_group]) / self.n_dict[client_group])
        
        # Incrementally updating beta parameter values
        if sold:
            self.beta_a[client_group] += torch.where(price_list <= price, torch.tensor(1., dtype=torch.float32), torch.tensor(0., dtype=torch.float32))
        else:
            self.beta_b[client_group] += torch.where(price_list >= price, torch.tensor(1., dtype=torch.float32), torch.tensor(0., dtype=torch.float32))
        


