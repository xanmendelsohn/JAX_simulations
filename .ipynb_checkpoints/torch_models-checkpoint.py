import torch

class LinearNet(torch.nn.Module):
    def __init__(self, n_outputs, n_inputs, norm=False):
        super(LinearNet, self).__init__()
        self.net = torch.nn.Linear(n_inputs, n_outputs, bias=False)
        self.norm = torch.nn.LayerNorm(
            dim_context, elementwise_affine=False) if norm else None

    def forward(self, x):
        '''
        Input:
            - x: context vector with dim_context dimensions, (N, dim_context)
        Output:
            - output: predicted reward for each arm, (N, num_arms)
        '''
        if self.norm:
            x = self.norm(x)
        output = self.net(x)
        return output

    def init_weights(self):
        self.net.reset_parameters()

class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
        # # initialize weights to zero so that bandit does not make 
        # # optimistic start but rather samples randomly
        # torch.nn.init.constant_(self.linear.weight, 0)
        
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

    def init_weights(self):
        self.linear.reset_parameters()
