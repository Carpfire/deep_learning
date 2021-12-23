import torch
class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        def drelu(X):
            return torch.where(X > 0, 1, 0)
        def didentity(X):
            n = X.size()[0]
            m = X.size()[1] 
            return torch.ones((n,m))
        def dsigmoid(X):
            return torch.sigmoid(X)*(1-torch.sigmoid(X))
        
        self.funcs = dict(
            relu = torch.nn.functional.relu,
            sigmoid = torch.sigmoid,
            identity = torch.nn.Identity()
        )
        self.derivs = dict(
            relu = drelu,
            identity = didentity,
            sigmoid = dsigmoid, 
        )

        self.f_function = self.funcs[f_function]
        self.g_function = self.funcs[g_function]

        self.back_f_function = self.derivs[f_function]
        self.back_g_function =  self.derivs[g_function]

        
        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features), #Lin_out x Lin_in Mat
            b1 = torch.randn(linear_1_out_features), # Need Column tensors for addition of bias
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict(
            x = 0.,
            z1 = 0.,
            z2= 0.,
            z3 = 0.,
            y_hat = 0.,
        )

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        self.cache['x'] = x
        z1 = x @ self.parameters['W1'].T + self.parameters['b1'] # Test 1 20x2 @ 2x10 -> 20x2 + 20x1
        self.cache['z1'] = z1 
        z2 = self.f_function(z1)
        self.cache['z2'] = z2
        z3 = z2 @ self.parameters['W2'].T + self.parameters['b2']
        self.cache['z3'] = z3
        y_hat = self.g_function(z3)
        self.cache['y_hat'] = y_hat
        # TODO: Implement the forward function
        return y_hat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        dgdz3 = self.back_g_function(self.cache['z3'])
        dJdz3 = dJdy_hat * dgdz3 
        self.grads['dJdW2'] = dJdz3.T @ self.cache['z2']
        self.grads['dJdb2'] = (dJdz3.T @ torch.ones((10,1))).reshape(-1,)
        dfdz1 = self.back_f_function(self.cache['z1'])
        self.grads['dJdW1'] = (dJdz3 @ self.parameters['W2'] * dfdz1).T @ self.cache['x']
        self.grads['dJdb1'] = (dJdz3 @ self.parameters['W2'] * dfdz1).T @ torch.ones((10,1)).reshape(-1,)
        # TODO: Implement the backward function
        pass

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    n_1, n_2 = y.size()[0], y.size()[1]
    loss = (torch.sum((y-y_hat)**2))/(n_1 *n_2) #Divided by features x samples/// NOT JUST SAMPLES
    dJdy_hat = (-2/(n_1*n_2))*(y-y_hat) #Pytorch gradient normalized over total features and samples

    # TODO: Implement the mse loss
    return loss,dJdy_hat


    # return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    n_1, n_2 = y.size()[0], y.size()[1]
    loss = torch.sum(y * torch.log(y_hat) + (1-y)*torch.log(1-y_hat))/(n_1 * n_2)
    dJdy_hat = (-1/(n_1 * n_2)) * (y/y_hat - (1-y)/(1-y_hat))
    return loss, dJdy_hat
    # return loss, dJdy_hat











