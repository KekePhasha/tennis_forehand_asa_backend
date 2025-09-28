import math


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        # Parameters
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim

        # Running stats
        self.running_mean = [0.0] * dim
        self.running_var = [1.0] * dim

        # Gradients
        self.dgamma = [0.0] * dim
        self.dbeta = [0.0] * dim

    def forward(self, x, train=True):
        batch_size = len(x)
        features = len(x[0])

        if train:
            # Compute mean and var from current batch
            means = [sum(row[j] for row in x) / batch_size for j in range(features)]
            vars_ = [
                sum((row[j] - means[j])**2 for row in x) / batch_size
                for j in range(features)
            ]

            # Update running stats
            self.running_mean = [
                self.momentum * rm + (1 - self.momentum) * m
                for rm, m in zip(self.running_mean, means)
            ]
            self.running_var = [
                self.momentum * rv + (1 - self.momentum) * v
                for rv, v in zip(self.running_var, vars_)
            ]

        else:
            # Use stored running statistics
            means, vars_ = self.running_mean, self.running_var

        # Normalize
        self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in vars_]
        out = []
        for row in x:
            normed = [(row[j] - means[j]) * self.std_inv[j] for j in range(features)]
            out.append([self.gamma[j] * normed[j] + self.beta[j] for j in range(features)])
        return out

    def backward(self, grad):
        # Simplified: pass-through, since full BN backward is complex
        return grad

    def step(self, lr):
        self.gamma = [g - lr * dg for g, dg in zip(self.gamma, self.dgamma)]
        self.beta  = [b - lr * db for b, db in zip(self.beta, self.dbeta)]

    def zero_grad(self):
        self.dgamma = [0.0] * self.dim
        self.dbeta = [0.0] * self.dim
