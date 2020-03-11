from torch import nn

class FeedForwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNeuralNetModel, self).__init__()
        
        # First(hidden) linear layer
        self.linearA = nn.Linear(input_dim, hidden_dim)

        # Logistic activation function
        self.sigmoid = nn.Sigmoid()

        # Second linear layer
        self.linearB = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.linearA(x)
        
        out = self.sigmoid(out)
        
        out = self.linearB(out)

        return out
        
        