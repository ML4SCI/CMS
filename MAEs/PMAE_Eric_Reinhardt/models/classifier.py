from torch import nn
from models.shared_layers import CustomActivationFunction

# Simple binary classifier
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, dim, dropout):
        super(BinaryClassifier, self).__init__()

        self.linear1 = nn.Linear(input_size, dim)

        self.linear2 = nn.Linear(dim, dim)

        self.linear3 = nn.Linear(dim, dim)

        self.linear4 = nn.Linear(dim, dim)

        self.linear5 = nn.Linear(dim, dim)

        self.linear6 = nn.Linear(dim, 1)

        self.custom_act = CustomActivationFunction()

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.custom_act(self.linear1(x)))
        x = self.dropout(self.custom_act(self.linear2(x)))
        x = self.dropout(self.custom_act(self.linear3(x)))
        x = self.dropout(self.custom_act(self.linear4(x)))
        x = self.dropout(self.custom_act(self.linear5(x)))
        x = self.sigmoid(self.linear6(x))
        return x