import torch
import numpy as np
import torch.nn as nn
from torch.optim.adam import Adam


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Net, self).__init__()
        
        # Input Layer: list of coordinates from media pipe hands
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        
        # Hidden Layers: the computer manipulates these layers to teach the neural network 
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer: labeling if it is safe or not
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Activation Function:removing negative numbers        
        self.activation_function = nn.ReLU()
        
        # Output Function: keeps numbers between 1 and 0
        self.output_function = nn.Sigmoid()
        
    # Chain that the input follows through each layer and then returning the output
    def forward(self, input):
        output = self.linear1(input)
        output = self.activation_function(output)
        
        output = self.linear2(output)
        output = self.activation_function(output)
        
        output = self.linear3(output)
        output = self.output_function(output)
        return output
    

if __name__ == '__main__':
    hand_landmark_num = 21
    
    # Inputs: Hand coordinates
    x = torch.rand((100, hand_landmark_num * 3))
    
    # Labels: Safe(1) / Not Safe(0) 
    y = torch.randint(0, 2, (100, 1), dtype=torch.float32) 
    
    # Initializing the neural network
    net = Net(hand_landmark_num * 3, 32)
    
    # The optimizer updates the parameters(weights) between the hidden layers in 
    # increments of the learning rate for the neural network to learn
    optimizer = Adam(net.parameters(), lr=1e-3)
    
    # Epoch: iterating through all of the data
    for epoch in range(100):
        # y_pred: the values the net thought for each given x value
        y_pred = net(x)
        
        # loss: how wrong was the neural network
        # BCELoss: Measures the Binary Cross Entropy between target and input probability
        loss = nn.BCELoss()(y_pred, y)
        
        # Zeros the gradient to stop gradients from compounding
        optimizer.zero_grad()
        
        # loss.backward(): creates the loss gradients
        loss.backward()
        
        # optimizer makes changes to the weights based on the loss gradients
        optimizer.step()
        print(loss)
    
    