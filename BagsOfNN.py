import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class BagOfNeuralNet(nn.Module):
    def __init__(self, input_size, rank_of_kernel):
        super().__init__()
        self.rank_of_kernel = rank_of_kernel
        self.dense_input = nn.Linear(input_size,max(rank_of_kernel,32))
        self.dense_intermediate_1 = nn.Linear(max(rank_of_kernel,32),max(rank_of_kernel,32))
        self.dense_intermediate_2 = nn.Linear(max(rank_of_kernel,32),max(rank_of_kernel,32))
        self.dense_intermediate_3 = nn.Linear(max(rank_of_kernel,32),max(rank_of_kernel,32))
        self.dense_intermediate_4 = nn.Linear(max(rank_of_kernel,32),max(rank_of_kernel,32))
        self.dense_intermediate_final = nn.Linear(max(rank_of_kernel,32), 1)  
        
    def Phi(self,x):
        x = torch.tanh(self.dense_input(x))
        x = F.dropout(x, p=0.2, training= self.training)
        x = torch.relu(self.dense_intermediate_1(x))   
        x = F.dropout(x, p=0.2, training= self.training)
        x = torch.relu(self.dense_intermediate_2(x)) 
        x = F.dropout(x, p=0.2, training= self.training)    
        x = torch.relu(self.dense_intermediate_3(x)) 
        x = F.dropout(x, p=0.2, training= self.training)
        x = torch.tanh(self.dense_intermediate_4(x)) 
        x = self.dense_intermediate_final(x)        
        return x

    def forward(self, x): 
        phi = self.Phi(x) 
        return phi
    
    def predict(self,xstar):  
        self.training = False
        phistar = self.Phi(xstar)
        return phistar
    
    def fit(self,x,y,num_of_iterations):
        optimizer = optim.Adam(self.parameters())
        nnloss = nn.MSELoss()
        self.training = True
        self.loss_history = []
        for ii in range(num_of_iterations): 
            optimizer.zero_grad()
            phi = self.forward(x)
            loss = nnloss(phi,y)
            self.loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
