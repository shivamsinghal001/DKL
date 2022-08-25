import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class LinearDKL(nn.Module):
    def __init__(self, rank_of_kernel, sigma_init, input_size):
        super().__init__()
        self.rank_of_kernel = rank_of_kernel
        self.dense_input = nn.Linear(input_size,max(rank_of_kernel,32))
        self.dense_intermediate_1 = nn.Linear(max(rank_of_kernel,32),max(rank_of_kernel,32))
        self.dense_intermediate_2 = nn.Linear(max(rank_of_kernel,32),max(rank_of_kernel,32))
        self.dense_intermediate_3 = nn.Linear(max(rank_of_kernel,32),max(rank_of_kernel,32))
        self.dense_intermediate_4 = nn.Linear(max(rank_of_kernel,32),max(rank_of_kernel,32))
        self.dense_intermediate_final = nn.Linear(max(rank_of_kernel,32),rank_of_kernel)
        self.cov_mat_finite_rank_layer = SigmaLayer(sigma_init)   
        
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
        phi,sigma = self.cov_mat_finite_rank_layer(phi)
        return phi,sigma
    
    def predict(self,x,y,xstar):       
        self.training = False
        phi = self.Phi(x)
        phistar = self.Phi(xstar)
        sigma = self.cov_mat_finite_rank_layer.sigma
        
        R = phi.size()[1]
        N = phi.size()[0]
        
        mean = (phi[:,0].view(-1,1).t() @ y)/(sigma**2+torch.norm(phi[:,0])**2)*phistar[:,0].view(-1,1)
        for ii in range(1,R):
            mean += (phi[:,ii].view(-1,1).t() @ y)/(sigma**2+torch.norm(phi[:,ii])**2)*phistar[:,ii].view(-1,1)
            
        var = sigma**2/(sigma**2+torch.norm(phi[:,0])**2)*phistar[:,0].view(-1,1) @ phistar[:,0].view(-1,1).t()
        for ii in range(1,R):
            var += sigma**2/(sigma**2+torch.norm(phi[:,ii])**2)*phistar[:,ii].view(-1,1) @ phistar[:,ii].view(-1,1).t() 
        return mean, var
    
    def fit(self, x, y, num_of_iterations, 
            lamb_reg, lamb_comp,
            lamb_l2_reg_loss):
        self.training = True
        optimizer = optim.Adam(self.parameters())
        nnloss = nn.MSELoss()
        
        self.loss_history = []
        for ii in range(num_of_iterations): 
            optimizer.zero_grad()
            phi,sigma = self.forward(x)
            loss, data_fit_loss, complexity_loss, regularity_loss= linear_dkl_loss(phi, sigma,
                                                                       y, 
                                                                       lamb_reg, 
                                                                       lamb_comp,
                                                                       lamb_l2_reg_loss)
            self.loss_history.append([loss.item(), data_fit_loss.item(),
                                      complexity_loss.item(),regularity_loss.item()])
            loss.backward()
            optimizer.step() 

# define the SigmaLayer class
class SigmaLayer(nn.Module):
    def __init__(self,sigma_init):
        super().__init__()  
        # initialize the parameters
        self.sigma = nn.Parameter(torch.Tensor(1,1))
        self.sigma.data.fill_(sigma_init)
        
    def forward(self,phi):
        return phi,self.sigma

    
# Define the loss
def linear_dkl_loss(phi, sigma, y, lamb_reg, lamb_comp, lamb_l2_reg_loss):
    R = phi.size()[1]
    N = phi.size()[0]  
    
    data_fit_loss = sigma**(-2)*torch.norm(y)**2
    
    complexity_loss = (N-R)*torch.log(sigma**2)  
    
    for ii in range(R):
        data_fit_loss -= (y.t() @ phi[:,ii].view(-1,1))**2 / (sigma**2 * (sigma**2 + torch.norm(phi[:,ii])**2) )
        complexity_loss += torch.log(sigma**2+torch.norm(phi[:,ii])**2)
        
    regularity_loss = torch.Tensor([[0]]) 
    l2_reg_loss = 0
    y_norm = torch.norm(y)**2
    for ii in range(R):
        v1 = phi[:,ii].view(-1,1)
        l2_reg_loss += torch.max(v1)
        for jj in range(ii+1,R):
            v2, var_val = phi[:,jj].view(-1,1), sigma**(-2) * y_norm
            regularity_loss += var_val *(v1.t() @ v2)**2 / (torch.norm(v1)**2 * torch.norm(v2)**2)
    l2_reg_loss /= torch.max(y)
    l2_reg_loss = torch.log(l2_reg_loss)
    total_loss = (data_fit_loss 
                  + lamb_comp * complexity_loss 
                  + lamb_reg * regularity_loss 
                  + lamb_l2_reg_loss * l2_reg_loss)
    return  total_loss, data_fit_loss, complexity_loss, regularity_loss
 
