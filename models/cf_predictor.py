from tqdm import trange
from time import sleep
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from models.functions import *
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class CF_PREDICTOR(torch.nn.Module): 

    def __init__(self,regressor, batch_size, epochs, seed, log_interval, device, model, lr, modelVAE, nb, lambdap): 
        super().__init__()
        self.batch_size=batch_size
        self.nbepoch=epochs
        self.seed=seed
        self.log_interval=log_interval
        self.device = device
        self.model_h =model()
        self.lr =lr
        self.modelVAE=modelVAE
        self.nb = nb
        self.lambdap=lambdap
        if regressor == 'mse':
          self.criterion = torch.nn.MSELoss(reduction='mean')
        elif regressor == 'rmse':
          self.criterion = RMSELoss()

    def repeat(self, tensor, dims):
        if len(dims) != len(tensor.shape):
            raise ValueError("The length of the second argument must equal the number of dimensions of the first.")
        for index, dim in enumerate(dims):
            repetition_vector = [1]*(len(dims)+1)
            repetition_vector[index+1] = dim
            new_tensor_shape = list(tensor.shape)
            new_tensor_shape[index] *= dim
            tensor = tensor.unsqueeze(index+1).repeat(repetition_vector).reshape(new_tensor_shape)
        return tensor  
    def predict(self, X,S): 
        yhat= self.model_h(X,S)
        return yhat
    
    def fit(self, X_train, y_train, Z_train): 
        batch_no = len(X_train) // self.batch_size

        self.optimizer_h = torch.optim.Adam(self.model_h.parameters(), lr=self.lr)
        self.model_h.to(self.device) 
        
        loss=0
        ypred_var=0
        t = trange(self.nbepoch + 1, desc='Bar desc', leave=True)
        for epoch in t: #tqdm(range(1, self.nbepoch + 1), 'Epoch: ', leave=False):
            x_train,  ytrain, senstrain = shuffle(X_train.values, np.expand_dims(y_train,axis = 1), np.expand_dims(Z_train,axis = 1) )
              # Mini batch learning
            for i in range(batch_no):
                start = i * self.batch_size
                end = start + self.batch_size
                x_var = Variable(torch.FloatTensor(x_train[start:end])).to(self.device)
                y_var = Variable(torch.FloatTensor(ytrain[start:end])).to(self.device)
                s_var = Variable(torch.FloatTensor(senstrain[start:end])).to(self.device)
            
                          
                recon_X_batch_a, z_a, recon_Y_batch_a, mu_a, logvar_a = self.modelVAE.predict(x_var.view(-1, X_train.shape[1]),s_var,y_var)
                ypred_var_a = self.model_h(recon_X_batch_a,s_var)
                ypred_var_a = ypred_var_a.detach()
                
                x = torch.FloatTensor(self.nb, 1).uniform_(Z_train.min(), Z_train.max()).to(self.device)
                
                self.optimizer_h.zero_grad()
                ypred_var = self.model_h(x_var,s_var)
                lossY =F.mse_loss(ypred_var, y_var)             
                Unif_X = x.repeat(x_var.shape[0],1) 
                data_X = self.repeat(x_var, [self.nb,1])#torch.tensor(np.repeat(x_var, nb,0), dtype=torch.float32)

                Y_X    = self.repeat(y_var, [self.nb,1])#torch.tensor(np.repeat(y_var, nb,0), dtype=torch.float32)

                recon_X_aprime, z_aprime, recon_Y_aprime, mu_aprime, logvar_aprime = self.modelVAE.predict(data_X,Unif_X,Y_X)
                predY_a_prime = self.model_h(recon_X_aprime,Unif_X)
                Z_train_X = self.repeat(s_var, [self.nb,1])#torch.tensor(np.repeat(s_var, nb,0), dtype=torch.float32)
                recon_X_a, z_a, recon_Y_a, mu_a, logvar_a = self.modelVAE.predict(data_X, Z_train_X , Y_X)
                predY_a  = self.model_h(recon_X_a,Z_train_X)
                MSEcount = torch.mean((predY_a_prime-predY_a)**2)

                loss = lossY + self.lambdap*MSEcount #+ 0.001*KLD
                loss.backward()
                self.optimizer_h.step()
                
            if epoch %100==0:
                print("MSE estimation :", lossY.cpu().detach().numpy(),"conterfactual estimation :", MSEcount.cpu().detach().numpy())
