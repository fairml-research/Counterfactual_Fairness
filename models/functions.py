from tqdm import trange
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

criterionMSE = nn.MSELoss()

H = 15
H2 = 15
sizelat = 5


def Normalize(X_train, y_train, sensitive, X_test, y_test, sensitivet):
    meanYtrain=np.mean(y_train)
    stdYtrain=np.std(y_train)
    y_train=(y_train-meanYtrain)/stdYtrain
    y_test =(y_test-meanYtrain)/stdYtrain

    Z_train=pd.Series(sensitive)
    Z_test=pd.Series(sensitivet)
    X_train=pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test)
    y_train=pd.Series(y_train)
    y_test=pd.Series(y_test)
    scaler = MinMaxScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler)   
    X_test[X_test>1]=1
    X_test[X_test<0]=0
    return X_train, y_train, Z_train, X_test, y_test, Z_test, scaler, scale_df, meanYtrain, stdYtrain

class Net_HGR(nn.Module):
    def __init__(self):
        super(Net_HGR, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4    


class Net2_HGR(nn.Module):
    def __init__(self):
        super(Net2_HGR, self).__init__()
        self.fc1 = nn.Linear(sizelat, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4    


model_Net_F = Net_HGR()
model_Net_G = Net2_HGR()



class HGR_NN(nn.Module):
    
    def __init__(self,model_F,model_G,device,display):
        super(HGR_NN, self).__init__()
        self.mF = model_Net_F
        self.mG = model_Net_G
        self.device = device
        self.optimizer_F = torch.optim.Adam(self.mF.parameters(), lr=0.0005)
        self.optimizer_G = torch.optim.Adam(self.mG.parameters(), lr=0.0005)
        self.display=display
    def forward(self, yhatvar, svar, nb):

        #svar = Variable(torch.FloatTensor(np.expand_dims(s_var,axis = 1))).to(self.device)
        #yhatvar = Variable(torch.FloatTensor(np.expand_dims(yhat,axis = 1))).to(self.device)
        yhatvar=yhatvar.detach()
        svar=svar.detach()
        
        self.mF.to(self.device)
        self.mG.to(self.device)   
        
        for j in range(nb) :

            pred_F  = self.mF(yhatvar)
            pred_G  = self.mG(svar)
            
            epsilon=0.000000001
            
            pred_F_norm = (pred_F-torch.mean(pred_F))/torch.sqrt((torch.std(pred_F).pow(2)+epsilon))
            pred_G_norm = (pred_G-torch.mean(pred_G))/torch.sqrt((torch.std(pred_G).pow(2)+epsilon))

            ret = torch.mean(pred_F_norm*pred_G_norm)
            loss = - ret  # maximize
            self.optimizer_F.zero_grad()
            self.optimizer_G.zero_grad()
            loss.backward()
            
            if (j%100==0) and (self.display==True):
                print(j, ' ', loss)
            
            self.optimizer_F.step()
            self.optimizer_G.step()
            
        return ret.cpu().detach().numpy()
    


class VAE(nn.Module):
    def __init__(self, nb_features):
        super(VAE, self).__init__()
        self.nb_features=nb_features
        self.fc1 = nn.Linear(self.nb_features+2, 128)  
        self.fc11 = nn.Linear(128, 64)  
        self.fc12 = nn.Linear(64, 32)
        
        self.fc21 = nn.Linear(32, sizelat) 
        self.fc22 = nn.Linear(32, sizelat) 
        
        self.fc3 = nn.Linear(sizelat + 1, 60)
        
        self.fc41 = nn.Linear(60, self.nb_features)
        self.fc42 = nn.Linear(60, 1) 

    def encode(self, x, a, y):
        h1 = F.relu(self.fc12(F.relu(self.fc11(F.relu(self.fc1(torch.cat([x.view(-1, self.nb_features),a.view(-1, 1),y.view(-1, 1)],1)))))))        
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z)) 
        return torch.sigmoid(self.fc41(h3)), self.fc42(h3)

    def forward(self, x, a, y):
        mu, logvar = self.encode(x.view(-1, self.nb_features), a, y)  
        z = self.reparameterize(mu, logvar) 
        recon_X_batch, recon_Y_batch = self.decode(torch.cat([z,a.view(-1, 1)],1))
        return  recon_X_batch, z, recon_Y_batch, mu, logvar 
