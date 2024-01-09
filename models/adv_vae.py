
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from tqdm import trange
from sklearn.utils import shuffle
import numpy as np
criterionMSE = nn.MSELoss()


#batch_size=128
#epochs=10
#seed=1
#log_interval=10
#device = torch.device("cpu")

class ADV_VAE(nn.Module):
    def __init__(self, batch_size, epochs, seed, log_interval, device, nb_features, model, model_adv, sizelat,betaX,betaY,betammd_E,betaadv):
        super(ADV_VAE, self).__init__()
        self.batch_size=batch_size
        self.nbepoch=epochs
        self.seed=seed
        self.log_interval=log_interval
        self.device = device #torch.device("cpu")
        self.nb_features = nb_features
        self.model =model#()
        self.model_adv=model_adv()
        self.sizelat=sizelat
        self.betaX = betaX
        self.betaY = betaY
        #betaKLD = 10
        self.betammd_E =betammd_E
        #betammd_F =0
        self.betaadv=betaadv
    def compute_kernel(self,x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
        return ((-(tiled_x - tiled_y) ** 2).mean(dim=2) / float(dim)).exp_()

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd

    def predict(self, X, S, Y): 
        #x_var = Variable(torch.FloatTensor(X.values)).to(self.device)
        #s_var = Variable(torch.FloatTensor(S.values)).to(self.device)
        return self.model(X, S, Y)
    def forward(self, X, S, Y): 
        #x_var = Variable(torch.FloatTensor(X.values)).to(self.device)
        #s_var = Variable(torch.FloatTensor(S.values)).to(self.device)
        return self.model(X, S, Y)

    def fit(self, X_train, y_train, Z_train): 
        
        self.model_adv.to(self.device)
        self.optimizer_adv = torch.optim.Adam(self.model_adv.parameters(), lr=1e-3)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        criterionwL = torch.nn.BCEWithLogitsLoss(reduction='mean')
        
        
        batch_no = len(X_train) // self.batch_size
 
        data_senstrain=torch.tensor(np.expand_dims(Z_train,axis = 1)).float().to(self.device)
        data=torch.tensor(X_train.values).float().to(self.device)
        ydata= Variable(torch.FloatTensor(np.expand_dims(y_train,axis = 1))).to(self.device)
        
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

                train_loss = 0
                self.optimizer.zero_grad()
                recon_X_batch, z, recon_Y_batch, mu, logvar = self.model(x_var,s_var,y_var)
                true_samples = Variable(
                            torch.randn(x_var.shape[0], self.sizelat),
                            requires_grad=False
                        ).to(self.device)
                nb_a=2

                BCE_X = F.binary_cross_entropy(recon_X_batch, x_var.view(-1, x_var.shape[1]), reduction='mean')  
                
                
                
                BCE_Y = F.mse_loss(recon_Y_batch, y_var, reduction='mean')  
                mmd = self.compute_mmd(true_samples, z)
                
                nombreit=50
                if epoch ==0:
                    nombreit = 300

                for k in range(nombreit):
                    self.optimizer_adv.zero_grad()
                    apred_adv = self.model_adv(z)
                    loss_adv = F.mse_loss(apred_adv, s_var)
                    loss_adv.backward(retain_graph=True)
                    self.optimizer_adv.step()
                  
                apred_adv = self.model_adv(z)
                loss_adv = F.mse_loss(apred_adv,s_var)


                LossGlob = self.betaX*BCE_X + self.betaY*BCE_Y + self.betammd_E*mmd -  self.betaadv*loss_adv 
                LossGlob.backward()
                train_loss += LossGlob.item()
                self.optimizer.step()
            if epoch%5==0:
                #print("Loss_X :", self.betaX*BCE_X.cpu().detach().numpy(),"Loss_Y :", self.betaY*BCE_Y.cpu().detach().numpy(), "ADV :", self.betaadv*loss_adv.cpu().detach().numpy())

                recon_X, z, recon_Y, mu, logvar = self.model(data.view(-1, data.shape[1]),data_senstrain,ydata)
                #Loss_X = F.binary_cross_entropy(recon_X, data.view(-1, data.shape[1]), reduction='mean')  
                Loss_Y = F.mse_loss(recon_Y, ydata, reduction='mean')
                print("Loss_Y",Loss_Y.cpu().detach().numpy() )
                # lambda = 0
                #HGR_NNP = HGR_NN(Net_HGR(),Net2_HGR(),device, display=False)
                #print("HGR NN Train",HGR_NNP(senstrain , z,1000))
        return print("Done")  #self.model #self.model
