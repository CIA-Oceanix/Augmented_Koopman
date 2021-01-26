import numpy as np
import torch 
from utils import torch_expm

class Aug_Koopman_ModelD(torch.nn.Module):
        def __init__(self, params):
            super(Aug_Koopman_Model, self).__init__()
            y_aug = np.random.uniform(size=(params['nb_Batch'],params['Batch_size'],params['dim_latent']))*0.0
            self.y_aug = torch.nn.Parameter(data=torch.from_numpy(y_aug).double(), requires_grad=True)
#            u = np.random.uniform(size=(nb_batch,batch_size,1))
#            self.u = torch.nn.Parameter(data=torch.from_numpy(u).float(), requires_grad=True)
            A = np.random.uniform(size=(params['dim_output'],params['dim_output']))
            self.A = torch.nn.Parameter(data=torch.from_numpy(A).double(), requires_grad=True)
        def forward(self, inp, dt, t0):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            aug_inp = inp
            rep_phi = self.A.repeat(aug_inp.shape[0],1,1)
            pred = torch.bmm(rep_phi,aug_inp.unsqueeze(-1))[:,:,0]
            #inp_for = torch.bmm(  ,aug_inp)
            grad, Phi, eig_vals, eig_vects = [], [], [], []
            return pred, grad, Phi, eig_vals, eig_vects

class Aug_Koopman_Model(torch.nn.Module):
        def __init__(self, params):
            super(Aug_Koopman_Model, self).__init__()
            y_aug = np.random.uniform(size=(params['nb_Batch'],params['Batch_size'],params['dim_latent']))*0.0
            self.y_aug = torch.nn.Parameter(data=torch.from_numpy(y_aug).double(), requires_grad=True)
#            u = np.random.uniform(size=(nb_batch,batch_size,1))
#            self.u = torch.nn.Parameter(data=torch.from_numpy(u).float(), requires_grad=True)
            self.imag_eigen = params['Imag_Eigen']
            A = np.random.uniform(size=(params['dim_output'],params['dim_output']))
            self.A = torch.nn.Parameter(data=torch.from_numpy(A).double(), requires_grad=True)
        def forward(self, inp, dt, t0):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            aug_inp = inp
            if self.imag_eigen:
                A = (self.A-self.A.T)/2
            else:
                A = self.A
            L_outp  = torch.nn.functional.linear(aug_inp,A)
            grad = L_outp #+ self.m*(u)
            eig_vals, eig_vects = (torch.eig(A,eigenvectors=True))
            Phi = torch_expm(A.unsqueeze(0)*dt)
            rep_phi = Phi.repeat(aug_inp.shape[0],1,1)
            pred = torch.bmm(rep_phi,aug_inp.unsqueeze(-1))[:,:,0]
            #inp_for = torch.bmm(  ,aug_inp)
            return pred, grad, Phi, eig_vals, eig_vects

class Aug_Koopman_Model_EQ(torch.nn.Module):
        def __init__(self, params):
            super(Aug_Koopman_Model_EQ, self).__init__()
            y_aug = np.random.uniform(size=(params['nb_Batch'],params['Batch_size'],params['dim_latent']))*0.0
            self.y_aug = torch.nn.Parameter(data=torch.from_numpy(y_aug).double(), requires_grad=True)
#            u = np.random.uniform(size=(nb_batch,batch_size,1))
#            self.u = torch.nn.Parameter(data=torch.from_numpy(u).float(), requires_grad=True)
            self.imag_eigen = params['Imag_Eigen']
            A = np.random.uniform(size=(params['dim_output'],params['dim_output']))
            self.A = torch.nn.Parameter(data=torch.from_numpy(A).double(), requires_grad=True)
        def forward(self, inp, dt, t0):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            aug_inp = inp
            if self.imag_eigen:
                A = (self.A-self.A.T)/2
            else:
                A = self.A
            L_outp  = torch.nn.functional.linear(aug_inp,A)
            grad = L_outp #+ self.m*(u)
            eig_vals, eig_vects = (torch.eig(A,eigenvectors=True))
            Phi = torch_expm(A.unsqueeze(0)*dt)
            rep_phi = Phi.repeat(aug_inp.shape[0],1,1)
            pred = torch.bmm(rep_phi,aug_inp.unsqueeze(-1))[:,:,0]
            #inp_for = torch.bmm(  ,aug_inp)
            return pred, grad, Phi, eig_vals, eig_vects
class Multi_INT_net(torch.nn.Module):
        def __init__(self, params, model):
            super(Multi_INT_net, self).__init__()
            self.Int_net = model
        def forward(self, inp, t0, nb, dt):

            pred = [inp]
            for i in range(nb):
                predic, grad, Phi, eig_vals, eig_vects = self.Int_net(pred[-1], dt, 0.0)
                pred.append(predic)
          
            return torch.stack(pred)


class Aug_Koopman_ModelQP(torch.nn.Module):
        def __init__(self,params, periodic_kernel):
            super(Aug_Koopman_ModelQP, self).__init__()
            self.Periodic_kernel = periodic_kernel
            self.linearCell   = torch.nn.Linear(params['dim_output']*2, params['dim_hidden_linear']).double()
            self.BlinearCell1 = torch.nn.ModuleList([torch.nn.Linear(params['dim_output']*2, 1).double() for i in range(params['bi_linear_layers'])])
            self.BlinearCell2 = torch.nn.ModuleList([torch.nn.Linear(params['dim_output']*2, 1).double() for i in range(params['bi_linear_layers'])])
            #self.QlinearCell1 = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent']+params['dim_input'], 1) for i in range(params['bi_linear_layers'])])
            #self.QlinearCell2 = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent']+params['dim_input'], 1) for i in range(params['bi_linear_layers'])])
            #self.QlinearCell3 = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent']+params['dim_input'], 1) for i in range(params['bi_linear_layers'])])
            #self.QlinearCell4 = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent']+params['dim_input'], 1) for i in range(params['bi_linear_layers'])])
            augmented_size    = 1# + params['bi_linear_layers'] + params['dim_hidden_linear']+params['dim_output']*2
            self.transLayers = torch.nn.ModuleList([torch.nn.Linear(augmented_size, augmented_size).double()])
            self.transLayers.extend([torch.nn.Linear(augmented_size, augmented_size).double() for i in range(1, params['transition_layers'])])
            self.outputLayer  = torch.nn.Linear(augmented_size, params['dim_output']).double()
            self.nb_tl = params['transition_layers']
            self.nb_bl = params['bi_linear_layers']
        def forward(self, inp, tf, t0):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
#            print('final : ', tf)
#            print('init : ', t0)
            predic, grad, Phi, eig_vals, eig_vects = self.Periodic_kernel(inp, (tf-t0)[0,0], t0)
            inp_qp = torch.cat((inp,predic),dim = -1)
            BP_outp = (torch.zeros((inp_qp.size()[0],self.nb_bl)).double())
            L_outp   = self.linearCell(inp_qp)
            for i in range((self.nb_bl)):
                BP_outp[:,i]=self.BlinearCell1[i](inp_qp)[:,0]*self.BlinearCell2[i](inp_qp)[:,0]
        #            for i in range((params['bi_linear_layers'])):
        #                QP_outp[:,i]=self.QlinearCell1[i](aug_inp)[:,0]*self.QlinearCell2[i](aug_inp)[:,0]*self.QlinearCell3[i](aug_inp)[:,0]*self.QlinearCell4[i](aug_inp)[:,0]      
            aug_vect = tf#torch.cat((L_outp, BP_outp, inp_qp, tf), dim=1)
            for i in range((self.nb_tl)):
                aug_vect = torch.relu(self.transLayers[i](aug_vect))
            out_qp = predic + self.outputLayer(aug_vect)
            return out_qp, grad, Phi, eig_vals, eig_vects

class Multi_INT_net(torch.nn.Module):
        def __init__(self, params, model):
            super(Multi_INT_net, self).__init__()
            self.Int_net = model
        def forward(self, inp, t0, nb, dt):

            pred = [inp]
            for i in range(nb):
                predic, grad, Phi, eig_vals, eig_vects = self.Int_net(pred[-1], dt, 0.0)
                pred.append(predic)
          
            return torch.stack(pred)

def get_initial_condition(model, time_series, train_series, dt, err_tol = 1E-4, n_train = 10000, get_init_in_train_set = True):
    criterion = torch.nn.MSELoss()#reduction = 'sum')
    if get_init_in_train_set :
        train_series_unbatched = train_series.reshape(train_series.shape[0]*train_series.shape[1],-1)
        y_aug_series_unbatched = model.Int_net.y_aug.reshape(train_series.shape[0]*train_series.shape[1],-1)
        aug_inp = torch.cat((train_series_unbatched,y_aug_series_unbatched),dim = -1)
        for p in model.parameters():
            p.requires_grad = False
        
        pred = model(aug_inp, 0.0, time_series.shape[0], dt)
        loss_init=[]
        for i in range(aug_inp.shape[0]):
            loss_init.append(criterion(pred[:,i,:train_series.shape[-1]][np.where(np.isnan(time_series)==0)], time_series[np.where(np.isnan(time_series)==0)]))
        min_idx = np.where(torch.stack(loss_init).data.numpy()==torch.stack(loss_init).data.numpy().min())
        inp_init = aug_inp.detach()[min_idx[0][0]].reshape(1,aug_inp.shape[-1])
    else:
        min_idx = None
        inp_init = torch.rand(1,(train_series.shape[-1]+model.Int_net.y_aug.shape[-1])).double()*0.0
        inp_init[:,:time_series.shape[-1]] = time_series.clone()[:1,:]
    init_cond_model = get_init(model,inp_init)
    optimizer = torch.optim.Adam(init_cond_model.parameters(), lr = 0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor = 0.1, patience=205, verbose=True, min_lr = 0.1)
    stop_cond = False
    count = 0
    while(stop_cond==False):
        # Forward pass: Compute predicted y by passing x to the model
        pred = init_cond_model(0.0,time_series.shape[0],dt)
        #pred1, grad, inp, aug_inp = modelRINN(test_vars[:1,:],dt, True, iterate = t)
        
        # Compute and print loss
        loss = criterion(pred[1:,0,:time_series.shape[-1]], time_series[:,:])
#        criterion(pred[1:,0,:time_series.shape[-1]], time_series[:,:])
        print(count,loss)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)
        count += 1
        if loss.detach().numpy()<err_tol or count>n_train:
            stop_cond = True
    return inp_init, min_idx, init_cond_model.estimate_init

class get_init(torch.nn.Module):
        def __init__(self, model_Multi_RINN, inp_init):
            super(get_init, self).__init__()
#            self.add_module('Dyn_net',FC_net(params))
            self.Multi_INT_net = model_Multi_RINN
            self.estimate_init = torch.nn.Parameter((inp_init.clone()))#torch.nn.Parameter(aug_inp[:1,:])
        def forward(self, t0, nb, dt):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            pred = self.Multi_INT_net(self.estimate_init, t0, nb, dt)
            return pred