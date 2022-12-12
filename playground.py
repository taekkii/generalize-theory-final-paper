

import torch
import torch.nn as nn
import torch.optim
import random
import numpy as np
import argparse
from visdom import Visdom

# #of dimension
d = 10

w_star = torch.rand(d)
w_star = w_star / w_star.norm()
gamma_0 = d**-0.5

vis = Visdom()

z = torch.rand(d)
z = z / z.norm()

class TwoLayerNN(nn.Module):
    
    def __init__(self, m, d):
        '''
        params: m, d are according to the paper
        '''
        super().__init__()
        self.d = d
        self.W = nn.Linear(2*d, m)
        self.w = nn.Linear(m, 1)
    
    def forward(self,x):
        return self.w( torch.relu(self.W(x) ) )

def loss(y,y_hat):
    return (-torch.log(torch.sigmoid(y*y_hat))).mean()

@torch.no_grad()
def test(x,y,model:nn.Module,types=None):
    y_hat = model(x).view(-1)
    
    test_acc = (torch.sign(y_hat) == y ).long().sum() / y.shape[0]
    test_loss = loss(y,y_hat)
    
    if types is None:
        return test_acc, test_loss, {}, {}

    test_acc_type, test_loss_type = {}, {}
    for type in range(3):
        idx = types==type
        y_hat_type = y_hat[idx]
        y_type = y[idx]
        
        acc_type = (torch.sign(y_hat_type) == y_type ).long().sum() / y_type.shape[0]
        loss_type = loss(y_type, y_hat_type)
    
        test_acc_type[type] = acc_type
        test_loss_type[type] = loss_type
        
    return test_acc, test_loss, test_acc_type, test_loss_type


def train(x, y, x_test, y_test, types_test, model:nn.Module, eta, threshold, wd=0.001, prev_graph={}):
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=eta, weight_decay=wd/2)
    
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()
    
    n_iter = 0
    continue_iteration = True
    train_losses = prev_graph.get('train_losses',[])
    train_accs = prev_graph.get('train_accs',[])
    test_losses = prev_graph.get('test_losses',[])
    test_accs = prev_graph.get('test_accs',[])
    
    test_losses_type = prev_graph.get('test_losses_type',{0:[],1:[],2:[]})
    test_accs_type = prev_graph.get('test_accs_type',{0:[],1:[],2:[]})
    
    while continue_iteration:
        n_iter += 1
        y_hat = model(x).view(-1)
        
        with torch.no_grad():
            train_acc = (torch.sign(y_hat) == y ).long().sum() / y.shape[0]
        train_loss = loss(y,y_hat)
        
        continue_iteration = train_loss.item() > threshold
        
        
        # logging
        if n_iter % 100 == 0:
            train_losses.append(train_loss.item())
            train_accs.append(train_acc.item())
            
            test_acc, test_loss, test_acc_type, test_loss_type = test(x_test, y_test, model, types_test)
            test_losses.append(test_loss.item())
            test_accs.append(test_acc.item())
            
            print('\n')
            print(f"iter = {n_iter:5d}, train_loss = {train_losses[-1]:7.4f}, train_acc = {train_accs[-1]*100:5.2f}%")
            print(f"              test_loss = {test_losses[-1]:7.4f}, test_acc = {test_accs[-1]*100:5.2f}%")
            
            
            for i in range(3):
                test_losses_type[i].append(test_loss_type[i].item())
                test_accs_type[i].append(test_acc_type[i].item())
                
                print(f"type {i}: test_loss={test_losses_type[i][-1]:7.4f}, test_acc={test_accs_type[i][-1]*100:5.2f}%")
                
            vis.line(Y=torch.tensor([train_losses, test_losses]).t(),
                     X=torch.arange(len(train_losses)) * 100,
                     win='loss',
                     opts=dict(title='Losses',
                               showlegend=True,
                               legend=['train','test']))
            
            vis.line(Y=torch.tensor([train_accs, test_accs]).t(),
                    X=torch.arange(len(train_losses)) * 100,
                    win='accuracy',
                    opts=dict(title='Accuracies',
                            showlegend=True,
                            legend=['train','test']))

            vis.line(Y=torch.tensor([test_losses_type[0], test_losses_type[1], test_losses_type[2]]).t(),
                     X=torch.arange(len(train_losses)) * 100,
                     win='loss_type',
                     opts=dict(title='Losses_Type',
                               showlegend=True,
                               legend=['P','Q','PQ']))
            
            vis.line(Y=torch.tensor([test_accs_type[0], test_accs_type[1], test_accs_type[2]]).t(),
                    X=torch.arange(len(train_losses)) * 100,
                    win='accuracy_type',
                    opts=dict(title='Accuracies_type',
                            showlegend=True,
                            legend=['P','Q','PQ']))

            

        train_loss.backward()
        optimizer.step()
        
        optimizer.zero_grad()
        
    print("[TRAIN COMPLETE] # iter = {}".format(n_iter))
    return train_losses,train_accs,test_losses,test_accs, test_losses_type, test_accs_type
    
    
    
def gen_p(y):
    '''
    generate 2d example of P
    params
        y: label vector, uniformly distributed in {-1,1}
    '''
    
    x = []
    for yi in y:
        z = torch.randn(d)/d
        if yi == 1:
            while w_star @ z >= 0:
                z = torch.randn(d)/d
            x1 = gamma_0*w_star + z
        else:
            while w_star @ z <= 0:
                z = torch.randn(d)/d
            x1 = gamma_0*w_star + z
        x.append(x1)
    return torch.stack(x)
    
    
def gen_q(y,r):
    '''
    generate 2d example of Q
    params
        y: label vector, uniformly distributed in {-1,1}
    '''
    # zeta = torch.tensor([z[-1],-z[0]]) * r
    zeta = torch.zeros_like(z)
    zeta[0] = z[1]
    zeta[1] = -z[0]
    zeta = zeta/zeta.norm()
    zeta = zeta*r
    
    print(zeta.norm())
    x = []
    
    for yi in y:
        alpha = random.random()
        if yi == 1:
            x2 = alpha*z
        else:
            b = torch.randint(0,2,(1,))*2-1
            x2 = alpha*(z + b*zeta)
        x.append(x2)
    
    return torch.stack(x)    

def myplot(x,y):
    if torch.is_tensor(x):
        x = x.numpy()
    if torch.is_tensor(y):
        y = y.numpy()  
    
    from matplotlib import pyplot as plt
    
    x_label1 = x[y==1,:]
    x_label2 = x[y==-1,:]
    plt.scatter(x_label1[:, 0], x_label1[:, 1], s=1, c='red')
    plt.scatter(x_label2[:, 0], x_label2[:, 1], s=1, c='blue')
    
    plt.show()
    

def gen_data(n, p0=0.2 , q0=0.2):
    
    y = torch.randint(0,2,(n,))*2 - 1


    weights = torch.tensor([p0, q0, 1 - p0 - q0])
    types = torch.multinomial(weights, n, replacement=True)


    x1 = gen_p(y[types != 1])
    x2 = gen_q(y[types != 0],r=d**(-0.25))

    x = torch.zeros(n, 2*d)
    x[types != 1, :d] = x1
    x[types != 0, d:] = x2

    return x, y, types

parser = argparse.ArgumentParser()
parser.add_argument('--n_train',type=int, default=10000)
parser.add_argument('--n_test',type=int, default=10000)
parser.add_argument('--algorithm',type=str,default='s',choices=['ls','s'])
parser.add_argument('--modelsize',type=int,default=10000)

parser.add_argument("--eta_large",type=float)
parser.add_argument('--threshold_large',type=float)
parser.add_argument('--eta_small',type=float)
parser.add_argument('--threshold_small',type=float)
args = parser.parse_args()

train_x, train_y, train_types = gen_data(args.n_train)
test_x, test_y, test_types = gen_data(args.n_test)
test_x = test_x.cuda()
test_y = test_y.cuda()
train_types = train_types.cuda()
test_types  = test_types.cuda()

model = TwoLayerNN(m=args.modelsize, d=d)

print(f"wd = {d**(-1.25):.4f}")
if args.algorithm == 'ls':
    train_losses, train_accs, test_losses, test_accs, test_losses_type, test_accs_type = train(train_x, 
                                                                                               train_y, 
                                                                                               test_x, 
                                                                                               test_y, 
                                                                                               test_types, 
                                                                                               model, 
                                                                                               eta=args.eta_large, 
                                                                                               threshold=args.threshold_large)
    train_losses, train_accs, test_losses, test_accs, test_losses_type, test_accs_type = train(train_x, train_y, test_x, test_y, test_types, model, 
                                                             eta=args.eta_small, threshold=args.threshold_small,
                                                             prev_graph=dict(train_losses=train_losses,
                                                                             train_accs=train_accs,
                                                                             test_losses=test_losses,
                                                                             test_accs=test_accs,
                                                                             test_losses_type=test_losses_type,
                                                                             test_accs_type=test_accs_type))

elif args.algorithm == 's':
    train_losses, train_accs, test_losses, test_accs, test_losses_type, test_accs_type = train(train_x, train_y, test_x, test_y, test_types, 
                                                                                               model, eta=args.eta_small, threshold=args.threshold_small)

else:
    raise NotImplementedError("wrong algorithm")


