import torch.optim
import torch
x=torch.randn(3,requires_grad=True)
print(x)
y=x+2
print(y)

z=y*y*2
print(z)

w=z.mean()
print(w)

w.backward()
print(x.grad)


x.requires_grad_(False)
print(x)



weights= torch.ones(4,requires_grad=True)

#jacobian * original grad vector=new grad vector

for epoch in range(4):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()


print(weights)





# optimizer=torch.optim.SGD(weights,lr=0.01)
# #stochastic gradient descemt    
# optimizer.step()
# optimizer.zero_grad()
# #set the gradient for new step only





#^y=wx
#y^ is the estimated value


w=torch.tensor(1.0,requires_grad=True)
x=torch.tensor(1.0)
y=torch.tensor(2.0)


#compute loss by forward


y_hat=w*x
loss=(y_hat-y)*(y_hat-y)
 

#dLoss/dw calculated    .backward() finds product of (jacobian mat prod for middle values)    (x*w-y)^2     dy^/dw * d(y^-y)/dy^ * d(y^-y)^2/d(y^-y)    if x=1,y=2,w=1    x * 1 * 2(1-2) = -2      

print(loss)
loss.backward()
print(w.grad)





'''
PYTORCH CONVERTS MACHINE LEARNING ALGO TO EASY WAY AS MANUAL TO FUNCTIONAL COMPUTATION


- Prediction: Manual
- Gradients Computation: Manual
- Loss Computation: Manual
- Parameter updates: Manual


- Prediction: Manual
- Gradients Computation: Autograd
- Loss Computation: Manual
- Parameter updates: Manual


- Prediction:Manual
- Gradients Computation: Autograd
- Loss Computation: PyTorch Loss
- Parameter updates: PyTorch Optimizer


- Prediction: PyTorch Model
- Gradients Computation: Autograd
- Loss Computation: PyTorch Loss
- Parameter updates: PyTorch Optimizer
'''



import numpy as np
x=np.array([1,2,3,4],dtype=np.float32)
y=np.array([2,4,6,8],dtype=np.float32)
w=0


#prediciton 
def forward(x):
    return w*x


#loss  mean squaered errror
def loss(y, y_predicted):
    return ((y-y_predicted)**2).mean()


#gradient d(MSE)/dw   2x(summation(wx-y))
def gradient(x,y,y_pred):
    return np.dot(2*x,y_pred-y).mean()


print(f"Pred b4 train {forward(5)}")



learningrate=0.01
n_iters=10



for epoch in range(n_iters):
    y_pred=forward(x)
    l=loss(y,y_pred)
    dw=gradient(x,y,y_pred)
    w-=learningrate*dw 


    if True:
        print(f"Epoch={epoch+1}: w={w}, loss={l}")
print(f"Pred a5 train {forward(5)}")


#type 2

x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)
w=torch.tensor(0,dtype=torch.float32,requires_grad=True)    

for epoch in range(n_iters):
    y_pred=forward(x)
    l=loss(y,y_pred)
    l.backward()
    with torch.no_grad():
        w-=learningrate*dw 
    w.grad.zero_()

    if True:
        print(f"Epoch={epoch+1}: w={w}, loss={l}")
print(f"Pred a5 train {forward(5)}")


#too slow for correct prediction

# learningrate=10 increased loss
# learningrate=1 still slow

learningrate=6.7
x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([2,4,6,8],dtype=torch.float32)
w=torch.tensor(0,dtype=torch.float32,requires_grad=True)    

for epoch in range(n_iters):
    y_pred=forward(x)
    l=loss(y,y_pred)
    l.backward()
    with torch.no_grad():
        w-=learningrate*dw 
    w.grad.zero_()

    if True:
        print(f"Epoch={epoch+1}: w={w}, loss={l}")
print(f"Pred a5 train {forward(5)}")



#type 3
import torch.nn as nn
#  simple model for linear basic



learningrate=0.01
# change shape for x and y 
x=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y=torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
w=torch.tensor(0,dtype=torch.float32,requires_grad=True)
x_test=torch.tensor([5],dtype=torch.float32)
n_samples,n_features=x.shape
print(n_samples,n_features)
input_size=n_features
output_size=n_features
model=nn.Linear(input_size,output_size)
n_iters=100
print(f"Pred b4 train {model(x_test).item()}")

learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(n_iters):
    # predict = forward pass with our model
    y_predicted = model(x)

    # loss
    l = loss(y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)

print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')