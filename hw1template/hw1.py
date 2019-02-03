import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Problem 2
def linear_gd(X,Y,lrate=0.1,num_iter=1000):
    w0 = np.zeros((X.shape[1], 1))
    w1 = np.zeros((X.shape[1], 1))
    Y = Y.reshape(-1, 1)
    counter = 0
    c = cost(w1,w0,X,Y)
    costs = [c]
    convergence_thres = 0.000001
    cprev = c + 10
#  theta0s = [w0]
    theta1s = [w1]
#     ( np.abs(cprev - c) > convergence_thres ) and
    while  (counter < num_iter):
          cprev = c
#          update0 = lrate * partial_cost_theta0(w0,w1,X,Y)
          update1  = lrate * partial_cost_thetal(w0,w1,X,Y)
          w1 -= update1
          theta1s.append(w1)
          c = cost(w0,w1,X,Y)
          costs.append(c)
          counter += 1   
return w1,costs,theta1s
#           w0 -= update0   
#           theta0s.append(w0)
 
X,Y = utils.load_reg_data()
w1,costs,theta1s =  linear_gd(X,Y,0.1,1000)
print(w1)        
plt.scatter(costs,theta1s)
plt.show()    

def cost(theta0,theta1, x, y):
    #Initialize cost
    J = 0
    # The number of observations
    m = len(x) 
    # Loop through each obervation
    for i in range(m):
        # Compute the hypothesis
        h = theta1 * x[i] + theta0
        # Add to cost
        J += (h-y[i]) ** 2
     # Average and normalize cost   
    J /= (2 * m)
    
    return J

print(cost(0,1,X,Y))

def partial_cost_thetal(theta0, theta1, x ,y):
     h = theta0 + theta1 * x
     diff = (h-y) * x
     partial = diff.sum()/(x.shape[0])
     return partial


def partial_cost_theta0(theta0, theta1, x ,y):
     h = theta0 + theta1 * x
     diff = (h-y)
     partial = diff.sum()/(x.shape[0])
     return partial

def linear_normal(X,Y):
    # return parameters as numpy array
    w = np.zeros((X.shape[1], 1))
    w = inv(X.T @ X) @ X.T @ Y
    return w

w = linear_normal(X,Y)
print(w)

from mpl_toolkits.mplot3d import Axes3D
def plot_linear():
    X,Y = utils.load_reg_data()
    w = linear_normal(X,Y)
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X=X,Y=Y, Z= (X + Y) * w / (X + Y) )
    plt.xlabel('x_data')
    plt.ylabel('y_data') 
    # return plot
    return plt

plt = plot_linear()
plt.show()   

# Problem 4
def poly_gd(X,Y,lrate=0.01,num_iter=3000):
    # return parameters as numpy array
    return []

def poly_normal(X,Y):
    # return parameters as numpy array
    return []

def plot_poly():
    # return plot
    return []

def poly_xor():
    # return labels for XOR from linear,polynomal models
    y_poly = []
    y_linear = []
    return y_linear,y_poly


# Problem 5
def nn(X,Y,X_test):
    # return labels for X_test as numpy array
    return []

def nn_iris():
    return 0.0

# Problem 6
def logistic(X,Y,lrate=1,num_iter=3000):
    # return parameters as numpy array
    return []

def logistic_vs_ols():
    # return plot
    return []


