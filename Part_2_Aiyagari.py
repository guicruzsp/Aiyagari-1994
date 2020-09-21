# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:41:52 2020

@author: gui_c
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AutoRegResults
from statsmodels.tsa.api import acf, pacf, graphics
import quantecon as qe
import math
from copy import deepcopy

# Generating the times series for labor according to equation (9)
sigma = np.array((0.2,0.4))
rho = np.array((0, 0.3, 0.6, 0.9))
epsilon = np.random.normal(0, 1, 500)
l = np.arange(0,500,1).astype(float)
l[0] = 1

j = 0
k = 0

series_list = list()
series_dict = {}

for k in range(2):
    for j in range(4):
        for i in range(1,500):
            l[i] = math.exp(rho[j]*math.log(l[i-1]) + sigma[k]*(1-rho[j]**2)**(1/2)*epsilon[i])
        midstep = deepcopy(l)
        series_list.append(midstep)
        series_dict[str(k)+"-"+str(j)] = midstep
        title = "k = "+str(k)+", j = "+str(j)
        plt.plot(midstep)
        plt.title(title)
        plt.show()


# Now that we have the series, we apply the Tauchen algorithm
# Approx with 7 points
# Here we can obtain the values from table 1

m = 3
n = 7

results_table = np.repeat(0,4*2).reshape(2,4).astype(float)

j,k = 0,0
for p in range(4):
    tauchen_approx = qe.markov.approximation.tauchen(rho[j], math.sqrt(sigma[k]**2*(1-rho[j]**2)), m =m,n=n)
    tauchen_sim = tauchen_approx.simulate(500)
    plt.plot(tauchen_sim, label = "Tauchen approx")
    plt.plot(np.log(series_list[p]), label = "AR(1)")
    plt.xlabel('Periods')
    plt.ylabel('l')
    plt.title("Approximation with 7 points")
    plt.legend()
    plt.show()
    tauchen_est = AutoReg(tauchen_sim,1).fit()
    est_sigma = str(math.sqrt((sigma[k]**2*(1-rho[j]**2))/(1-tauchen_est.params[1]**2)))
    print(str(j)+"/"+str(k)+"/"+str(tauchen_est.params[1])+"/"+est_sigma)
    j = j+1

j,k = 0,1
for p in range(4,8):
    tauchen_approx = qe.markov.approximation.tauchen(rho[j], math.sqrt(sigma[k]**2*(1-rho[j]**2)), m =m,n=n)
    tauchen_sim = tauchen_approx.simulate(500)
    plt.plot(tauchen_sim, label = "Tauchen approx")
    plt.plot(np.log(series_list[p]), label = "AR(1)")
    plt.xlabel('Periods')
    plt.ylabel('l')
    plt.title("Approximation with 7 points")
    plt.legend()
    plt.show()
    tauchen_est = AutoReg(tauchen_sim,1).fit()
    est_sigma = str(math.sqrt((sigma[k]**2*(1-rho[j]**2))/(1-tauchen_est.params[1]**2)))
    print(str(j)+"/"+str(k)+"/"+str(tauchen_est.params[1])+"/"+est_sigma)
    j = j+1



# Parameters for table 2
sigma = np.array((0.2,0.4))
rho = np.array((0, 0.3, 0.6, 0.9))
gamma = 3 
k = 1 # Here we choose sigma
j = 2 # Here we choose rho
m = 3
n = 7
tauchen_approx = qe.markov.approximation.tauchen(rho[j], math.sqrt(sigma[k]**2*(1-rho[j]**2)), m =m,n=n) # Generate a Tauchen approx. with the chosen parameters


beta = 0.96
alpha = 0.36
delta = 0.08
hours = 1

grid_min  = 1e-3
grid_max  = 30
grid_size = 1000
step = (grid_max-grid_min)/grid_size
a = np.linspace(grid_min, grid_max, grid_size)
z_grid = 7

pi_z = tauchen_approx.P #transition matrix
z = tauchen_approx.state_values # state values
z = np.exp(z) # exp so we don't have negative values

pi_z_temp = np.power(pi_z,1000)
pi_z_inv = tauchen_approx.stationary_distributions

N = 0
for i in range(7):
    N = N + hours*z[i]*pi_z_inv[0][i]

#Define the functions
@jit("float64(float64)",nopython = True)
def u(x):
    if gamma == 1:
        return(math.log(x))
    else:
        return (x**(1-gamma)-1)/(1-gamma)


@jit(nopython = True)
def utility(util, r, w):
    for ia in range(grid_size):
        for iz in range(z_grid):
            for ja in range(grid_size):
                c = (1+r)*a[ia] + w*hours*z[iz] - a[ja]
                if (c>0):
                    util[ia,iz,ja] = u(c)
                else:
                    util[ia,iz,ja] = -100000000


@jit
def update(V,V_new,ga,gc,ind_a,r,w):
    for ia in range(grid_size):
        for iz in range(z_grid):
            vtemp = np.repeat(-math.inf,grid_size)
            for ja in range(grid_size):
                EV = 0
                for jz in range(z_grid):
                    EV = EV + pi_z[iz,jz]*V[ja,jz]
                vtemp[ja] = util[ia,iz,ja] + beta*EV
            V_max = max(vtemp)
            temp_list = list(vtemp)
            ind_a[ia,iz]  = int(temp_list.index(max(temp_list)))
            V_new[ia,iz] = V_max
            ga[ia,iz] = a[int(ind_a[ia,iz])]
            gc[ia,iz] = (1+r)*a[ia]+w*hours*z[iz] - ga[ia,iz]



@jit(nopython = True)
def lambda_iter(lambda_new, lambda_var, ind_a, pi_z):
    for ia in range(grid_size):
        for iz in range(z_grid):
            for jz in range(z_grid):
                lambda_new[ind_a[ia,iz],jz] = lambda_new[ind_a[ia,iz],jz] + pi_z[iz,jz]*lambda_var[ia,iz] 
    return(lambda_new)


erro_k = 100
iter_k = 0
r = 0.03
r_new = 0.03

V  = np.repeat(0,grid_size*z_grid).reshape(grid_size,z_grid).astype(float)
V_new = np.repeat(0,grid_size*z_grid).reshape(grid_size,z_grid).astype(float)
ind_a = np.repeat(0,grid_size*z_grid).reshape(grid_size,z_grid).astype(int)
ga= np.repeat(0,grid_size*z_grid).reshape(grid_size,z_grid).astype(float)
gc= np.repeat(0,grid_size*z_grid).reshape(grid_size,z_grid).astype(float)
util = np.repeat(0,grid_size*grid_size*z_grid).reshape(grid_size,z_grid,grid_size).astype(float)


while abs(erro_k)>1e-3 and iter_k<200:
    iter_k = iter_k + 1
    erro_v = 100
    iter_v = 0
    r = 0.9*r + 0.1*r_new
    w = (1-alpha)*((alpha/(r+delta))**(alpha/(1-alpha)))
    K = ((alpha*(N**(1-alpha)))/(r+delta))**(1/(1-alpha))
    Y = (K**alpha)*(N**(1-alpha))
    utility(util, r, w)
    while erro_v > 1e-3 and iter_v<200:
        update(V,V_new,ga,gc,ind_a,r,w)
        erro_v = np.max(abs(V_new - V))
        # print(iter_v)
        # print(erro_v) 
        # print("----------//------------")
        iter_v = iter_v + 1
        V = deepcopy(V_new)
    
    
    # This is the simulation part
    lambda_var = np.repeat(0,grid_size*z_grid).reshape(grid_size,z_grid).astype(float)
    lambda_var[0,:] = np.repeat(1/z_grid,z_grid)
    erro_dis = 100
    iter_dis = 0
    while erro_dis > 1e-6 and iter_dis<400:
        iter_dis = iter_dis+1
        lambda_new = np.repeat(0,grid_size*z_grid).reshape(grid_size,z_grid).astype(float)
        lambda_iter(lambda_new, lambda_var, ind_a, pi_z)
        erro_dis = np.max(abs(lambda_new - lambda_var))
        lambda_var = deepcopy(lambda_new)
    
    Am = 0
    EV_bench = 0
    EV_counter = 0
    for ia in range(grid_size):
        for iz in range(z_grid):
            Am = Am + ga[ia,iz]*lambda_var[ia,iz]
            EV_bench = EV_bench + V[ia,iz]*lambda_var[ia,iz]

    erro_k = (K-Am)/K
    K = Am
    r_new = alpha*((K/N)**(alpha-1))-delta
    
    
    print("erro_k = "+str(erro_k))
    print("iter_k = "+str(iter_k))
    print("r_new = "+str(r_new))
    print("r = "+str(r))
    print("Am = "+str(Am))
    print("savings rate is "+ str(delta*alpha/(r+delta)))
    print("--------------------------------")


# Value and policy functions plot

plt.plot(a.T,V[:,0].T, label = "Low Productivity")
plt.plot(a.T,V[:,6].T, label = "High Productivity")
plt.xlabel('Assets')
plt.ylabel('Value Function')
plt.title("Value Function")
plt.legend()
plt.show()


plt.plot(a.T,ga[:,0].T, label = "Low Productivity")
plt.plot(a.T,ga[:,6].T, label = "High Productivity")
plt.xlabel('Assets')
plt.ylabel('Policy for Assets')
plt.title("Policy for Assets")
plt.legend()
plt.show()

plt.plot(a.T,gc[:,0].T, label = "Low Productivity")
plt.plot(a.T,gc[:,6].T, label = "High Productivity")
plt.xlabel('Assets')
plt.ylabel('Policy for Consumption')
plt.title("Policy for Consumption")
plt.legend()
plt.show()
