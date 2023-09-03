#!/usr/bin/env python
# coding: utf-8

# # $\text{Imported Libraries and Data}$

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sea
import time

z_obs_val = np.loadtxt("mcmc_mu_data.txt")[:,0]
mu_obs_val = np.loadtxt("mcmc_mu_data.txt")[:,1]
covarince = np.loadtxt("mcmc_covariance_data.txt")

C = covarince.reshape(31,31)
C_inverse = np.linalg.inv(C)


# # $\text{Model or Functions}$

# In[2]:


def s_(omega_m):
    k = ((1-omega_m)/omega_m)**(1/3)
    return k

def eta(a,omega_m):
    s = s_(omega_m)
    t1 = ((1/a**4)-(0.1540*s/a**3)+0.4304*(s**2)/a**2)
    t2 = ((0.19097*(s**3)/a) + 0.066941*s**4)
    return 2*(np.sqrt(s**3 + 1))*(t1 + t2)**(-1/8)

def D_l(z,omega_m):
    H0 = 100
    c = 3*10**5
    t1 = eta(1,omega_m) - eta(1/(1+z),omega_m)
    return (c*(1+z)/H0)*t1

def mu(z,h,omega_m):
    return 25 - 5*np.log10(h) + 5*np.log10(D_l(z,omega_m))

def like_h(h,omega_m):
    summ = 0
    mu_theory = mu(z_obs_val,h,omega_m)
    for i in range(len(mu_obs_val)):
        for j in range(len(mu_obs_val)):
            summ += (mu_obs_val[i]- mu_theory[i])*C_inverse[i,j]*(mu_obs_val[j]- mu_theory[j])
    return -0.5*summ
### here in likely-hood, we take final value in the form of log. because we were facing some nan values without log


# # $\text{Part-1 Plots for Different h & Ω𝑚 values}$

# In[3]:


mu1 = mu(z_obs_val,0.7,0.3)
mu2 = mu(z_obs_val,0.5,0.1)
mu3 = mu(z_obs_val,0.9,0.7)
mu4 = mu(z_obs_val,0.6,0.2)


# In[4]:


plt.figure(figsize=(15,12))
plt.plot(z_obs_val,mu_obs_val,label="observed data",lw = 2.5,color = 'yellow')
plt.plot(z_obs_val,mu1,label="h = 0.7 and $\Omega_m$ = 0.3",ls= '--',color = 'red')
plt.plot(z_obs_val,mu2,label="h = 0.6 and $\Omega_m$ = 0.2",color = 'green')
plt.plot(z_obs_val,mu3,label="h = 0.9 and $\Omega_m$ = 0.7",color = 'purple')
plt.plot(z_obs_val,mu4,label="h = 0.5 and $\Omega_m$ = 0.6",color = 'black')
plt.legend(fontsize = 15,edgecolor='black',framealpha=1.0)
plt.title('Distance modulus for Supernova Ia ',fontsize = 25)
plt.xticks(fontsize = 15,color = 'red')
plt.xlabel('Redshift z - values',fontsize = 20,color = 'red')
plt.yticks(fontsize = 15,color = 'red')
plt.ylabel('Distance Modulus $\mu$ - values',fontsize = 20,color = 'red')
plt.show()


# # $\text{Part-2 Parameter Estimation}$

# ## $\text{ MCMC algorithm}$

# In[5]:


def MCMC(steps,h0,omega0,sigma1,sigma2):
    
    burn_in = int(0.2*steps)  ## to burn in some starting values
    
    h_vals = np.zeros(steps)
    omega_vals = np.zeros(steps)
    h_vals[0] = h0                 ## initial value of h and Ω𝑚
    omega_vals[0] = omega0
    
    Accept_value = 0 ## to count acceptance ratio
    
    for i in range(1,steps):
        h_current = h_vals[i-1]
        omega_current = omega_vals[i-1]
        
        ### now we will found proposed value of h and Ω𝑚 by generate random number from uniform distribution

        h_proposed = np.random.normal(h_current,sigma1)
        omega_proposed = np.random.normal(omega_current,sigma2)

        t1 = like_h(h_current,omega_current) ## likely hood for current h and Ω𝑚
        
        ## since we know that h and Ω𝑚 can not take negative value as well as more than 1,so we put this condition
        if h_proposed<=0 or h_proposed>=1 or  omega_proposed<=0 or omega_proposed>=1:
            t2 = -np.inf     ####here we took infinite value such that likely hood becomes zero for this condition

        else:
            t2 = like_h(h_proposed,omega_proposed) ## likely hood for proposed h and Ω𝑚
            
        acceptance_prob = min(1,np.exp(t2-t1)) # here we use exponantial because above we took log of likely hood
        
        ## Choosing an random number between 0 and 1
        delta = np.random.rand(1)[0]
        
        if delta<acceptance_prob:
            h_vals[i] = h_proposed
            omega_vals[i] = omega_proposed
            Accept_value += 1
            
        else:
            h_vals[i] = h_current
            omega_vals[i] = omega_current
            
    ### now for Acceptance ratio
    Accept_ratio = Accept_value/steps
            
    return h_vals[burn_in:],omega_vals[burn_in:],Accept_ratio


# In[6]:


h_vals1,omega_vals1,ar1 = MCMC(1000,0.2,0.9,0.01,0.03)
h_vals2,omega_vals2,ar2 = MCMC(1000,0.5,0.5,0.01,0.03)
h_vals3,omega_vals3,ar3 = MCMC(1000,0.9,0.2,0.01,0.03)


# ## $\text{Visualization of Data by Histogram and Plots}$

# ### $\text{Histograms for different ℎ𝑜 & Ω𝑚𝑜 values}$

# In[7]:


plt.hist(omega_vals1,color = 'red',label = '$\Omega_m $ values ',bins = 15)
plt.hist(h_vals1,color = 'purple',label = 'h values')
plt.title("For $ ℎ_o $ = 0.2 and $\Omega_{mo}$ = 0.9 values",fontsize = 20)
plt.xticks(fontsize = 15,color = 'red')
plt.xlabel('Parameters $ h$ and $\Omega_m$',fontsize = 20,color = 'red')
plt.yticks(fontsize = 15,color = 'red')
plt.ylabel('Frequency',fontsize = 20,color = 'red')
plt.legend()
plt.show()

plt.hist(omega_vals2,color = 'black',label = '$\Omega_m $ values',bins = 12,alpha = 0.2)
plt.hist(h_vals2,color = 'yellow',label = 'h values',alpha = 0.5)
plt.title("For $ ℎ_o $ = 0.5 and $\Omega_{mo}$ = 0.5 values",fontsize = 20)
plt.xticks(fontsize = 15,color = 'black')
plt.xlabel('Parameters $ h$ and $\Omega_m$',fontsize = 20,color = 'red')
plt.yticks(fontsize = 15,color = 'red')
plt.ylabel('Frequency',fontsize = 20,color = 'red')
# plt.legend(fontsize = 20)
plt.show()

plt.hist(omega_vals3,color = 'blue',label = '$\Omega_m $ values',bins = 12,alpha = 0.4)
plt.hist(h_vals3,color = 'skyblue',label = 'h values',alpha = 0.4)
plt.title("For $ ℎ_o $ = 0.9 and $\Omega_{mo}$ = 0.2 values",fontsize = 20)
plt.xticks(fontsize = 15,color = 'red')
plt.xlabel('Parameters $ h$ and $\Omega_m$',fontsize = 20,color = 'red')
plt.yticks(fontsize = 15,color = 'red')
plt.ylabel('Frequency',fontsize = 20,color = 'red')
# plt.legend(fontsize = 20)
plt.show()


# ### $\text{Plots for different ℎ𝑜 & Ω𝑚𝑜 values}$

# In[7]:


figure, ax = plt.subplots(figsize=(10,8))
ax.plot(omega_vals1,h_vals1,color='red',alpha=0.6,marker = 'o',ms=6,                                       label="for $h_0 = 0.2$ and $\Omega_{m_0}$ = 0.9")
ax.plot(omega_vals2,h_vals2,color='black',alpha=0.5,marker = 'o',ms=4,                                       label="for $h_0 = 0.5$ and $\Omega_{m_0}$ = 0.5")
ax.plot(omega_vals3,h_vals3,color='green',alpha=0.4,marker = '*',ms=2,                                        label="for $h_0 = 0.9$ and $\Omega_{m_0}$ = 0.2")

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Parameter estimation",fontsize = 20,fontweight="bold")
plt.xlabel('$\Omega_m $ values',fontsize=18,fontweight = "bold")
plt.ylabel('h values',fontsize=18,fontweight = "bold")
ax.set_xlim(0,0.6)
ax.set_ylim(0.5,0.8)
plt.legend(fontsize = 15,facecolor = 'white',edgecolor = 'black',framealpha = 0.8)
plt.show()


# ### $\text{Animated Plots}$
# #### $\text{Basically this part of the code visually shows us step by step that whatever the initial values of ℎ & Ω𝑚 are, their final values will }$ $\text{  converge to the expected values of ℎ & Ω𝑚. but here is an possibility that this animation run in jupyter notebook only.}$ $\text{ So please if possible than run this in jupyter notbook instead of google colab. }$ 

# In[8]:


get_ipython().run_line_magic('matplotlib', 'tk')

figure, ax = plt.subplots(figsize=(15,14))
ax.plot(omega_vals1[0],h_vals1[0],color='red',alpha=0.4,marker = 'o',ms=2,                                               label="for $h_0 = 0.2$ and $\Omega_{m_0}$ = 0.9")
ax.plot(omega_vals2[0],h_vals2[0],color='black',alpha=0.4,marker = 'o',ms=2,                                               label="for $h_0 = 0.5$ and $\Omega_{m_0}$ = 0.5")
ax.plot(omega_vals3[0],h_vals3[0],color='green',alpha=0.4,marker = 'o',ms=2,                                               label="for $h_0 = 0.9$ and $\Omega_{m_0}$ = 0.2")

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Parameter estimation",fontsize = 35,fontweight="bold")
plt.xlabel('$\Omega_m $ values',fontsize=18,fontweight = "bold")
plt.ylabel('h values',fontsize=18,fontweight = "bold")
ax.set_xlim(0.2,0.5)
ax.set_ylim(0.65,0.75)
plt.legend(fontsize =18,framealpha = 0.8,edgecolor = 'black',facecolor = 'white')
plt.show()

for i in range(1,500):
    ax.plot(omega_vals1[i],h_vals1[i],color='red',marker = 'o',alpha=0.4,ms=2)
    ax.plot(omega_vals2[i],h_vals2[i],color='black',marker = 'o',alpha=0.4,ms=2)
    ax.plot(omega_vals3[i],h_vals3[i],color='green',marker = 'o',alpha=0.4,ms=2)
    
    ax.plot([omega_vals1[i-1],omega_vals1[i]],[h_vals1[i-1],h_vals1[i]],color='red',ls='-',alpha=0.4,ms=3)
    ax.plot([omega_vals2[i-1],omega_vals2[i]],[h_vals2[i-1],h_vals2[i]],color='black',ls='-',alpha=0.4,ms=3)
    ax.plot([omega_vals3[i-1],omega_vals3[i]],[h_vals3[i-1],h_vals3[i]],color='green',ls='-',alpha=0.4,ms=3)
    if i == 300:
        plt.axvline(0.3,color='purple',lw = 3,ls = '--',ms = 3,label= '$\Omega_m$ = 0.3 line')
        plt.axhline(0.7,color='black',lw = 2.5,ls = '--',ms = 3,label = 'h = 0.7 line')
        plt.legend(fontsize =20,framealpha = 0.2,edgecolor = 'black')

    plt.pause(1.e-150)    ### to pause every step for little time(second) such that we can see all steps
plt.show()
    



# ## $\text{Acceptance Ratio}$

# In[10]:


sigma1 = np.linspace(0.001,0.01,10)
sigma2 = np.linspace(0.02,0.1,10)
sigma3 = np.linspace(0.2,1.0,10)
sigma4 = np.linspace(2,10,10)
sigma5 = np.linspace(11,100,10)

sigma = np.concatenate([sigma1,sigma2,sigma3,sigma4,sigma5])

AR_values = np.zeros(len(sigma))

for i in range(len(sigma)):
    AR_values[i] = MCMC(10000,0.2,0.8,sigma[i],sigma[i])[2]
    


# In[11]:


plt.figure(figsize=(14,8))
plt.plot(sigma,AR_values,color='blue',ls = '--')
plt.scatter(sigma,AR_values,color='red')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(' size of proposal distribution or standerd deviation $\sigma$ ',fontsize=18,fontweight = "bold")
plt.ylabel('Acceptance ratio',fontsize=18,fontweight = "bold")
plt.title('Effect of standerd deviation on acceptence ratio',fontsize=18,fontweight = "bold")
plt.xscale('log')
# plt.yscale('log')
plt.show()


# ## $\text{Statistics of above data}$

# ### $\text{Average Value of Estimated Parameters }$

# In[12]:


### from above plot Acceptance ratio vs sigma, we can seen that acceptance ratio is decreasing with increasing 
# sigma,so we can take any value of variance in this decresing phase. 

## sigma[10:18] we can take one of them ;  let we are taking sigma[16]

h_final,omega_final,arr= MCMC(1000,0.1,0.9,sigma[16],sigma[16])
avg_h = np.mean(h_final)
avg_omega = np.mean(omega_final)
print('h =' ,avg_h,'omega =',avg_omega)


# ### $\text{Average,Variance & Covariance of estimated parameters}$

# In[13]:


avg_h = np.mean(h_final)
avg_omega = np.mean(omega_final)

variance_h = np.var(h_final)
covariance_h = np.cov(h_final)

variance_omega = np.var(omega_final)
covariance_omega = np.cov(omega_final)

print("average value of h = ", avg_h)
print('average value of omega = ', avg_omega)

print('variance of h values = ',variance_h)
print('covariance of h values = ',covariance_h)

print('variance of omega values = ',variance_omega)
print('covariance of omega values = ',covariance_omega)


# ## $\text{Final Plot of Distance Modulus with Estimated Parameters}$

# In[14]:


estimated_distance_modulus = mu(z_obs_val,avg_h,avg_omega)
plt.figure(figsize=(15,12))
plt.scatter(z_obs_val,mu_obs_val,label = 'Observed distance modulus',color='red',lw = 3.0)
plt.plot(z_obs_val,estimated_distance_modulus,label = 'Estimated distance modulus',                                                                   color='black',ls = '--',lw = 2)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('redshift z values',fontsize=18,fontweight = "bold")
plt.ylabel('Distance modulus $\mu$ values',fontsize=18,fontweight = "bold")
plt.legend(fontsize = 18,edgecolor = 'black',framealpha = 0.8,facecolor = 'white')
plt.show()


# In[ ]:




