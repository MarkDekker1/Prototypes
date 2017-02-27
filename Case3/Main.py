# Preambule
import numpy as np
import matplotlib.pyplot as plt

# Integration parameters
tmax=1000.
dt=0.1
tvec=np.linspace(0,tmax,tmax/dt+dt)

# Physical parameters
lda=1.
beta=0.1
c1=0.7
c2=1.
lag=500.
noiser=-0.5
counter=1
p=1.

# Versions forcing
def alpha_step(t):
    if t==0:
        return -0.6
    if t>0:
        return 0.6
        
def alpha_seasonal(t):
    alpha0=0.1
    return alpha0*np.sin(t/36.*2.*np.pi)
    
def alpha_slowincrease(t): # with ceiling
    if t==0:
        return -0.6
    return np.min([t/100.-0.4,0.6])
    
def alpha_noisy(t):
    global noiser
    global counter
    counter=counter+1
    if t==0:
        return -0.5
    if np.mod(counter,250)==0:
        noiser= (np.random.random(1)*2-1)*0.8
    return noiser
    
def alpha_chosen(t):
    return alpha_step(t)

# Variations of mu(x)
def mu_simple(x):
    return c1+c2*x
    
def mu_var(x):
    global counter
    global noiser
    c1=0
    c2=2.5
    counter=counter+1
    if np.mod(counter,10)==0:
        noiser= c1+np.random.normal(0,c2*np.max([x,0.2]))
    return noiser
    
def mu_mean(x):
    c1=1.
    c2=5.
    global counter
    global noiser
    counter=counter+1
    if np.mod(counter,10)==0:
        noiser=c1+np.random.normal(c2*x,1)
    return noiser
    
def mu_chosen(x):
    return mu_var(x)

# General equations
def dx(t,x):
    if t>=lag:
        return alpha_chosen(t-lag)-lda*x+p*x**3./3.
    else:
        return alpha_chosen(0)-lda*x+p*x**3./3.

def dy_par(x,y):
    return beta + mu_chosen(x) * y - p* y**3./3.
    
def dy_forc(x,y):
    gamma=-1.
    return beta - gamma*y -p* y**3./3. + mu_var(x)
    
def dy_chosen(x,y):
    return dy_forc(x,y)

# Forward loop
xvecs=[]
yvecs=[]
dxvecs=[]
dyvecs=[]
#for alpha0 in -np.linspace(-0.7,0.7,15):
x0=0.5
y0=-0.5
xvec=[x0]
dxvec=[dx(0,x0)]
tvec=[0]
yvec=[y0]
dyvec=[dy_chosen(x0,y0)]
i=0
while np.abs(yvec[len(yvec)-1])<1000 and i<tmax/dt:
    t=i*dt
    k1 = dt*dx(t,xvec[i])
    k2 = dt*dx(t+0.5*dt,xvec[i]+k1*0.5)
    k3 = dt*dx(t+0.5*dt,xvec[i]+k2*0.5)
    k4 = dt*dx(t+dt,xvec[i]+k3)
    xnew=xvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dxvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dy_chosen(xvec[i],yvec[i])
    k2 = dt*dy_chosen(xvec[i],yvec[i]+0.5*k1)   
    k3 = dt*dy_chosen(xvec[i],yvec[i]+0.5*k2)    
    k4 = dt*dy_chosen(xvec[i],yvec[i]+k3) 
    ynew=yvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    xvec.append(xnew)
    yvec.append(ynew)
    dyvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    tvec.append(t)
    i=i+1
#xvecs.append(xvec)
#yvecs.append(yvec)
#dxvecs.append(dxvec)
#dyvecs.append(dyvec)
#print alpha0
    
# Plot
vertext=3
plt.semilogx(tvec,xvec,'b',linewidth=3)
plt.semilogx(tvec,yvec,'r',linewidth=3)
#plt.semilogx(tvec,dxvec,'b',linewidth=1)
#plt.semilogx(tvec,dyvec,'r',linewidth=1)
plt.semilogx([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'k--',linewidth=2)
plt.plot([lag,lag],[-vertext,vertext],'k--',linewidth=2)
plt.ylim([-vertext,vertext])
plt.xlim([1,tmax])
plt.tick_params(axis='both',which='major',labelsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x,y',fontsize=15)
plt.show()

plt.plot(tvec,xvec,'b',linewidth=3)
plt.plot(tvec,yvec,'r',linewidth=3)
#plt.plot(tvec,dxvec,'b',linewidth=1)
#plt.plot(tvec,dyvec,'r',linewidth=1)
plt.plot([0.1,1,10,100,tmax],[0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-vertext,vertext])
plt.plot([lag,lag],[-vertext,vertext],'k--',linewidth=2)
plt.tick_params(axis='both',which='major',labelsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x,y',fontsize=15)
plt.xlim([0,tmax])
plt.show()

#%%
# Plots
plt.semilogx(tvec,np.transpose(xvecs),linewidth=3)
#plt.semilogx(tvec,np.transpose(yvecs),'r',linewidth=3)
plt.semilogx([0.1,1,10,100,tmax],[0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-2.,2.])
plt.tick_params(axis='both',which='major',labelsize=15)
plt.legend(-np.linspace(-0.7,0.7,15),fontsize=8)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x',fontsize=15)
plt.show()
#%%
plt.style.use('ggplot')
plt.hist(yvec[10:np.int(tmax/dt/2.)],bins=20)
plt.hist(yvec[np.int(tmax/dt/2.):],bins=100)
plt.tick_params(axis='both',which='major',labelsize=15)
plt.legend(['Before Tipping','After Tipping'],fontsize=12)
plt.xlabel('Value of x or y',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
#plt.gca().set_yscale("log")
#plt.ylim([50,10000])

#%%
x1=xvec
dx1=dxvec


#%%
x2=xvec
dx2=dxvec
#%%
plt.plot(x1,dx1,linewidth=3)
plt.plot(x2,dx2,linewidth=3)
plt.xlabel(r'$x$',fontsize=15)
plt.ylabel(r'$dx/dt$',fontsize=15)