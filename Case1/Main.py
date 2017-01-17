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
c1=1.
c2=1.
lag=2.
noiser=-0.6
counter=1

# Versions forcing
def alpha_step(t):
    if t==0:
        return -0.6
    if t>0:
        return -0.4
        
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
    return alpha_noisy(t)

# Equations
def mu(x):
    return c1+c2*x

def dx(t,x):
    if t>=lag:
        return alpha_chosen(t-lag)-lda*x+x**3./3.
    else:
        return alpha_chosen(0)-lda*x+x**3./3.

def dy(x,y):
    return beta - mu(x) * y + y**3./3.

# Forward loop
xvecs=[]
yvecs=[]
dxvecs=[]
dyvecs=[]
#for alpha0 in -np.linspace(-0.7,0.7,15):
x0=-1.3
y0=-0.3
xvec=[x0]
dxvec=[dx(0,x0)]
yvec=[y0]
dyvec=[dy(x0,y0)]
for i in range(0,np.int(tmax/dt)-1):
    t=tvec[i]
    k1 = dt*dx(t,xvec[i])
    k2 = dt*dx(t+0.5*dt,xvec[i]+k1*0.5)
    k3 = dt*dx(t+0.5*dt,xvec[i]+k2*0.5)
    k4 = dt*dx(t+dt,xvec[i]+k3)
    xnew=xvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dxvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dy(xvec[i],yvec[i])
    k2 = dt*dy(xvec[i],yvec[i]+0.5*k1)   
    k3 = dt*dy(xvec[i],yvec[i]+0.5*k2)    
    k4 = dt*dy(xvec[i],yvec[i]+k3) 
    ynew=yvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    xvec.append(xnew)
    yvec.append(ynew)
    dyvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
#xvecs.append(xvec)
#yvecs.append(yvec)
#dxvecs.append(dxvec)
#dyvecs.append(dyvec)
#print alpha0
    
# Plot
vertext=2
plt.semilogx(tvec,xvec,'b',linewidth=3)
plt.semilogx(tvec,yvec,'r',linewidth=3)
plt.semilogx(tvec,dxvec,'b',linewidth=1)
plt.semilogx(tvec,dyvec,'r',linewidth=1)
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
plt.plot(tvec,dxvec,'b',linewidth=1)
plt.plot(tvec,dyvec,'r',linewidth=1)
plt.plot([0.1,1,10,100,tmax],[0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-vertext,vertext])
plt.plot([lag,lag],[-vertext,vertext],'k--',linewidth=2)
plt.tick_params(axis='both',which='major',labelsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x,y',fontsize=15)
plt.xlim([0,50])
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