# Preambule
import numpy as np
import matplotlib.pyplot as plt

# Integration parameters
tmax=100.
dt=0.01
tvec=np.linspace(0,tmax,tmax/dt+dt)

# Physical parameters
alpha0=0.5
omega=1.
lda=1.
c1=0.
c2=1.
mu=1.

# Equations
def alpha(t):
    if t<14:
        return -0.6
    if t>=14:
        return 0.6
    #return alpha0#*t/400.#np.sin(t/36.*2.*np.pi)
    
def lda(z):
    #return c1+c2*z
    return -0

def dx(x,y,z):
    return lda(z)*x-omega*y-x*(x**2+y**2)

def dy(x,y,z):
    return lda(z)*y+omega*x-y*(x**2+y**2)
    
def dz(t,z):
    return alpha(t)-mu*z+z**3./3.

# Forward loop
zvecs=[]
dzvecs=[]
#for alpha in np.linspace(-2,2.,15):
z0=-0.1
x0=0.1
y0=0.1
zvec=[z0]
xvec=[x0]
yvec=[y0]
dzvec=[dz(0,z0)]
dxvec=[dx(x0,y0,z0)]
dyvec=[dy(x0,y0,z0)]

for i in range(0,np.int(tmax/dt)-1):
    t=tvec[i]
    k1 = dt*dz(t,zvec[i])
    k2 = dt*dz(t+0.5*dt,zvec[i]+k1*0.5)   
    k3 = dt*dz(t+0.5*dt,zvec[i]+k2*0.5)    
    k4 = dt*dz(t+dt,zvec[i]+k3)   
    znew=zvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    zvec.append(znew)
    dzvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dx(xvec[i],yvec[i],zvec[i])
    k2 = dt*dx(xvec[i]+0.5*k1,yvec[i],zvec[i])  
    k3 = dt*dx(xvec[i]+0.5*k2,yvec[i],zvec[i])   
    k4 = dt*dx(xvec[i]+k3,yvec[i],zvec[i])  
    xnew=xvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    xvec.append(xnew)
    dxvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dy(xvec[i],yvec[i],zvec[i])
    k2 = dt*dy(xvec[i],yvec[i]+0.5*k1,zvec[i])  
    k3 = dt*dy(xvec[i],yvec[i]+0.5*k2,zvec[i])   
    k4 = dt*dy(xvec[i],yvec[i]+k3,zvec[i])  
    ynew=yvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    yvec.append(ynew)
    dyvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
#zvecs.append(zvec)
#dzvecs.append(dzvec)
#print mu
# Plot
plt.semilogx(tvec,xvec,'b',linewidth=3)
plt.semilogx(tvec,yvec,'r',linewidth=3)
plt.semilogx(tvec,zvec,'y',linewidth=3)
plt.semilogx([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-2.,2.])
plt.tick_params(axis='both',which='major',labelsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x,y,z',fontsize=15)
plt.show()

plt.plot(tvec,zvec,'b',linewidth=3)
plt.plot(tvec,yvec,'r',linewidth=3)
plt.plot(tvec,zvec,'y',linewidth=3)
plt.plot([0.1,1,10,100,tmax],[0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-2.,2.])
plt.tick_params(axis='both',which='major',labelsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x,y,z',fontsize=15)
plt.show()

#%%
# Plots
    
plt.semilogx(tvec,np.transpose(zvecs),linewidth=3)
#plt.semilogx(tvec,np.transpose(yvecs),'r',linewidth=3)
plt.semilogx([0.01, 0.1,1,10,100,tmax],[0,0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-5.,5.])
plt.tick_params(axis='both',which='major',labelsize=15)
#plt.legend(np.linspace(-2,2,15),fontsize=8,loc='lower left')
plt.xlabel('Time',fontsize=15)
plt.ylabel('z',fontsize=15)
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