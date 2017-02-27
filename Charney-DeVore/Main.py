# Preambule
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# Integration parameters
tmax=5000.
dt=0.1
lag = 2000
tvec=np.linspace(0,tmax,tmax/dt+dt)

# Physical parameters

# Versions forcing
def greenhouse(t):
    c3=0.0001
    return np.min([c3*t,0.1])
    
def nothing(t):
    return 0.

def forcing(t):
    return greenhouse(t)
    
# Versions of coupling
                   
# Helper definitions
def alpha(T,A):
    return (1-p)*alpha0(T)#+p*(av*A+ag*(1-A))

def beta(T):
    return max(0,1-k*(T-Topt)**2)

def alpha0(T):
    return max(alphamax+min(0,(alphamin-alphamax)*(T-Tal)/(Tau-Tal)),alphamin)

def R0(T):
    return B0+B1*(T-Topt)
    
# General equations
      F(1) = -k01*(Xa-Xa0)+h01*Xl
      F(2) = -(alphan1*Xa-betan1)*Xl-deltan1*Xc*Xn-kn1*(Xk-Xk0)
      F(3) = (alphan1*Xa-betan1)*Xk+deltan1*Xc*Xm-kn1*(Xl-Xl0)-hn1*Xa
      F(4) = epsilonn*(Xk*Xn-Xl*Xm)-k02*(Xc-Xc0)+h02*Xn
      F(5) = -(alphan2*Xa-betan2)*Xn-deltan2*Xc*Xl-kn2*(Xm-Xm0)
      F(6) = (alphan2*Xa-betan2)*Xm+deltan2*Xc*Xk+hn2*Xc-kn2*(Xn-Xn0)
def psia(Xa,Xl):
    return -k01*(Xa-Xa0)+h01*Xl

# Forward loop
Tvecs=[]
Avecs=[]
dTvecs=[]
dAvecs=[]
#for alpha0 in -np.linspace(-0.7,0.7,15):
T0=240.
A0=0.5
Tvec=[T0]
dTvec=[dT(0,0,T0)]
tvec=[0]
Avec=[A0]
dAvec=[dA(T0,A0)]
i=0
while np.abs(Avec[len(Avec)-1])<1000 and i<tmax/dt:
    t=i*dt
    k1 = dt*dT(t+0.5*dt,Tvec[i],Avec[i])
    k2 = dt*dT(t+0.5*dt,Tvec[i]+k1*0.5,Avec[i])
    k3 = dt*dT(t+0.5*dt,Tvec[i]+k2*0.5,Avec[i])
    k4 = dt*dT(t+0.5*dt,Tvec[i]+k3,Avec[i])
    xnew=Tvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dTvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dA(Tvec[i],Avec[i])
    k2 = dt*dA(Tvec[i],Avec[i]+0.5*k1)   
    k3 = dt*dA(Tvec[i],Avec[i]+0.5*k2)    
    k4 = dt*dA(Tvec[i],Avec[i]+k3) 
    ynew=Avec[i]+(k1+2.*k2+2.*k3+k4)/6.
    Tvec.append(xnew)
    Avec.append(ynew)
    dAvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    tvec.append(t)
    i=i+1
#Tvecs.append(Tvec)
#Avecs.append(Avec)
#dTvecs.append(dTvec)
#dAvecs.append(dAvec)
#print alpha0
    
fig, ax1 = plt.subplots()
ax1.plot(tvec, Tvec, 'b',linewidth=3)
ax1.plot([2000,2000],[240,330],'k--',linewidth=3)
ax1.plot([3000,3000],[240,330],'k--',linewidth=3)
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature', color='b')
ax1.tick_params('y', colors='b')
#ax1.set_ylim([299.5,300.5])

ax2 = ax1.twinx()
ax2.plot(tvec,Avec,'r',linewidth=3)
ax2.set_ylim([0,1])
ax2.set_ylabel('Vegetation cover', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()  

fig, ax1 = plt.subplots()
ax1.plot([2000,2000],[240,330],'k--',linewidth=3)
ax1.plot([3000,3000],[240,330],'k--',linewidth=3)
ax1.semilogx(tvec, Tvec, 'b',linewidth=3)
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature', color='b')
ax1.tick_params('y', colors='b')
#ax1.set_ylim([299.5,300.5])

ax2 = ax1.twinx()
ax2.semilogx(tvec,Avec,'r',linewidth=3)
ax2.set_ylim([0,1])
ax2.set_ylabel('Vegetation cover', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout() 

fig, ax1 = plt.subplots()
ax1.plot([2000,2000],[0,1.3],'k--',linewidth=3)
ax1.plot([3000,3000],[0,1.3],'k--',linewidth=3)
ax1.set_ylim([0,0.4])
ax1.plot(tvec, dTvec, 'b',linewidth=3)
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature derivative', color='b')
ax1.tick_params('y', colors='b')
#ax1.set_ylim([299.5,300.5])

ax2 = ax1.twinx()
ax2.plot(tvec,dAvec,'r',linewidth=3)
ax2.set_ylim([-0.01,0.01])
ax2.set_ylabel('Vegetation cover derivative', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()    
    
    
#%%
T=np.linspace(200,350,100)
dTvecs=[]
for a in [0,0.2,0.4,0.6,0.8,1]:
    dTvec=[]
    for t in T:
        dTvec.append(dT(t,a))
    dTvecs.append(dTvec)

plt.plot(T,np.transpose(dTvecs),linewidth=2)
plt.plot(T,np.zeros(100),'k--',linewidth=2)
    
#%%
# Plot
vertext=3
plt.semilogx(tvec,Tvec,'b',linewidth=3)
plt.semilogx(tvec,Avec,'r',linewidth=3)
#plt.semilogx(tvec,dTvec,'b',linewidth=1)
#plt.semilogx(tvec,dAvec,'r',linewidth=1)
plt.semilogx([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'k--',linewidth=2)
plt.plot([lag,lag],[-vertext,vertext],'k--',linewidth=2)
plt.ylim([-vertext,vertext])
plt.xlim([1,tmax])
plt.tick_params(axis='both',which='major',labelsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x,y',fontsize=15)
plt.show()

plt.plot(tvec,Tvec,'b',linewidth=3)
plt.plot(tvec,Avec,'r',linewidth=3)
#plt.plot(tvec,dTvec,'b',linewidth=1)
#plt.plot(tvec,dAvec,'r',linewidth=1)
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
plt.semilogx(tvec,np.transpose(Tvecs),linewidth=3)
#plt.semilogx(tvec,np.transpose(Avecs),'r',linewidth=3)
plt.semilogx([0.1,1,10,100,tmax],[0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-2.,2.])
plt.tick_params(axis='both',which='major',labelsize=15)
plt.legend(-np.linspace(-0.7,0.7,15),fontsize=8)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x',fontsize=15)
plt.show()
#%%
plt.style.use('ggplot')
plt.hist(Avec[10:np.int(tmax/dt/2.)],bins=20)
plt.hist(Avec[np.int(tmax/dt/2.):],bins=100)
plt.tick_params(axis='both',which='major',labelsize=15)
plt.legend(['Before Tipping','After Tipping'],fontsize=12)
plt.xlabel('Value of x or y',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
#plt.gca().set_yscale("log")
#plt.ylim([50,10000])

#%%
x1=Tvec
dT1=dTvec


#%%
x2=Tvec
dT2=dTvec
#%%
plt.plot(x1,dT1,linewidth=3)
plt.plot(x2,dT2,linewidth=3)
plt.xlabel(r'$x$',fontsize=15)
plt.ylabel(r'$dT/dt$',fontsize=15)