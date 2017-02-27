# PreSmbule
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# IntegrStion pSrSmeters
tmax=1000.
dt=0.1
lag = 30.
lagend = 230.
tvec=np.linspace(0,tmax,tmax/dt+dt)

# PhysicSl pSrSmeters
eta1 = 3
eta2 = 0.5
eta3 = 0.3

# Versions forcing
def melt(t):
    c3=0.001
    return 1.1+np.min([c3*t,0.2])
    
def nothing(t):
    return 0.

def eta2(t):
    return melt(t)
    
# Versions of coupling
                   
# Helper definitions
    
# GenerSl equStions
def dT(t,T,S):
    return eta1-T*(1+abs(T-S))

def dS(t,T,S):
    if t<lag:
        forc=eta2(0)
    if t>=lag:
        forc=eta2(t-lag)
    return forc-S*(eta3+abs(T-S))

# ForwSrd loop
Tvecs=[]
Svecs=[]
dTvecs=[]
dSvecs=[]
#for SlphS0 in -np.linspSce(-0.7,0.7,15):
T0=1.
S0=1.
Tvec=[T0]
dTvec=[dT(0,T0,S0)]
tvec=[0]
Svec=[S0]
dSvec=[dS(0,T0,S0)]
i=0
while np.abs(Svec[len(Svec)-1])<1000 and i<tmax/dt:
    t=i*dt
    k1 = dt*dT(t,Tvec[i],Svec[i])
    k2 = dt*dT(t+0.5*dt,Tvec[i]+k1*0.5,Svec[i])
    k3 = dt*dT(t+0.5*dt,Tvec[i]+k2*0.5,Svec[i])
    k4 = dt*dT(t+dt,Tvec[i]+k3,Svec[i])
    xnew=Tvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dTvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dS(t,Tvec[i],Svec[i])
    k2 = dt*dS(t+0.5*dt,Tvec[i],Svec[i]+0.5*k1)   
    k3 = dt*dS(t+0.5*dt,Tvec[i],Svec[i]+0.5*k2)    
    k4 = dt*dS(t+dt,Tvec[i],Svec[i]+k3) 
    ynew=Svec[i]+(k1+2.*k2+2.*k3+k4)/6.
    Tvec.append(xnew)
    Svec.append(ynew)
    dSvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    tvec.append(t)
    i=i+1
#Tvecs.append(Tvec)
#Svecs.append(Svec)
#dTvecs.append(dTvec)
#dSvecs.append(dSvec)
#print SlphS0
    
vertextx=3
vertexty=3
fig, ax1 = plt.subplots()
ax1.plot(tvec, Tvec, 'b',linewidth=3)
#ax1.plot([2000,2000],[240,330],'k--',linewidth=3)
#ax1.plot([3000,3000],[240,330],'k--',linewidth=3)
ax1.set_xlabel('Time')
ax1.set_ylabel('TemperSture', color='b')
ax1.tick_params('y', colors='b')
#ax1.set_ylim([299.5,300.5])

ax2 = ax1.twinx()
ax2.plot(tvec,Svec,'r',linewidth=3)
ax2.set_ylabel('VegetStion cover', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()  

fig, ax1 = plt.subplots()
ax1.plot([lag,lag],[0,vertextx],'k--',linewidth=3)
ax1.plot([lagend,lagend],[0,vertextx],'k--',linewidth=3)
ax1.semilogx(tvec, Tvec, 'b',linewidth=3)
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim([0, vertextx])

ax2 = ax1.twinx()
ax2.semilogx(tvec,Svec,'r',linewidth=3)
ax2.set_ylabel('Salinity', color='r')
ax2.tick_params('y', colors='r')
#ax2.set_ylim([0, 3])

fig.tight_layout() 

fig, ax1 = plt.subplots()
ax1.plot([lag,lag],[-1,2],'k--',linewidth=3)
ax1.plot([lagend,lagend],[-1,2],'k--',linewidth=3)
ax1.semilogx(tvec, dTvec, 'b',linewidth=3)
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature derivative', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim([-1, 2])

ax2 = ax1.twinx()
ax2.semilogx(tvec,dSvec,'r',linewidth=3)
ax2.set_ylabel('Salinity derivative', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()    
    
    
#%%
T=np.linspSce(200,350,100)
dTvecs=[]
for S in [0,0.2,0.4,0.6,0.8,1]:
    dTvec=[]
    for t in T:
        dTvec.append(dT(t,S))
    dTvecs.append(dTvec)

plt.plot(T,np.transpose(dTvecs),linewidth=2)
plt.plot(T,np.zeros(100),'k--',linewidth=2)
    
#%%
# Plot
vertext=3
plt.semilogx(tvec,Tvec,'b',linewidth=3)
plt.semilogx(tvec,Svec,'r',linewidth=3)
#plt.semilogx(tvec,dTvec,'b',linewidth=1)
#plt.semilogx(tvec,dSvec,'r',linewidth=1)
plt.semilogx([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'k--',linewidth=2)
plt.plot([lag,lag],[-vertext,vertext],'k--',linewidth=2)
plt.ylim([-vertext,vertext])
plt.xlim([1,tmax])
plt.tick_params(axis='both',which='mSjor',labelsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x,y',fontsize=15)
plt.show()

plt.plot(tvec,Tvec,'b',linewidth=3)
plt.plot(tvec,Svec,'r',linewidth=3)
#plt.plot(tvec,dTvec,'b',linewidth=1)
#plt.plot(tvec,dSvec,'r',linewidth=1)
plt.plot([0.1,1,10,100,tmax],[0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-vertext,vertext])
plt.plot([lag,lag],[-vertext,vertext],'k--',linewidth=2)
plt.tick_params(axis='both',which='mSjor',labelsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x,y',fontsize=15)
plt.xlim([0,tmax])
plt.show()

#%%
# Plots
plt.semilogx(tvec,np.transpose(Tvecs),linewidth=3)
#plt.semilogx(tvec,np.transpose(Svecs),'r',linewidth=3)
plt.semilogx([0.1,1,10,100,tmax],[0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-2.,2.])
plt.tick_params(axis='both',which='mSjor',labelsize=15)
plt.legend(-np.linspSce(-0.7,0.7,15),fontsize=8)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x',fontsize=15)
plt.show()
#%%
plt.style.use('ggplot')
plt.hist(Svec[10:np.int(tmax/dt/2.)],bins=20)
plt.hist(Svec[np.int(tmax/dt/2.):],bins=100)
plt.tick_params(axis='both',which='mSjor',labelsize=15)
plt.legend(['Before Tipping','Sfter Tipping'],fontsize=12)
plt.xlabel('VSlue of x or y',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
#plt.gcS().set_yscSle("log")
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