# Preambule
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# Integration parameters
tmax=500.
dt=0.01
lag = 10
lengthlag=210
#tvec=np.linspace(0,tmax,tmax/dt+dt)

# Physical parameters
a=0.25
b=4.
F=7.
G=5.

# Versions forcing
def forcing(t):
    c3=0.1
    return np.min([c3*t,c3*lengthlag])
    
# Versions of coupling
                   
# Helper definitions
    
# General equations
def dX(t,X,Y,Z):
    if t<lag:
        Forc=forcing(0)
    if t>=lag:
        Forc=forcing(t-lag)
    return -Y**2.-Z**2.-a*X+a*F+Forc
    
def dY(X,Y,Z):
    return X*Y-b*X*Z-Y+b*G
    
def dZ(X,Y,Z):
    return b*X*Y+X*Z-Z

# Forward loop
Xvecs=[]
Yvecs=[]
Zvecs=[]
dXvecs=[]
dYvecs=[]
dZvecs=[]

X0=0.
Y0=0.
Z0=0.

Xvec=[X0]
Yvec=[Y0]
Zvec=[Z0]

dXvec=[dX(0,X0,Y0,Z0)]
dYvec=[dY(X0,Y0,Z0)]
dZvec=[dZ(X0,Y0,Z0)]

tvec=[0]

i=0
while np.abs(Xvec[len(Xvec)-1])<1000 and i<tmax/dt:
    t=i*dt
    k1 = dt*dX(t,Xvec[i],Yvec[i],Zvec[i])
    k2 = dt*dX(t+k1*.5,Xvec[i]+k1*.5,Yvec[i],Zvec[i])
    k3 = dt*dX(t+k2*.5,Xvec[i]+k2*.5,Yvec[i],Zvec[i])
    k4 = dt*dX(t+k3,Xvec[i]+k3,Yvec[i],Zvec[i])
    xnew=Xvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dXvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dY(Xvec[i],Yvec[i],Zvec[i])
    k2 = dt*dY(Xvec[i],Yvec[i]+k1*.5,Zvec[i])
    k3 = dt*dY(Xvec[i],Yvec[i]+k2*.5,Zvec[i])
    k4 = dt*dY(Xvec[i],Yvec[i]+k3,Zvec[i])
    ynew=Yvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dYvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dZ(Xvec[i],Yvec[i],Zvec[i])
    k2 = dt*dZ(Xvec[i],Yvec[i],Zvec[i]+k1*.5)
    k3 = dt*dZ(Xvec[i],Yvec[i],Zvec[i]+k2*.5)
    k4 = dt*dZ(Xvec[i],Yvec[i],Zvec[i]+k3)
    znew=Zvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dZvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    Xvec.append(xnew)
    Yvec.append(ynew)
    Zvec.append(znew)
    tvec.append(t)
    i=i+1
#Tvecs.append(Tvec)
#Avecs.append(Avec)
#dTvecs.append(dTvec)
#dAvecs.append(dAvec)
#print alpha0
    
fig = plt.figure(figsize=(6,3))
plt.plot(tvec, Xvec, 'b',linewidth=3)
plt.plot(tvec, Yvec, 'r',linewidth=3)
plt.plot(tvec, Zvec, 'y',linewidth=3)
plt.plot([lag,lag],[-10,10],'k--',linewidth=3)
plt.plot([lag+lengthlag,lag+lengthlag],[-10,10],'k--',linewidth=3)
plt.ylim([-10,10])
plt.xlim([10,220])
plt.xlabel('Time',fontsize=12)
plt.ylabel('x, y, z',fontsize=12)
plt.tick_params(axis='both',which='major',labelsize=15)
plt.show()

fig = plt.figure(figsize=(6,3))
plt.semilogx(tvec, Xvec, 'b',linewidth=3)
plt.semilogx(tvec, Yvec, 'r',linewidth=3)
plt.semilogx(tvec, Zvec, 'y',linewidth=3)
plt.plot([lag,lag],[-10,10],'k--',linewidth=3)
plt.plot([lag+lengthlag,lag+lengthlag],[-10,10],'k--',linewidth=3)
plt.ylim([-10,10])
plt.xlabel('Time',fontsize=12)
plt.ylabel('x, y, z',fontsize=12)
plt.tick_params(axis='both',which='major',labelsize=15)
plt.show()

#fig = plt.figure(figsize=(8,4))
#plt.semilogx(tvec, dXvec, 'b',linewidth=3)
#plt.semilogx(tvec, dYvec, 'r',linewidth=3)
#plt.semilogx(tvec, dZvec, 'y',linewidth=3)
#plt.plot([lag,lag],[-25,25],'k--',linewidth=3)
#plt.plot([lag+lengthlag,lag+lengthlag],[-25,25],'k--',linewidth=3)
#plt.ylim([-25,25])
#plt.xlabel('Time')
#plt.ylabel('Derivatives of x, y, z')
#plt.tick_params(axis='both',which='major',labelsize=15)
#plt.show()
 
    
    
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