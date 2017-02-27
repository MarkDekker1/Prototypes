# Preambule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Integration parameters
tmax=200.
dt=0.05
tvec=np.linspace(0,tmax,tmax/dt+dt)

# Physical parameters
mu=1.
omega=1.
lag=60

c1=1
c2=0.5
p1=1.
p2=1
q1=-1
q2=0.3
s1=1
s2=0.5

x0=0.
y0=1.1
z0=0.5

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Versions forcing
def alpha(t):
    if t>lag:
        return min(-0.1+0.02*(t-lag),0.2)
    else:
        return -0.1
    
def alphax(t):
    return np.sin((t)/12)
    
def alphay(t):
    return np.sin((t)/5)
    
def beta(x,y):
    return c1+c2*y
    
def runningmean(tlen,vec):
    vec2=[]
    for i in range(0,len(vec)-tlen):
        vec2.append(np.mean(vec[i:(tlen+i)]))
    return vec2

# Variations of coupling

# General equations
def dx(t,x,y,z):
    return alpha(t)*x+p1*y-p2*x*(x**2+y**2)-0.4

def dy(t,x,y,z):
    return alpha(t)*y+q1*x-q2*y*(x**2+y**2)-0.2
    
def dz(x,y,z):
    return -s1*z**3+s2*z+beta(x,y)
    
def potentialx(t,x,y,z):
    return -0.5*alpha(t)*x**2-p1*y*x+0.25*p2*x**4+0.5*p2*y**2*x**2
    
def potentialy(t,x,y,z):
    return -0.5*alpha(t)*y**2-q1*y*x+0.25*q2*y**4+0.5*p2*y**2*x**2
    
def potentialz(x,y,z):
    return 0.25*s1*z**4-beta(x,y)*0.5*z**2-s2*z
    
def ev1(x,y,z):
    return mu- 3.*z**2.
    
def ev2(x,y,z):
    return c1+c2*z-2*x**2.-np.sqrt(x**4.+2*x**2*y**2+y**4-omega**2+0J)-2*y**2.
    
def ev3(x,y,z):
    return c1+c2*z-2*x**2.+np.sqrt(x**4.+2*x**2*y**2+y**4-omega**2+0J)-2*y**2.
        
# Forward loop
xvec=[x0]
yvec=[y0]
zvec=[z0]

dxvec=[dx(0,x0,y0,z0)]
dyvec=[dy(0,x0,y0,z0)]
dzvec=[dz(x0,y0,z0)]
       
tvec=[0]
fvec=[alpha(0)]
ev1vec=[ev1(x0,y0,z0)]
ev2vec=[ev2(x0,y0,z0)]
ev3vec=[ev3(x0,y0,z0)]
        
i=0
while np.abs(yvec[len(yvec)-1])<1000 and i<tmax/dt:
    t=i*dt
    
    k1 = dt*dz(xvec[i],yvec[i],zvec[i])
    k2 = dt*dz(xvec[i],yvec[i],zvec[i]+k1*0.5)
    k3 = dt*dz(xvec[i],yvec[i],zvec[i]+k2*0.5)
    k4 = dt*dz(xvec[i],yvec[i],zvec[i]+k3)
    znew=zvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dzvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dx(t,xvec[i],yvec[i],zvec[i])
    k2 = dt*dx(t+0.5*dt,xvec[i]+k1*0.5,yvec[i],zvec[i])
    k3 = dt*dx(t+0.5*dt,xvec[i]+k2*0.5,yvec[i],zvec[i])
    k4 = dt*dx(t+dt,xvec[i]+k3,yvec[i],zvec[i])
    xnew=xvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dxvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dy(t,xvec[i],yvec[i],zvec[i])
    k2 = dt*dy(t+0.5*dt,xvec[i],yvec[i]+0.5*k1,zvec[i])   
    k3 = dt*dy(t+0.5*dt,xvec[i],yvec[i]+0.5*k2,zvec[i])    
    k4 = dt*dy(t+dt,xvec[i],yvec[i]+k3,zvec[i]) 
    ynew=yvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dyvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    xvec.append(xnew)
    yvec.append(ynew)
    zvec.append(znew)
    
    fvec.append(alpha(t))
    
    ev1vec.append(ev1(xnew,ynew,znew))
    ev2vec.append(ev2(xnew,ynew,znew))
    ev3vec.append(ev3(xnew,ynew,znew))
    
    tvec.append(t)
    i=i+1

def producepotentials(tuse):
    xpot=[]
    ypot=[]
    zpot=[]
    for xk in xveccy:
        xpot.append(potentialx(tuse*dt, xk,yvec[tuse],zvec[tuse]))
    for yk in yveccy:
        ypot.append(potentialy(tuse*dt, xvec[tuse],yk,zvec[tuse]))
    for zk in zveccy:
        zpot.append(potentialz(xvec[tuse],yvec[tuse],zk))
    
    ymin1=np.where(ypot==np.min(ypot[0:np.int(res/2.)]))[0][0]
    xmin1=np.where(xpot==np.min(xpot[0:np.int(res/2.)]))[0][0]
    
    ymin2=np.where(ypot==np.min(ypot[np.int(res/2.):]))[0][0]
    xmin2=np.where(xpot==np.min(xpot[np.int(res/2.):]))[0][0]
    
    zmin1=np.where(zpot==np.min(zpot[0:np.int(res/2.)]))[0][0]
    zmin2=np.where(zpot==np.min(zpot[np.int(res/2.):]))[0][0]
    return xpot,ypot,xmin1,xmin2,ymin1,ymin2,zpot,zmin1,zmin2
#%
    
width=5
height=5
bot=5
res=500
vertext=2
t=0
pause = False

xveccy=np.linspace(-width,width,res)
yveccy=np.linspace(-width,width,res)
zveccy=np.linspace(-width,width,res)


tuse=len(tvec)
tend=tmax
vertext=3
fig, ax2=plt.subplots(figsize=(6,5))
line3, = ax2.plot(tvec[0:tuse], xvec[0:tuse], 'b',linewidth=3)
line4, = ax2.plot(tvec[0:tuse], yvec[0:tuse], 'r',linewidth=3)
line7, = ax2.plot(tvec[0:tuse], zvec[0:tuse], 'g',linewidth=3)
line5, = ax2.plot(tvec[0:tuse], fvec[0:tuse], 'k',linewidth=3,zorder=-1)
ax2.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'--',color='Grey',linewidth=2,zorder=-1)
ax2.plot([lag,lag],[-1000,1000],'--',color='Grey',linewidth=2,zorder=-1)
ax2.plot([lag+14,lag+14],[-1000,1000],'--',color='Grey',linewidth=2,zorder=-1)
ax2.set_ylim([-vertext,vertext])
ax2.set_xlim([1,tend])
ax2.tick_params(axis='both',which='major',labelsize=15)
ax2.set_xlabel('Time',fontsize=15)
ax2.set_ylabel('Variables',fontsize=15)
ax2.legend(['x','y','z',r'$\alpha(t)$'],loc='lower right',fontsize=15,labelspacing=-0.1)

#runningmeans
Rmean=620
vertext=1.5
fig, ax2=plt.subplots(figsize=(6,5))
line3, = ax2.plot(tvec[Rmean:], runningmean(Rmean,xvec), 'b',linewidth=3)
line4, = ax2.plot(tvec[Rmean:], runningmean(Rmean,yvec), 'r',linewidth=3)
line7, = ax2.plot(tvec[Rmean:], runningmean(Rmean,zvec), 'g',linewidth=3)
line5, = ax2.plot(tvec[Rmean:], runningmean(Rmean,fvec), 'k',linewidth=3,zorder=-1)
ax2.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'--',color='Grey',linewidth=2,zorder=-1)
ax2.plot([lag,lag],[-1000,1000],'--',color='Grey',linewidth=2,zorder=-1)
ax2.plot([lag+14,lag+14],[-1000,1000],'--',color='Grey',linewidth=2,zorder=-1)
ax2.set_ylim([-vertext,vertext])
ax2.set_xlim([1,tend])
ax2.tick_params(axis='both',which='major',labelsize=15)
ax2.set_xlabel('Time',fontsize=15)
ax2.set_ylabel('Variables',fontsize=15)
#ax2.legend(['x','y','z',r'$\alpha(t)$'],loc='lower right',fontsize=15,labelspacing=-0.1)