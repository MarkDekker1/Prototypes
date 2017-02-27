# Preambule
import numpy as np
import matplotlib.pyplot as plt

# Integration parameters
tmax=100.
dt=0.1
tvec=np.linspace(0,tmax,tmax/dt+dt)

# Physical parameters
alpha=0.1
beta=1
gamma=0.1

c1=-0.5
c2=0.9

x0=-0.8
y0=-1.5

# Versions forcing
def alpha(t):
    return min(0.1+0.01*t,0.40)

# Variations of coupling

# General equations
def dx(t,x):
    return alpha(t)+beta*x-x**3.

def dy(x,y):
    return gamma+(c1+c2*x) * y - y**3./3.
    
def potentialx(t,x,y):
    return -alpha(t)*x - 0.5* beta*x**2.+0.25*x**4.
    
def potentialy(x,y):
    return -gamma*y - 0.5*  (c1+c2*x)*y**2.+0.25*y**4.
        
# Forward loop
xvecs=[]
yvecs=[]
dxvecs=[]
dyvecs=[]

xvec=[x0]
dxvec=[dx(0,x0)]
tvec=[0]
yvec=[y0]
dyvec=[dy(x0,y0)]
i=0
while np.abs(yvec[len(yvec)-1])<1000 and i<tmax/dt:
    t=i*dt
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
    tvec.append(t)
    i=i+1
    
# Plot
vertext=2
plt.plot(tvec,xvec,'b',linewidth=3)
plt.plot(tvec,yvec,'r',linewidth=3)
plt.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'k--',linewidth=2)
plt.ylim([-vertext,vertext])
plt.xlim([1,tmax])
plt.tick_params(axis='both',which='major',labelsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('x,y',fontsize=15)
plt.show()

#%% PLOT POTENTIALS
from matplotlib import animation

#tuse=40

width=2
height=1
bot=1
res=1000
xveccy=np.linspace(-width,width,res)
yveccy=np.linspace(-width,width,res)


fig=plt.figure()
line1=plt.plot(xveccy,producepotentials(0)[0],'b',linewidth=3)
time_text = plt.text(0,1, '', zorder=10)

def producepotentials(tuse):
    xpot=[]
    ypot=[]
    for xk in xveccy:
        xpot.append(potentialx(tuse*dt,xk,yvec[tuse]))
    for yk in yveccy:
        ypot.append(potentialy(xvec[tuse],yk))
    return xpot,ypot

def init():
    b=producepotentials(0)[0]
    line1.set_ydata(b)
    time_text.set_text('time = 0:00')
    return line1,time_text

def animate(self):
    global t
    global data
    t+=1
    b=producepotentials(t)[0]
    line1.set_ydata(b)
    time_text.set_text('time = %.0f:00' % t )
    return line1,time_text

anim = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,interval=50,blit=True)
    
#%%
ymin1=np.where(ypot==np.min(ypot[0:np.int(res/2.)]))[0][0]
xmin1=np.where(xpot==np.min(xpot[0:np.int(res/2.)]))[0][0]
ymin2=np.where(ypot==np.min(ypot[np.int(res/2.):]))[0][0]
xmin2=np.where(xpot==np.min(xpot[np.int(res/2.):]))[0][0]

plt.figure(figsize=(6,3))
plt.plot(xveccy,np.array(xpot),'b',linewidth=3)
if xmin1!=np.int(res/2.)-1:
    plt.scatter([xveccy[xmin1]],[xpot[xmin1]],c='b',zorder=10,s=200)
plt.scatter([xveccy[xmin2]],[xpot[xmin2]],c='b',zorder=10,s=200)
plt.scatter([xvec[tuse]],[potentialx(tuse*dt,xvec[tuse],yvec[tuse])],c='LightBlue',zorder=10,s=50)
plt.plot(yveccy,np.array(ypot),'r',linewidth=3)
if ymin1!=np.int(res/2.)-1:
    plt.scatter([yveccy[ymin1]],[ypot[ymin1]],c='r',zorder=10,s=200)
plt.scatter([yveccy[ymin2]],[ypot[ymin2]],c='r',zorder=10,s=200)
plt.scatter([yvec[tuse]],[potentialy(xvec[tuse],yvec[tuse])],c='Orange',zorder=10,s=50)
plt.xlabel('x,y',fontsize=15)
plt.ylabel('Vx,Vy',fontsize=15)
plt.tick_params(axis='both',which='major',labelsize=15)
plt.xlim([-width,width])
plt.ylim([-bot,height])

#%%
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