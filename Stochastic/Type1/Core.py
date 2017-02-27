# Preambule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Integration parameters
tmax=300.
dt=0.5
tvec=np.linspace(0,tmax,tmax/dt+dt)
savy=False

# Physical parameters
p1=-1
p2=1
q1=-1
q2=1

c1=0.01
c2=0.08

x0=-0.8
y0=-1

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Versions forcing
def xi_x(t,x,y):
    mean= min(0.1+0.01*max(t-75,0),0.01+np.sqrt(-4.*p1**3.*p2**3./27./(p1**4.)))
    sigma=0.1
    return np.random.normal(mean,sigma)

# Variations of coupling
def xi_y(t,x,y):
    mean= 0.33
    sigma=max(c1+c2*x,c1)
    return np.random.normal(mean,sigma)
    
# General equations
def dx(t,x,y):
    global xix
    if np.mod(t,2)==0:
        xix=xi_x(t,x,y)
    return xix+p2*x+p1*x**3.

def dy(t,x,y):
    global xiy
    if np.mod(t,2)==0:
        xiy=xi_y(t,x,y)
    return xiy + q2*y +q1* y**3.
    
#def potentialx(t,x,y):
#    return -alpha(t)*x - 0.5* p2*x**2.-p1*0.25*x**4.
    
#def potentialy(x,y):
#    return -(c1+c2*x)*y - 0.5*  q2*y**2.-q1*0.25*y**4.
    
#def ev1(x,y):
#    return 3*p1*x**2.+p2

#def ev2(x,y):
#    return 3*q1*y**2.+q2
        
# Forward loop
tvec=[0]
yvec=[y0]
xvec=[x0]
dxvec=[dx(0,x0,y0)]
dyvec=[dy(0,x0,y0)]

i=0
while np.abs(yvec[len(yvec)-1])<1000 and i<tmax/dt:
    t=i*dt
    k1 = dt*dx(t,xvec[i],yvec[i])
    k2 = dt*dx(t+0.5*dt,xvec[i]+k1*0.5,yvec[i])
    k3 = dt*dx(t+0.5*dt,xvec[i]+k2*0.5,yvec[i])
    k4 = dt*dx(t+dt,xvec[i]+k3,yvec[i])
    xnew=xvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dxvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dy(t,xvec[i],yvec[i])
    k2 = dt*dy(t+0.5*dt,xvec[i],yvec[i]+0.5*k1)   
    k3 = dt*dy(t+0.5*dt,xvec[i],yvec[i]+0.5*k2)    
    k4 = dt*dy(t+dt,xvec[i],yvec[i]+k3) 
    ynew=yvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dyvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    xvec.append(xnew)
    yvec.append(ynew)
    tvec.append(t)
    i=i+1
    
#def producepotentials(tuse):
#    xpot=[]
#    ypot=[]
#    for xk in xveccy:
#        xpot.append(potentialx(tuse*dt,xk,yvec[tuse]))
#    for yk in yveccy:
#        ypot.append(potentialy(xvec[tuse],yk))
#    
#    ymin1=np.where(ypot==np.min(ypot[0:np.int(res/2.)]))[0][0]
#    xmin1=np.where(xpot==np.min(xpot[0:np.int(res/2.)]))[0][0]
#    ymin2=np.where(ypot==np.min(ypot[np.int(res/2.):]))[0][0]
#    xmin2=np.where(xpot==np.min(xpot[np.int(res/2.):]))[0][0]
#    return xpot,ypot,xmin1,xmin2,ymin1,ymin2

width=2
height=1
bot=2
res=500
vertext=3
t=0
xveccy=np.linspace(-width,width,res)
yveccy=np.linspace(-width,width,res)

dxv=[]
dyv=[]
for i in range(1,np.int(tmax/dt)-1):
    dxv.append((xvec[i+1]-xvec[i-1])/(2*dt))
    dyv.append((yvec[i+1]-yvec[i-1])/(2*dt))
    
#EE1=np.where(np.array(dxv)==np.max(dxv))[0][0]
#EE2=np.where(np.array(dyv)==np.max(dyv))[0][0]


#EE10=np.where(np.array(dxv)>0.2*np.max(dxv))[0][0]
#EE11=np.where(np.array(dxv)>0.2*np.max(dxv))[0][-1]
#EE20=np.where(np.array(dyv)>0.2*np.max(dyv))[0][0]
#EE21=np.where(np.array(dyv)>0.2*np.max(dyv))[0][-1]