# Preambule
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Integration parameters
tmax=500.
dt=0.2
tvec=np.linspace(0,tmax,tmax/dt+dt)
savy=False

# Physical parameters
p1=1
p2=1
#p3=1
q1=-1
q2=1
#q3=1
s1=-1
s2=1
#s3=1

c1=-0.1
c2=0.12

x0=1.
y0=-1.
z0=-0.5

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Versions forcing
def alpha(t):
    return min(-0.7+0.05*max(t-100,0),0.4)

# Variations of coupling
def beta(z):
    return c1+c2*z

# General equations
def dx(x,y,z):
    p3=beta(z)
    return p1*y+p2*(p3-(x**2.+y**2.))*x+0.001*z

def dy(x,y,z):
    q3=beta(z)
    return q1*x+q2*(q3-(x**2.+y**2.))*y
    
def dz(t,x,y,z):
    return s1*z**3.+s2*z+alpha(t)
    
def potentialx(x,y,z):
    p3=beta(z)
    return -x*y*p1-0.5*p2*p3*x**2+0.25*p2*x**4+0.5*p2*x**2*y**2
    
def potentialy(x,y,z):
    q3=beta(z)
    return -x*y*q1-0.5*q2*q3*y**2+0.25*q2*y**4+0.5*q2*x**2*y**2
    
def potentialz(t,x,y,z):
    return -0.25*s1*z**4-0.5*s2*z**2-z*alpha(t)
    
def ev1(x,y,z):
    return -0.5*(4*x**2+4*y**2-0.24*z+np.sqrt(0J+4*x**4+8*y**2*x**2+4*y**4-4)+0.2)
    
def ev2(x,y,z):
    return -0.5*(4*x**2+4*y**2-0.24*z-np.sqrt(0J+4*x**4+8*y**2*x**2+4*y**4-4)+0.2)

def ev3(x,y,z):
    return 1-3*z**2
    
    
def G(x,y,z):
    #p3=beta(z)
    #q3=beta(z)    
    return (-c1 *p2 - c1* q2 - c2 *p2* z - c2 *q2 *z + 3 *p2 *x**2 + p2* y**2 + q2 *x**2 + 3* q2 *y**2)**2 - 4 *(c1**2 *p2 *q2 + 2* c1* c2 *p2 *q2* z - 4 *c1* p2 *q2 *x**2 - 4 *c1 *p2* q2 *y**2 + c2**2 *p2 *q2* z**2 - 4* c2* p2 *q2 *x**2* z - 4* c2* p2 *q2* y**2* z - p1 *q1 + 2 *p1 *q2 *x* y + 2 *p2 *q1 *x* y + 3 *p2 *q2 *x**4 + 6* p2* q2* x**2* y**2 + 3* p2 *q2 *y**4)
    
def G2(x,y):
    return np.sqrt(x**4.+2.*x**2.*y**2.+y**4.-1.)
    
def H(x,y,z):
    return c1* p2 + c1 *q2 + c2 *p2 *z + c2 *q2*z - 3 *p2 *x**2 - p2* y**2 - q2 *x**2 - 3 *q2* y**2
        
# Forward loop
xvec=[x0]
yvec=[y0]
zvec=[z0]

dxvec=[dx(x0,y0,z0)]
dyvec=[dy(x0,y0,z0)]
dzvec=[dz(0,x0,y0,z0)]
       
tvec=[0]
bvec=[beta(x0)]
fvec=[alpha(0)]
gvec=[G(x0,y0,z0)]
gvec2=[G2(x0,y0)]
hvec=[H(x0,y0,z0)]
ev1vec=[ev1(x0,y0,z0)]
ev2vec=[ev2(x0,y0,z0)]
ev3vec=[ev3(x0,y0,z0)]
        
i=0
while np.abs(yvec[len(yvec)-1])<1000 and i<tmax/dt:
    t=i*dt
    
    k1 = dt*dz(t,xvec[i],yvec[i],zvec[i])
    k2 = dt*dz(t+0.5*dt,xvec[i],yvec[i],zvec[i]+k1*0.5)
    k3 = dt*dz(t+0.5*dt,xvec[i],yvec[i],zvec[i]+k2*0.5)
    k4 = dt*dz(t+dt,xvec[i],yvec[i],zvec[i]+k3)
    znew=zvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dzvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dx(xvec[i],yvec[i],zvec[i])
    k2 = dt*dx(xvec[i]+k1*0.5,yvec[i],zvec[i])
    k3 = dt*dx(xvec[i]+k2*0.5,yvec[i],zvec[i])
    k4 = dt*dx(xvec[i]+k3,yvec[i],zvec[i])
    xnew=xvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dxvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    k1 = dt*dy(xvec[i],yvec[i],zvec[i])
    k2 = dt*dy(xvec[i],yvec[i]+0.5*k1,zvec[i])   
    k3 = dt*dy(xvec[i],yvec[i]+0.5*k2,zvec[i])    
    k4 = dt*dy(xvec[i],yvec[i]+k3,zvec[i]) 
    ynew=yvec[i]+(k1+2.*k2+2.*k3+k4)/6.
    dyvec.append((k1+2.*k2+2.*k3+k4)/6./dt)
    
    gvec.append(G(xnew,ynew,znew)) 
    gvec2.append(G2(xnew,ynew))    
    xvec.append(xnew)
    yvec.append(ynew)
    zvec.append(znew)
    hvec.append(H(xnew,ynew,znew))
    
    fvec.append(alpha(t))
    bvec.append(beta(znew))
    
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
        xpot.append(potentialx(xk,yvec[tuse],zvec[tuse]))
    for yk in yveccy:
        ypot.append(potentialy(xvec[tuse],yk,zvec[tuse]))
    for zk in zveccy:
        zpot.append(potentialz(tuse*dt,xvec[tuse],yvec[tuse],zk))
    
    ymin1=np.where(ypot==np.min(ypot[0:np.int(res/2.)]))[0][0]
    xmin1=np.where(xpot==np.min(xpot[0:np.int(res/2.)]))[0][0]
    
    ymin2=np.where(ypot==np.min(ypot[np.int(res/2.):]))[0][0]
    xmin2=np.where(xpot==np.min(xpot[np.int(res/2.):]))[0][0]
    
    zmin1=np.where(zpot==np.min(zpot[0:np.int(res/2.)]))[0][0]
    zmin2=np.where(zpot==np.min(zpot[np.int(res/2.):]))[0][0]
    return xpot,ypot,xmin1,xmin2,ymin1,ymin2,zpot,zmin1,zmin2

    
width=5
height=5
bot=5
res=500
vertext=2
t=0
xveccy=np.linspace(-width,width,res)
yveccy=np.linspace(-width,width,res)
zveccy=np.linspace(-width,width,res)


dxv=[]
dyv=[]
dzv=[]
for i in range(1,np.int(tmax/dt)-1):
    dxv.append((xvec[i+1]-xvec[i-1])/(2*dt))
    dyv.append((yvec[i+1]-yvec[i-1])/(2*dt))
    dzv.append((zvec[i+1]-zvec[i-1])/(2*dt))
    
EE1=np.where(np.array(dxv)==np.max(dxv))[0][0]+1
EE2=np.where(np.array(dyv)==np.max(dyv))[0][0]+1
EE3=np.where(np.array(dzv)==np.max(dzv))[0][0]+1


EE10=np.where(np.array(dzv)>0.2*np.max(dzv))[0][0]+1
EE11=np.where(np.array(dzv)>0.2*np.max(dzv))[0][-1]+1

EE20=np.where(np.array(dxv[600:])>0.2*np.max(dxv[600:]))[0][0]+600
EE21=np.where(np.array(dxv[600:])>0.95*np.max(dxv[600:]))[0][0]+600

##
#tuse=np.int(tmax/dt)
#tend=tmax
#fig, ax2=plt.subplots(figsize=(6,5))
#line3, = ax2.plot(tvec[0:tuse], xvec[0:tuse], 'b',linewidth=3)
#line4, = ax2.plot(tvec[0:tuse], yvec[0:tuse], 'r',linewidth=3)
#line7, = ax2.plot(tvec[0:tuse], zvec[0:tuse], 'g',linewidth=3)
#line7, = ax2.plot(tvec[0:tuse], gvec[0:tuse], 'y',linewidth=3)
#line7, = ax2.plot(tvec[0:tuse], hvec[0:tuse], 'm',linewidth=3)
#line5, = ax2.plot(tvec[0:tuse], fvec[0:tuse], 'k',linewidth=3,zorder=-1)
#line7, = ax2.plot(tvec[0:tuse], bvec[0:tuse], 'grey',linewidth=3)
#ax2.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'--',color='Grey',linewidth=2,zorder=-1)
#ax2.plot([20,20],[-1000,1000],'--',color='Grey',linewidth=2,zorder=-1)
#ax2.set_ylim([-5,5])
#ax2.set_xlim([1,tend])
#ax2.tick_params(axis='both',which='major',labelsize=15)
#ax2.set_xlabel('Time',fontsize=15)
#ax2.set_ylabel('Variables',fontsize=15)
#ax2.legend(['x','y','z','G','H',r'$\alpha(t)$', r'$\beta(z)$'],loc='best',fontsize=15,labelspacing=-0.1)