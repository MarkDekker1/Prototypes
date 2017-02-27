
#%% Losse plots
t1=3000
t2=14348
t3=17254
t4=14035
t5=16941
hx=-1.5
hy=-1.7
hz=-1.9
headl=0.2
headw=0.1

width=3.5
height=1
bot=2
vertext=3

#%% PLOT 1: POTENTIALS
ymax=1
ymin=-1
yperc=(ymax-ymin)/12.
width=2
headl=0.1
headw=0.05

E1=np.where(np.array(fvec)<np.sqrt(-4.*s1**3.*s2**3./27./(s1**4.)))[0][-30]
E1end=len(tvec)-1

#E2=np.where(np.array(bvec)<0)[0][-1]
#E2end=len(tvec)-1

E2=np.where(np.array(xvec[1500:])==np.max(xvec[1500:]))[0][0]+1500
E2end=np.where(np.array(xvec[1500:])==np.min(xvec[1500:]))[0][0]+1500

E3=np.where(np.array(yvec[1500:])==np.max(yvec[1500:]))[0][0]+1500
E3end=np.where(np.array(yvec[1500:])==np.min(yvec[1500:]))[0][0]+1500

hx=-2.4
hy=-2.7
fig, ax1=plt.subplots(figsize=(8,4))
line1, = ax1.plot(xveccy, producepotentials(E2end)[0], 'r',linewidth=4,zorder=5)
line2, = ax1.plot(yveccy, producepotentials(E3end)[1], 'g',linewidth=4)
line2, = ax1.plot(zveccy, producepotentials(E1end)[6], 'b',linewidth=4)
line1, = ax1.plot(xveccy, producepotentials(E2)[0], 'r--',linewidth=4,zorder=-1)
line2, = ax1.plot(yveccy, producepotentials(E3)[1], 'g--',linewidth=4,zorder=-1)
line2, = ax1.plot(zveccy, producepotentials(E1)[6], 'b--',linewidth=4,zorder=-1)
ax1.set_xlim([-width,width])
ax1.set_ylim([ymin,ymax])
ax1.set_xlabel('Variable x, y, z',fontsize=15)
ax1.set_ylabel(r'Potentials',fontsize=15)
ax1.tick_params(axis='both',which='major',labelsize=15)
ax1.legend(['Vx','Vy','Vz'],loc='upper center',fontsize=15,labelspacing=-0.1)
#time_text = ax1.text(0,1, '', zorder=10)

ax1.plot([xvec[E2],xvec[E2]],[-1000,potentialx(xvec[E2],yvec[E2],zvec[E2])],'-',color='darkred',zorder=51,markersize=15)
ax1.plot([yvec[E3],yvec[E3]],[-1000,potentialy(xvec[E3],yvec[E3],zvec[E3])],'-',color='darkgreen',zorder=50,markersize=15)
ax1.plot([zvec[E1],zvec[E1]],[-1000,potentialz(E1*dt,xvec[E1],yvec[E1],zvec[E1])],'-',color='darkblue',zorder=50,markersize=15)

ax1.plot([xvec[E2end],xvec[E2end]],[-1000,potentialx(xvec[E2end],yvec[E2end],zvec[E2end])],'-',color='darkred',zorder=51,markersize=15)
ax1.plot([yvec[E3end],yvec[E3end]],[-1000,potentialy(xvec[E3end],yvec[E3end],zvec[E3end])],'-',color='darkgreen',zorder=50,markersize=15)
ax1.plot([zvec[E1end],zvec[E1end]],[-1000,potentialz(E1end*dt,xvec[E1end],yvec[E1end],zvec[E1end])],'-',color='darkblue',zorder=50,markersize=15)

ax1.plot([xvec[E2]],[potentialx(xvec[E2],yvec[E2],zvec[E2])],'o',color='r',markersize=9,zorder=52)
ax1.plot([yvec[E3]],[potentialy(xvec[E3],yvec[E3],zvec[E3])],'o',color='g',zorder=4,markersize=9)
ax1.plot([zvec[E1]],[potentialz(E1*dt,xvec[E1],yvec[E1],zvec[E1])],'o',color='b',zorder=50,markersize=9)

ax1.plot([xvec[E2end]],[potentialx(xvec[E2end],yvec[E2end],zvec[E2end])],'o',color='r',markersize=9,zorder=52)
ax1.plot([yvec[E3end]],[potentialy(xvec[E3end],yvec[E3end],zvec[E3end])],'o',color='g',zorder=4,markersize=9)
ax1.plot([zvec[E1end]],[potentialz(E1end*dt,xvec[E1end],yvec[E1end],zvec[E1end])],'o',color='b',zorder=50,markersize=9)

ax1.arrow(xvec[E2],ymin+0.4*yperc ,xvec[E2end]-xvec[E2]+headl, 0, head_width=headw, head_length=headl, fc='darkred', ec='darkred',zorder=5)
ax1.arrow(yvec[E3],ymin+0.2*yperc ,yvec[E3end]-yvec[E3]+headl, 0, head_width=headw, head_length=headl, fc='darkgreen', ec='darkgreen')
ax1.arrow(xvec[E2end],ymin+0.4*yperc ,-(xvec[E2end]-xvec[E2]+headl), 0, head_width=headw, head_length=headl, fc='darkred', ec='darkred',zorder=5)
ax1.arrow(yvec[E3end],ymin+0.2*yperc,-(yvec[E3end]-yvec[E3]+headl), 0, head_width=headw, head_length=headl, fc='darkgreen', ec='darkgreen')
ax1.arrow(zvec[E1],ymin+0.3*yperc+yperc ,zvec[E1end]-zvec[E1]-0.1, 0, head_width=headw, head_length=headl, fc='darkblue', ec='darkblue')

ax1.text((zvec[E1end]+zvec[E1])/2.-0.1,ymin+0.5*yperc+yperc,'T1',fontsize=15,color='darkblue',weight='bold')
ax1.text(-0.07,ymin+0.55*yperc,'T2',fontsize=15,color='darkred',weight='bold')

ax1.plot([zvec[E1]],[ymin+0.3*yperc+yperc],'s',color='b',zorder=50,markersize=5)
ax1.plot([0],[ymin+0.2*yperc ],'s',color='g',zorder=50,markersize=5)
ax1.plot([0],[ymin+0.4*yperc ],'s',color='r',zorder=50,markersize=5)


#%% PLOT 2: TIME EVOLUTION
ymax=2
ymin=-2.5
yperc=(ymax-ymin)/12.

E1=np.where(np.array(fvec)<np.sqrt(-4.*s1**3.*s2**3./27./(s1**4.)))[0][-1]
E2=np.where(np.array(bvec)<0)[0][-1]

t1=30
t2=np.int(tmax/dt)
headl=0.1
headw=0.1
fig, ax2=plt.subplots(figsize=(8,4))
line3, = ax2.plot(tvec[0:t2], xvec[0:t2], 'r',linewidth=5,zorder=10)
line4, = ax2.plot(tvec[0:t2], yvec[0:t2], 'g',linewidth=5,zorder=9)
line4, = ax2.plot(tvec[0:t2], zvec[0:t2], 'b',linewidth=5)
line5, = ax2.plot(tvec[0:t2], fvec[0:t2], '-',color='darkblue',linewidth=2,zorder=-0.5)
line6, = ax2.plot(tvec[0:t2], bvec[0:t2], '-',color='darkred',linewidth=2,zorder=-0.5)
ax2.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'--',color='k',linewidth=1,zorder=-1)

ax2.plot([tvec[E1]],[fvec[E1]],'o',color='darkblue',ms=9)
ax2.plot([tvec[E1],tvec[E1]],[-1000,fvec[E1]],'--',color='darkblue',linewidth=1,zorder=-1)
ax2.text(tvec[E1]+3,ymin+0.5*yperc+yperc,'T1',fontsize=15,color='darkblue',weight='bold')
ax2.plot([tvec[E2]],[bvec[E2]],'o',color='darkred',ms=9,zorder=11)
ax2.plot([tvec[E2],tvec[E2]],[-1000,bvec[E2]],'--',color='darkred',linewidth=1,zorder=-1)
ax2.text(tvec[E2]+3,ymin+0.5*yperc,'T2',fontsize=15,color='darkred',weight='bold')


ax2.set_ylim([ymin,ymax])
ax2.set_xlim([50,300])
ax2.tick_params(axis='both',which='major',labelsize=15)
ax2.set_xlabel('Time',fontsize=15)
ax2.set_ylabel('Variables',fontsize=15)
ax2.legend(['x','y','z',r'$\alpha(t)$',r'$\beta(z)$'],loc='best',fontsize=15,labelspacing=-0.1)

L1=np.zeros(len(tvec))+ymin+0.25*yperc+yperc+yperc
L2=np.zeros(len(tvec))+ymin+0.25*yperc+yperc
L3=np.zeros(len(tvec))+ymin+0.25*yperc

ax2.fill_between(tvec[:E1], L1[:E1], L2[:E1], where=L1[:E1] >= L2[:E1], facecolor='blue', interpolate=True,alpha=0.6,color='b')
ax2.fill_between(tvec[E1:], L1[E1:], L2[E1:], where=L1[E1:] >= L2[E1:], facecolor='b', interpolate=True,alpha=0.2,color='b')
ax2.fill_between(tvec[:E2], L2[:E2], L3[:E2], where=L2[:E2] >= L3[:E2], facecolor='red', interpolate=True,alpha=0.6,color='r')
ax2.fill_between(tvec[E2:], L2[E2:], L3[E2:], where=L2[E2:] >= L3[E2:], facecolor='red', interpolate=True,alpha=0.2,color='red')
#ax3.fill_between(tvec, np.zeros(len(tvec)), Ev1, where=Ev1 >= np.zeros(len(ev1vec)), facecolor='tan', interpolate=True,alpha=0.5)

#%% PLOT 3: EIGENVALUES
ymax=1.1
ymin=-1.8
yperc=(ymax-ymin)/12.

E1=np.where(np.array(fvec)<np.sqrt(-4.*s1**3.*s2**3./27./(s1**4.)))[0][-1]
E2=np.where(np.array(bvec)<0)[0][-1]

fig, ax3=plt.subplots(figsize=(8,4))
minimum=min(np.min(ev1vec),np.min(ev2vec))
maximum=max(np.max(ev1vec),np.max(ev2vec))

ev1vr,=ax3.plot(tvec,np.real(ev1vec),'r',linewidth=4,zorder=5)
ev2vr,=ax3.plot(tvec,np.real(ev2vec),'g',linewidth=4)
ev2vr,=ax3.plot(tvec,np.real(ev3vec),'b',linewidth=4)
ev1vi,=ax3.plot(tvec,np.imag(ev1vec),'--',color='r',linewidth=3,zorder=5)
ev2vi,=ax3.plot(tvec,np.imag(ev2vec),'--',color='g',linewidth=3)
ev2vi,=ax3.plot(tvec,np.imag(ev3vec),'--',color='b',linewidth=3)
ax3.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'--',color='Grey',linewidth=2,zorder=-1)
#ax3.plot([30,30],[-1000,1000],'--',color='Grey',linewidth=2,zorder=-1)
Ev1=np.real(ev1vec)
Ev2=np.real(ev2vec)
Ev3=np.real(ev3vec)
ax3.fill_between(tvec, np.zeros(len(Ev1)), Ev1, where=Ev1 >= np.zeros(len(ev1vec)), facecolor='r', interpolate=True,alpha=0.2)
ax3.fill_between(tvec, np.zeros(len(Ev1)), Ev2, where=Ev2 >= np.zeros(len(ev1vec)), facecolor='g', interpolate=True,alpha=0.2)
ax3.fill_between(tvec, np.zeros(len(Ev1)), Ev3, where=Ev3 >= np.zeros(len(ev1vec)), facecolor='b', interpolate=True,alpha=0.2)
ax3.set_ylim([ymin,ymax])
ax3.set_xlim([0,tmax])
ax3.tick_params(axis='both',which='major',labelsize=15)
ax3.set_xlabel('Time',fontsize=15)
ax3.set_ylabel('Eigenvalues',fontsize=15)
ax3.legend([r'$Re(\lambda_1)$',r'$Re(\lambda_2)$',r'$Re(\lambda_3)$',r'$Im(\lambda_1)$',r'$Im(\lambda_2)$',r'$Im(\lambda_3)$'],loc='upper right',fontsize=15,labelspacing=-0.1)
#fig.tight_layout()


L1=np.zeros(len(tvec))+ymin+0.25*yperc+yperc+yperc
L2=np.zeros(len(tvec))+ymin+0.25*yperc+yperc
L3=np.zeros(len(tvec))+ymin+0.25*yperc

#ax3.plot([tvec[E1]],[fvec[E1]],'o',color='darkblue',ms=9)
ax3.plot([tvec[E1],tvec[E1]],[-1000,0],'--',color='darkblue',linewidth=1,zorder=-1)
ax3.text(tvec[E1]+3,ymin+0.5*yperc+yperc,'T1',fontsize=15,color='darkblue',weight='bold',zorder=100)
#ax3.plot([tvec[E2]],[bvec[E2]],'o',color='darkred',ms=9)
ax3.plot([tvec[E2],tvec[E2]],[-1000,0],'--',color='darkred',linewidth=1,zorder=-1)
ax3.text(tvec[E2]+3,ymin+0.5*yperc,'T2',fontsize=15,color='darkred',weight='bold',zorder=100)

ax3.fill_between(tvec[:E1], L1[:E1], L2[:E1], where=L1[:E1] >= L2[:E1], facecolor='blue', interpolate=True,alpha=0.6,color='b',zorder=100)
ax3.fill_between(tvec[E1:], L1[E1:], L2[E1:], where=L1[E1:] >= L2[E1:], facecolor='b', interpolate=True,alpha=0.2,color='b',zorder=100)
ax3.fill_between(tvec[:E2], L2[:E2], L3[:E2], where=L2[:E2] >= L3[:E2], facecolor='red', interpolate=True,alpha=0.6,color='r',zorder=100)
ax3.fill_between(tvec[E2:], L2[E2:], L3[E2:], where=L2[E2:] >= L3[E2:], facecolor='red', interpolate=True,alpha=0.2,color='red',zorder=100)


#%% PLOT 4: AUTOCORRELATION
import statsmodels.tsa.stattools as stat
from pandas import *

ymax=1.01
ymin=0.77
yperc=(ymax-ymin)/12.

l=125
ACFvecx=[]
ACFvecy=[]
ACFvecz=[]
ACFvecx2=[]
ACFvecy2=[]
ACFvecz2=[]
ACFmatx=[]
ACFmaty=[]
ACFmatz=[]
xv=Series(xvec)
yv=Series(yvec)
zv=Series(zvec)
tv=[]
for i in range(0,np.int(1+np.int((tmax-l)/dt))):
    ACFvecx.append(stat.acf(xvec[i:i+l])[1])
    ACFvecy.append(stat.acf(yvec[i:i+l])[1])
    ACFvecz.append(stat.acf(zvec[i:i+l])[1])
    
    ACFvecx2.append(pandas.Series.autocorr(xv[i:i+l],lag=1))
    ACFvecy2.append(pandas.Series.autocorr(yv[i:i+l],lag=1))
    ACFvecz2.append(pandas.Series.autocorr(zv[i:i+l],lag=1))
    
    ACFmatx.append(stat.acf(xvec[i:i+l]))
    ACFmaty.append(stat.acf(yvec[i:i+l]))
    ACFmatz.append(stat.acf(zvec[i:i+l]))
    tv.append((i+l)*dt)

E1=np.where(np.array(fvec)<np.sqrt(-4.*s1**3.*s2**3./27./(s1**4.)))[0][-1]
E2=np.where(np.array(bvec)<0)[0][-1]

fig, ax2=plt.subplots(figsize=(8,4))
#line3, = ax2.plot(tvec[np.int(l/dt):], ACFvecx, 'b',linewidth=5)
#line4, = ax2.plot(tvec[np.int(l/dt):], ACFvecy, 'r',linewidth=5)
line3, = ax2.plot(tv, ACFvecx, 'r',linewidth=4)
line4, = ax2.plot(tv, ACFvecy, 'g',linewidth=4)
line4, = ax2.plot(tv, ACFvecz, 'b',linewidth=4)
#ax2.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'--',color='k',linewidth=1,zorder=-1)

#ax2.plot([tvec[E1],tvec[E1]],[-1000,1000],'--',color='darkblue',zorder=1,linewidth=2)
#ax2.plot([tvec[EE1],tvec[EE1]],[-1000,1000],color='darkblue',zorder=1,linewidth=2)
#ax2.plot([tvec[E2],tvec[E2]],[-1000,1000],'--',color='darkred', zorder=1,linewidth=2)
#ax2.plot([tvec[EE2],tvec[EE2]],[-1000,1000],'darkred',zorder=1,linewidth=2)

ax2.plot([tvec[EE10],tvec[EE10]],[-1000,1000],color='darkblue', zorder=1,linewidth=2)
ax2.plot([tvec[EE11],tvec[EE11]],[-1000,1000],'darkblue',zorder=1,linewidth=2)

ax2.plot([tvec[EE20],tvec[EE20]],[-1000,1000],color='darkred', zorder=1,linewidth=2)
ax2.plot([tvec[EE21],tvec[EE21]],[-1000,1000],'darkred',zorder=1,linewidth=2)

#ax2.plot([tvec[EE20],tvec[EE20]],[-1000,1000],color='darkred', zorder=1,linewidth=2)
#ax2.plot([tvec[EE21],tvec[EE21]],[-1000,1000],'darkred',zorder=1,linewidth=2)


ax2.set_ylim([ymin,ymax])
#ax2.set_xlim([tv[0],tv[len(tv)-1]])
ax2.set_xlim([80,250])
ax2.tick_params(axis='both',which='major',labelsize=15)
ax2.set_xlabel('Time',fontsize=15)
ax2.set_ylabel('Autocorrelation coefficient AR(1)',fontsize=15)
ax2.legend([r'AR$_x$(1)',r'AR$_y$(1)',r'AR$_z$(1)'],loc='lower right',fontsize=15,labelspacing=-0.1)
ax2.text(tvec[E2]+1,ymin+0.3*yperc,'T2',fontsize=15,color='darkred',weight='bold')
ax2.text(tvec[E1]+1,ymin+0.35*yperc+yperc,'T1',fontsize=15,color='darkblue',weight='bold')


L1=np.zeros(len(tvec))+ymin+0.25*yperc+yperc+yperc
L2=np.zeros(len(tvec))+ymin+0.25*yperc+yperc
L3=np.zeros(len(tvec))+ymin+0.25*yperc

L4=np.zeros(len(tvec))+1000
L5=np.zeros(len(tvec))-1000

ax2.fill_between(tvec[:E1], L1[:E1], L2[:E1], where=L1[:E1] >= L2[:E1], facecolor='blue', interpolate=True,alpha=0.6,color='b')
ax2.fill_between(tvec[E1:], L1[E1:], L2[E1:], where=L1[E1:] >= L2[E1:], facecolor='b', interpolate=True,alpha=0.2,color='b')
ax2.fill_between(tvec[:E2], L2[:E2], L3[:E2], where=L2[:E2] >= L3[:E2], facecolor='red', interpolate=True,alpha=0.6,color='r')
ax2.fill_between(tvec[E2:], L2[E2:], L3[E2:], where=L2[E2:] >= L3[E2:], facecolor='red', interpolate=True,alpha=0.2,color='red')

#ax2.fill_between(tvec[EE20:EE21], L4[EE20:EE21], L5[EE20:EE21], where=L4[EE20:EE21] >= L5[EE20:EE21], facecolor='red', interpolate=True,alpha=0.2,color='red')

ax2.fill_between(tvec[EE10:EE11], L4[EE10:EE11], L5[EE10:EE11], where=L4[EE10:EE11] >= L5[EE10:EE11], facecolor='blue', interpolate=True,alpha=0.2,color='red')
ax2.fill_between(tvec[EE20:EE21], L4[EE20:EE21], L5[EE20:EE21], where=L4[EE20:EE21] >= L5[EE20:EE21], facecolor='red', interpolate=True,alpha=0.2,color='red')

#%% PLOT 5: ACF contourfplot
xmin=100
xmax=250

fig, (ax2,ax3)=plt.subplots(2,1,figsize=(14,5),sharex=True)
a=ax2.contourf(tv,np.linspace(0,40,41),np.transpose(np.array(ACFmatz)),150,zorder=5,vmin=-1,vmax=1,cmap=plt.cm.coolwarm)
ax2.contour(tv,np.linspace(0,40,41),np.transpose(np.array(ACFmatz)),[-0.8,0,0.8],zorder=5,colors='k')
ax2.plot([tvec[E1],tvec[E1]],[-1000,1000],'k--',zorder=10,linewidth=5)
ax2.plot([tvec[EE3],tvec[EE3]],[-1000,1000],'k',zorder=10,linewidth=5)
ax2.text(tvec[E1]+3,1,'T1',fontsize=15,color='k',weight='bold',zorder=50)
#ax2.set_xlabel('Time',fontsize=15)
ax2.set_ylabel(r'Time lag ($\Delta t$)',fontsize=15)
ax2.tick_params(axis='both',which='major',labelsize=15)
ax2.set_ylim([0,40])
ax2.set_xlim([xmin,xmax])
#plt.colorbar(a)

#fig, ax2=plt.subplots(figsize=(12,2))
a=ax3.contourf(tv,np.linspace(0,40,41),np.transpose(np.array(ACFmatx)),150,zorder=5,vmin=-1,vmax=1,cmap=plt.cm.coolwarm)
ax3.contour(tv,np.linspace(0,40,41),np.transpose(np.array(ACFmatx)),[-0.8,0,0.8],zorder=5,colors='k')
ax3.plot([tvec[E2],tvec[E2]],[-1000,1000],'k--',zorder=10,linewidth=5)
ax3.plot([tvec[EE21],tvec[EE21]],[-1000,1000],'k',zorder=10,linewidth=5)
ax3.text(tvec[E2]+3,1,'T2',fontsize=15,color='k',weight='bold',zorder=50)
ax3.set_xlabel('Time',fontsize=15)
ax3.set_ylabel(r'Time lag ($\Delta t$)',fontsize=15)
ax3.tick_params(axis='both',which='major',labelsize=15)
ax3.set_ylim([0,40])
ax3.set_xlim([xmin,xmax])
#plt.colorbar(a)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
fig.colorbar(a, cax=cbar_ax, ticks=np.linspace(-1,1,11))