
#%% Losse plots
t1=1000
t2=np.where(yvec==np.min(yvec[3000:]))[0][0]
t3=np.where(yvec==np.max(yvec[3000:]))[0][0]
t4=np.where(xvec==np.min(xvec[3000:]))[0][0]
t5=np.where(xvec==np.max(xvec[3000:]))[0][0]
hx=-1.5
hy=-1.7
hz=-1.9
headl=0.2
headw=0.1

width=3.5
height=1
bot=2

#% PLOT 1: POTENTIALS
fig, ax1=plt.subplots(figsize=(6,5))
line1, = ax1.plot(xveccy, producepotentials(t5)[0], 'b',linewidth=3)
line2, = ax1.plot(yveccy, producepotentials(t2)[1], 'r',linewidth=3)
line6, = ax1.plot(zveccy, producepotentials(t2)[6], 'g',linewidth=3)
line1, = ax1.plot(xveccy, producepotentials(t4)[0], 'b--',linewidth=3)
line2, = ax1.plot(yveccy, producepotentials(t3)[1], 'r--',linewidth=3)
line6, = ax1.plot(zveccy, producepotentials(t1)[6], 'g--',linewidth=3)
ax1.set_xlim([-width,width])
ax1.set_ylim([-bot,height])
ax1.set_xlabel('Variable x, y, z',fontsize=15)
ax1.set_ylabel('Potentials',fontsize=15)
ax1.tick_params(axis='both',which='major',labelsize=15)
ax1.legend(['Vx','Vy','Vz'],loc='upper left',fontsize=15,labelspacing=-0.1)
#time_text = ax1.text(0,1, '', zorder=10)

ax1.plot([xvec[t4],xvec[t4]],[-1000,potentialx(t4*dt,xvec[t4],yvec[t4],zvec[t4])],'-',color='b',zorder=50,markersize=15)
ax1.plot([yvec[t2],yvec[t2]],[-1000,potentialy(t2*dt,xvec[t2],yvec[t2],zvec[t2])],'-',color='r',zorder=50,markersize=15)
ax1.plot([zvec[t1],zvec[t1]],[-1000,potentialz(xvec[t1],yvec[t1],zvec[t1])],'-',color='g',zorder=50,markersize=15)

ax1.plot([xvec[t5],xvec[t5]],[-1000,potentialx(t5*dt,xvec[t5],yvec[t5],zvec[t5])],'-',color='b',zorder=50,markersize=15)
ax1.plot([yvec[t3],yvec[t3]],[-1000,potentialy(t3*dt,xvec[t3],yvec[t3],zvec[t3])],'-',color='r',zorder=50,markersize=15)
ax1.plot([zvec[t2],zvec[t2]],[-1000,potentialz(xvec[t2],yvec[t2],zvec[t2])],'-',color='g',zorder=50,markersize=15)

ax1.plot([xvec[t4]],[potentialx(t4*dt,xvec[t4],yvec[t4],zvec[t4])],'o',color='b',zorder=50,markersize=9)
ax1.plot([yvec[t2]],[potentialy(t2*dt,xvec[t2],yvec[t2],zvec[t2])],'o',color='r',zorder=50,markersize=9)
ax1.plot([zvec[t1]],[potentialz(xvec[t1],yvec[t1],zvec[t1])],'o',color='g',zorder=50,markersize=9)

ax1.plot([xvec[t5]],[potentialx(t5*dt,xvec[t5],yvec[t5],zvec[t5])],'o',color='b',zorder=50,markersize=9)
ax1.plot([yvec[t3]],[potentialy(t3*dt,xvec[t3],yvec[t3],zvec[t3])],'o',color='r',zorder=50,markersize=9)
ax1.plot([zvec[t2]],[potentialz(xvec[t2],yvec[t2],zvec[t2])],'o',color='g',zorder=50,markersize=9)

ax1.arrow(xvec[t4],hx ,xvec[t5]-xvec[t4]-headl, 0, head_width=headw, head_length=headl, fc='b', ec='b',zorder=51)
ax1.arrow(yvec[t3],hy ,yvec[t2]-yvec[t3]+headl, 0, head_width=headw, head_length=headl, fc='r', ec='r',zorder=51)
ax1.arrow(xvec[t5],hx ,xvec[t4]-xvec[t5]+headl, 0, head_width=headw, head_length=headl, fc='b', ec='b',zorder=51)
ax1.arrow(yvec[t2],hy ,yvec[t3]-yvec[t2]-headl, 0, head_width=headw, head_length=headl, fc='r', ec='r',zorder=51)
ax1.arrow(zvec[t1],hz ,zvec[t2]-zvec[t1]-headl, 0, head_width=headw, head_length=headl, fc='g', ec='g',zorder=51)
ax1.plot([zvec[t1]],[hz],'s',color='g',zorder=50,markersize=5)

#%% PLOT 2: TIME EVOLUTION
tuse=len(tvec)
tend=200
vertext=2
fig, ax2=plt.subplots(figsize=(6,5))
line3, = ax2.plot(tvec[0:tuse], xvec[0:tuse], 'b',linewidth=3)
line4, = ax2.plot(tvec[0:tuse], yvec[0:tuse], 'r',linewidth=3)
line7, = ax2.plot(tvec[0:tuse], zvec[0:tuse], 'g',linewidth=3)
line5, = ax2.plot(tvec[0:tuse], fvec[0:tuse], 'k',linewidth=3,zorder=1)
ax2.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'--',color='Grey',linewidth=2,zorder=-1)
ax2.plot([lag,lag],[-1000,1000],'--',color='Grey',linewidth=2,zorder=-1)
ax2.plot([lag+14,lag+14],[-1000,1000],'--',color='Grey',linewidth=2,zorder=-1)
ax2.set_ylim([-vertext,vertext])
ax2.set_xlim([1,tend])
ax2.tick_params(axis='both',which='major',labelsize=15)
ax2.set_xlabel('Time',fontsize=15)
ax2.set_ylabel('Variables',fontsize=15)
ax2.legend(['x','y','z',r'$\alpha(t)$'],loc='upper left',fontsize=15,labelspacing=-0.1)

#%% PLOT 3.5: Runningmean
Rmean=620
vertext=0.5
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

#%% PLOT 3: EIGENVALUES
fig, ax3=plt.subplots(figsize=(6,5))
minimum=min(np.min(ev1vec),np.min(ev2vec))
maximum=max(np.max(ev1vec),np.max(ev2vec))
ev1vr,=ax3.plot(tvec,np.real(ev1vec),'Brown',linewidth=3)
ev2vr,=ax3.plot(tvec,np.real(ev2vec),'purple',linewidth=3)
ev3vr,=ax3.plot(tvec,np.real(ev3vec),'darkgreen',linewidth=3)
ev1vi,=ax3.plot(tvec,np.imag(ev1vec),'--',color='Brown',linewidth=3)
ev2vi,=ax3.plot(tvec,np.imag(ev2vec),'--',color='purple',linewidth=3)
ev3vi,=ax3.plot(tvec,np.imag(ev3vec),'--',color='darkgreen',linewidth=3)
ax3.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'--',color='Grey',linewidth=2,zorder=-1)
ax3.plot([20,20],[-1000,1000],'--',color='Grey',linewidth=2,zorder=-1)
Ev1=np.real(ev1vec)
Ev2=np.real(ev2vec)
Ev3=np.real(ev3vec)
ax3.fill_between(tvec, np.zeros(len(Ev1)), Ev1, where=Ev1 >= np.zeros(len(ev1vec)), facecolor='tan', interpolate=True,alpha=0.5)
ax3.fill_between(tvec, np.zeros(len(Ev1)), Ev2, where=Ev2 >= np.zeros(len(ev1vec)), facecolor='plum', interpolate=True,alpha=0.5)
ax3.fill_between(tvec, np.zeros(len(Ev1)), Ev3, where=Ev3 >= np.zeros(len(ev1vec)), facecolor='LightGreen', interpolate=True,alpha=0.5)
ax3.set_ylim([-23,2])
ax3.set_xlim([1,tend])
ax3.tick_params(axis='both',which='major',labelsize=15)
ax3.set_xlabel('Time',fontsize=15)
ax3.set_ylabel('Eigenvalues',fontsize=15)
ax3.legend([r'$Re(\lambda_1)$',r'$Re(\lambda_2)$',r'$Re(\lambda_3)$',r'$Im(\lambda_1)$',r'$Im(\lambda_2)$',r'$Im(\lambda_3)$'],loc='lower left',fontsize=15,labelspacing=-0.1)
fig.tight_layout()

# DOTS : CURRENT POINTS
#dot1,=ax1.plot([xvec[0]],[potentialx(xvec[0],yvec[0],zvec[0])],'o',color='LightBlue',zorder=50,markersize=15)
#dot2,=ax1.plot([yvec[0]],[potentialy(xvec[0],yvec[0],zvec[0])],'o',color='Orange',zorder=50,markersize=15)
#dot3,=ax1.plot([zvec[0]],[potentialz(0,xvec[0],yvec[0],zvec[0])],'o',color='LightGreen',zorder=50,markersize=15)