#%% PLOT POTENTIALS
width=2
height=1
bot=2
res=500
vertext=2
t=0
xveccy=np.linspace(-width,width,res)
yveccy=np.linspace(-width,width,res)
pause = False
def simData():
    t_max = 100.0
    dt = 0.5
    l1 = []
    l2 = []
    lt1 = []
    lt2 = []
    t = 0.0
    while t < t_max:
        if not pause:
            l1 = producepotentials(np.int(t/dt))[0]
            l2 = producepotentials(np.int(t/dt))[1]
            
            lt1 = xvec[0:np.int(t/dt)]
            lt2 = yvec[0:np.int(t/dt)]
            lt3 = fvec[0:np.int(t/dt)]
                       
            ev1 = ev1vec[0:np.int(t/dt)]
            ev2 = ev2vec[0:np.int(t/dt)]
                       
            tv = tvec[0:np.int(t/dt)]
            
            d1x = xvec[np.int(t/dt)]
            d1y = potentialx(t,xvec[np.int(t/dt)],yvec[np.int(t/dt)])
            d2x = yvec[np.int(t/dt)]
            d2y = potentialy(xvec[np.int(t/dt)],yvec[np.int(t/dt)])
                        
            if producepotentials(np.int(t/dt))[2]<np.int(res/2.)-1:
                dmx1 = (xveccy[producepotentials(np.int(t/dt))[2]],producepotentials(np.int(t/dt))[0][producepotentials(np.int(t/dt))[2]])
            else:
                dmx1 = ([],[])
                
            if producepotentials(np.int(t/dt))[3]>np.int(res/2.):
                dmx2 = (xveccy[producepotentials(np.int(t/dt))[3]],producepotentials(np.int(t/dt))[0][producepotentials(np.int(t/dt))[3]])
            else:
                dmx2 = ([],[])
                
            if producepotentials(np.int(t/dt))[4]<np.int(res/2.):
                dmy1 = (yveccy[producepotentials(np.int(t/dt))[4]],producepotentials(np.int(t/dt))[1][producepotentials(np.int(t/dt))[4]])
            else:
                dmy1 = ([],[])
                
            if producepotentials(np.int(t/dt))[5]>np.int(res/2.):
                dmy2 = (yveccy[producepotentials(np.int(t/dt))[5]],producepotentials(np.int(t/dt))[1][producepotentials(np.int(t/dt))[5]])
            else:
                dmy2 = ([],[])
            
            t = np.mod(t + dt,tmax)
        yield t, l1, l2, lt1, lt2, lt3, tv, d1x, d1y, d2x, d2y, dmx1, dmx2, dmy1, dmy2, ev1, ev2

def onClick(event):
    global pause
    pause ^= True

def simPoints(simData):
    t, l1, l2, lt1, lt2, lt3, tv = simData[0], simData[1], simData[2], simData[3], simData[4], simData[5], simData[6]
    d1x, d1y, d2x, d2y = simData[7], simData[8], simData[9], simData[10]
    dmx1, dmx2, dmy1, dmy2 = simData[11], simData[12], simData[13], simData[14]
    ev1r, ev2r = np.real(simData[15]), np.real(simData[16])
    
    line1.set_ydata(l1)
    line2.set_ydata(l2)
    
    line3.set_xdata(tv)
    line4.set_xdata(tv)
    line5.set_xdata(tv)
    line3.set_ydata(lt1)
    line4.set_ydata(lt2)
    line5.set_ydata(lt3)
    
    dot1.set_xdata(d1x)
    dot1.set_ydata(d1y)
    dot2.set_xdata(d2x)
    dot2.set_ydata(d2y)
    
    mindotx1.set_data(dmx1)
    mindotx2.set_data(dmx2)
    mindoty1.set_data(dmy1)
    mindoty2.set_data(dmy2)
    
    ev1vr.set_xdata(tv)
    ev1vr.set_ydata(ev1r)
    ev2vr.set_xdata(tv)
    ev2vr.set_ydata(ev2r)
    
    time_text.set_text('timesteps = %.0f' % t )
    return line1, line2, line3, line4, line5, time_text, dot1, dot2, mindotx1, mindotx2, mindoty1, mindoty2, ev1vr, ev2vr

# PLOT
fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,10))

# PLOT 1: POTENTIALS
line1, = ax1.plot(xveccy, producepotentials(0)[0], 'b',linewidth=3)
line2, = ax1.plot(yveccy, producepotentials(0)[1], 'r',linewidth=3)
ax1.set_xlim([-width,width])
ax1.set_ylim([-bot,height])
ax1.set_xlabel('x,y',fontsize=15)
ax1.set_ylabel('Potentials',fontsize=15)
ax1.tick_params(axis='both',which='major',labelsize=15)
ax1.legend(['Vx','Vy'],loc='upper left',fontsize=10)
time_text = ax1.text(0,1, '', zorder=10)

# PLOT 2: TIME EVOLUTION
line3, = ax2.plot(tvec[0:0], xvec[0:0], 'b',linewidth=3)
line4, = ax2.plot(tvec[0:0], yvec[0:0], 'r',linewidth=3)
line5, = ax2.plot(tvec[0:0], fvec[0:0], 'g',linewidth=1)
ax2.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'k--',linewidth=2)
ax2.plot([30,30],[-1000,1000],'k--',linewidth=2)
ax2.set_ylim([-vertext,vertext])
ax2.set_xlim([1,tmax])
ax2.tick_params(axis='both',which='major',labelsize=15)
ax2.set_xlabel('Time',fontsize=15)
ax2.set_ylabel('Variables',fontsize=15)
ax2.legend(['x','y',r'$\alpha(t)$'],loc='upper left',fontsize=10)

# PLOT 3: EIGENVALUES
minimum=min(np.min(ev1vec),np.min(ev2vec))
maximum=max(np.max(ev1vec),np.max(ev2vec))
ev1vr,=ax3.plot(tvec,np.real(ev1vec),'g',linewidth=3)
ev2vr,=ax3.plot(tvec,np.real(ev2vec),'purple',linewidth=3)
ev1i=ax3.plot(tvec,np.imag(ev1vec),'g--',linewidth=3)
ev2i=ax3.plot(tvec,np.imag(ev2vec),'--',color='purple',linewidth=3)
ax3.plot([0.01, 0.1,1,10,100,tmax],[0, 0,0,0,0,0],'k--',linewidth=2)
ax3.plot([30,30],[-1000,1000],'k--',linewidth=2)
ax3.fill_between(tvec, np.zeros(len(ev1vec)), ev1vec, where=ev1vec >= np.zeros(len(ev1vec)), facecolor='LightGreen', interpolate=True)
ax3.fill_between(tvec, np.zeros(len(ev1vec)), ev2vec, where=ev2vec >= np.zeros(len(ev1vec)), facecolor='plum', interpolate=True)
ax3.set_ylim([-5,3])
ax3.set_xlim([1,tmax])
ax3.tick_params(axis='both',which='major',labelsize=15)
ax3.set_xlabel('Time',fontsize=15)
ax3.set_ylabel('Eigenvalues',fontsize=15)
ax3.legend([r'$Re(\lambda_1)$',r'$Re(\lambda_2)$',r'$Im(\lambda_1)$',r'$Im(\lambda_2)$'],loc='lower right',fontsize=10)
fig.tight_layout()

# DOTS : CURRENT POINTS
dot1,=ax1.plot([xvec[0]],[potentialx(0*dt,xvec[0],yvec[0])],'o',color='LightBlue',zorder=50)
dot2,=ax1.plot([yvec[0]],[potentialy(xvec[0],yvec[0])],'o',color='Orange',zorder=50)

# DOTS : MINIMA/MAXIMA
if producepotentials(0)[2]<np.int(res/2.):
    mindotx1,= ax1.plot([xveccy[producepotentials(0)[2]]],[producepotentials(0)[0][producepotentials(0)[2]]],'bo',zorder=10,markersize=13)
else:
    mindotx1,=ax1.plot([],[],'bo',zorder=10,markersize=13)
if producepotentials(0)[4]<np.int(res/2.):
    mindoty1,= ax1.plot([yveccy[producepotentials(0)[4]]],[producepotentials(0)[1][producepotentials(0)[4]]],'ro',zorder=10,markersize=13)
else:
    mindoty1,=ax1.plot([],[],'ro',zorder=10,markersize=13)
if producepotentials(0)[3]>np.int(res/2.)+1:
    mindotx2,= ax1.plot([xveccy[producepotentials(0)[3]]],[producepotentials(0)[0][producepotentials(0)[3]]],'bo',zorder=10,markersize=13)
else:
    mindotx2,=ax1.plot([],[],'bo',zorder=10,markersize=13)
if producepotentials(0)[5]>np.int(res/2.)+1:
    mindoty2,= ax1.plot([yveccy[producepotentials(0)[5]]],[producepotentials(0)[1][producepotentials(0)[5]]],'ro',zorder=10,markersize=13)
else:
    mindoty2,=ax1.plot([],[],'ro',zorder=10,markersize=13)



# ANIMATION STUFF
fig.canvas.mpl_connect('button_press_event', onClick)
ani = animation.FuncAnimation(fig, simPoints, simData, blit=False, interval=25,
                              save_count=500)#,repeat=True)
if savy==True:
    ani.save('Movie1.mp4', writer="ffmpeg")
fig.show()

