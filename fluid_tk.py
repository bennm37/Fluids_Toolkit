import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flow_elements import FLOW_DICT

##GENERAL FLOWS
class Fluid():
    #TODO lots of repeated code - could somehow combine to add pathlines,streaklines,streamlines 
    # and vector field independantly and then animate ? 
    #TODO make pathlines work for multiple inital conditions x_0 = [[1,2],[3,4]]
    def __init__(self,flow,x_0= np.array([1,1]),t_0 =0):
        self.flow = flow
        self.r = x_0 
        self.t = t_0

    def update(self,dt):
        U,V = self.flow(self.r[0],self.r[1],self.t)
        self.r = self.r + dt*np.array([U,V])

    ##DATA GENERATION
    def find_pathlines(self,num_steps,dt):
        u,v = self.flow(self.r[0],self.r[1],self.t)
        r_data = np.zeros((num_steps+1,2))
        r_data[0,:] =self.r 
        for i in range(1,num_steps+1):
            self.update(dt)
            self.t += dt
            r_data[i] = self.r
        return r_data

    def find_streaklines(self,x_0,t,dt,t_0_range):
        ##TODO fix calling pathlines
        num_samples = 50
        t_0s = np.linspace(t_0_range[0],t_0_range[1],num_samples)
        streak_line = np.zeros((num_samples,2))
        for i,t_0 in enumerate(t_0s):
            self.r = x_0
            self.t = t_0
            num_steps = int((t-t_0)//dt)
            streak_line[i,:] = self.find_pathlines(num_steps,dt)[-1,:]
        return streak_line

    def find_locus(self,points,t_0,num_steps,dt):
        locus = np.zeros((points.shape[0],2))
        for i,point in enumerate(points):
            self.r = point
            self.t =t_0 
            locus[i,:] = self.find_pathlines(num_steps,dt)[-1,:]
        return locus 
    
    ##DISPLAYERS
    def plot_streamlines(self,xrange=(-5,5),yrange=(-5,5),t=0):
        x = np.linspace(xrange[0],xrange[1],50)
        y = np.linspace(yrange[0],yrange[1],50)
        X,Y = np.meshgrid(x,y)
        U,V = self.flow(X,Y,t)
        fig,ax = plt.subplots()
        sp = ax.streamplot(X,Y,U,V)
        return sp 
    
    def plot_pathlines(self,num_steps,dt):
        r_data = self.find_pathlines(num_steps,dt)
        fig,ax = plt.subplots()
        fig.set_size_inches(7,7)
        pl = ax.plot(r_data[:,0],r_data[:,1])
        return pl 

    def plot_streakline(self,x_0,t,dt,t_0_range):
        r_data = self.find_streaklines(x_0,t,dt,t_0_range)
        fig,ax = plt.subplots()
        fig.set_size_inches(7,7)
        pl = ax.plot(r_data[:,0],r_data[:,1])
        return pl 

    def animate_streamlines(self,T,dt,xrange=(-5,5),yrange=(-5,5)):
        x = np.linspace(xrange[0],xrange[1],50)
        y = np.linspace(yrange[0],yrange[1],50)
        X,Y = np.meshgrid(x,y)
        U,V = self.flow(X,Y,0)
        fig,ax = plt.subplots()
        sp = ax.streamplot(X,Y,U,V)
        def update(i):
            t = i*dt
            ax.clear()
            U,V = self.flow(X,Y,t)
            sp = ax.streamplot(X,Y,U,V)
        anim = animation.FuncAnimation(fig,update,int(np.round(T/dt,1)))
        return anim

    def animate_vector_field(self,T,dt,xrange=(-5,5),yrange=(-5,5)):
        x = np.linspace(xrange[0],xrange[1],25)
        y = np.linspace(yrange[0],yrange[1],25)
        X,Y = np.meshgrid(x,y)
        U,V = self.flow(X,Y,0)
        fig,ax = plt.subplots()
        q = ax.quiver(X,Y,U,V)
        def update(i):
            t = i*dt
            U,V = self.flow(X,Y,t)
            q.set_UVC(U,V)
        anim = animation.FuncAnimation(fig,update,int(np.round(T/dt,1)))
        return anim
    
    def animate_locus(self,points,t_0,num_steps,dt,vector_field=False,xrange=(-2,2),yrange=(-2,2)):
        locus = self.find_locus(points,t_0,0,dt)
        fig,ax = plt.subplots()
        ax.set(xlim=(-xrange[0],xrange[1]),ylim=(-yrange[0],yrange[1]))
        fig.set_size_inches(7,7)
        l = ax.plot(locus[:,0],locus[:,1])
        if vector_field:
            x = np.linspace(xrange[0],xrange[1],20)
            y = np.linspace(yrange[0],yrange[1],20)
            X,Y = np.meshgrid(x,y)
            U,V = self.flow(X,Y,0)
            q = ax.quiver(X,Y,U,V)
        def update(i):
            ax.clear()
            locus = self.find_locus(points,t_0,i,dt)
            ax.set(xlim=(-2,2),ylim=(-2,2))
            l = ax.plot(locus[:,0],locus[:,1])
            if vector_field:
                t = i*dt 
                U,V = self.flow(X,Y,t)
                q = ax.quiver(X,Y,U,V)
        anim = animation.FuncAnimation(fig,update,num_steps)
        return anim 

class Potential_Flow():
    """Currently supports point node,dipole and uniform flow."""
    def __init__(self):
        self.stream = lambda x: 0

    def phi(self):
        """Calculates velocity potential from stream function"""#
        ##TODO work this out
        self.potential = self.stream  

    def add_element(self,name,parameters):
        old_stream =self.stream
        self.stream = lambda x:old_stream(x)+FLOW_DICT[name](x,parameters)

    def display(self,xrange=(-5,5),yrange=(-5,5),stream = True,potential = False,num_contours = 20):
        fig,ax = plt.subplots()
        fig.set_size_inches(5,5)
        sample_points =100
        x = np.linspace(xrange[0],xrange[1],sample_points)
        y = np.linspace(yrange[0],yrange[1],sample_points)
        X,Y = np.meshgrid(x,y)
        if stream:
            s = ax.contour(X,Y,self.stream([X,Y]),levels=num_contours)
        if potential:
            p = ax.contour(X,Y,self.potential(X,Y),levels=num_contours)
        plt.show()


