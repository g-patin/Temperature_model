import os
from xmlrpc.server import DocCGIXMLRPCRequestHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from ipywidgets import Layout, Button, Box, interact, interactive, fixed, interact_manual
import ipywidgets as wg
from IPython.display import display, clear_output

import plot_utils
import Thermal_model
style = {"description_width": "initial"}


simulation_output = wg.Output()


R = wg.FloatSlider(
        value=0.0005, 
        min=0,
        max=1,
        step=0.001,
        description="Radius (m)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

L = wg.FloatSlider(
        value=0.0005,
        min=0,
        max=1,
        step=0.0001,
        description="Height (m)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
       
n_rows = wg.IntSlider(
        value = 60,
        min=1,
        max=1000,
        description="Rows nb",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

n_cols = wg.IntSlider(
        value = 60,
        min=1,
        max=1000,
        description="Columns nb",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
k = wg.FloatSlider(
        value=0.1,
        min=0.01,
        max=10,
        step=0.01,
        description="Therm. cond. (W/(m K))",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

C_p = wg.IntSlider(
        value = 2060,
        min=1,
        max=5000,
        description="Sp. heat cap. (J/(kg K))",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
rho = wg.FloatSlider(
        value=1000,
        min=0,
        max=5000,
        step=0.01,
        description="Density paint (kg/m^3)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

h = wg.FloatSlider(
        value=12,
        min=0,
        max=50,
        step=0.01,
        description="Heat transfer coeff. (W/(m^2 K))",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

dt = wg.FloatSlider(
        value=0.0001,
        min=0,
        max=1,        
        description="Time step (s)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
T_inf = wg.FloatSlider(
        value=297,
        min=0,
        max=500,
        step=0.01,
        description="Far field Temp.(K)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )    
    
T_init = wg.FloatSlider(
        value=297,
        min=0,
        max=500,
        step=0.01,
        description="Initial Temp. (K)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )   
    
RS = wg.FloatSlider(
        value = 0.05,
        min=0,
        max=1,
        step=0.01,
        description="Reflectance spectrum",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
power_density = wg.IntSlider(
        value = 5,
        min=0,
        max=100000,
        step=1000,
        description="Power density",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

bs = wg.FloatSlider(
        value = 300,
        min=0,
        max=2000,
        step=1,
        description="Beam size",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

run_button = wg.Button(description='Run simulation')



class simulate(object):
    
    #def __init__(self, n_rows,n_cols,R,L,k,C_p,rho,h,dt,T_inf,T_init,RS,power_density,bs):    
    def __init__(self): 
      
        self.n_rows = n_rows.value
        self.n_cols = n_cols.value 
        self.R = R.value
        self.L = L.value
        self.k = k.value
        self.C_p = C_p.value
        self.rho = rho.value
        self.h = h.value
        self.dt = dt.value
        self.T_inf = T_inf.value
        self.T_init = T_init.value
        self.RS = RS.value
        self.power_density = power_density.value
        self.bs = bs.value
        self.run_button = run_button
        self.run_button.on_click(self.button_pressed)  

       
        list_widgets = [wg.VBox([R,L,n_rows,n_cols,k,C_p,rho,h,dt,T_inf,T_init,RS,power_density,bs])]    
        accordion = wg.Accordion(children = list_widgets)
        accordion.set_title(0, 'Parameters')

        display(wg.VBox([accordion,run_button,simulation_output]))     


    def __repr__(self) -> str:
        return f'Cylindrically symmetric thermal model of transient temperature evolution in a substrate lit from above.'


    def params(self):
        
        simulate.update_params(self)
        parameters = ['n_rows','n_cols','R','L','k','C_p','rho','h','dt','T_inf','T_init','RS','power_density','bs']
        values = [self.n_rows,self.n_cols,self.R,self.L,self.k,self.C_p,self.rho,self.h,self.dt,self.T_inf,self.T_init,self.RS,self.power_density,self.bs]

        df_params = pd.DataFrame([parameters,values], index=['parameter','value']).T
        df_params = df_params.set_index('parameter')

        return df_params


    def matrices_MC(self):
        (M,C) = Thermal_model.T_model(self.n_rows,self.n_cols,self.R,self.L,self.k,self.C_p,self.rho,self.h,self.dt,self.T_inf,self.T_init,self.RS,self.power_density,self.bs)

        return M,C

    def beam_intensity(self):
        (M,C) = Thermal_model.T_model(self.n_rows,self.n_cols,self.R,self.L,self.k,self.C_p,self.rho,self.h,self.dt,self.T_inf,self.T_init,self.RS,self.power_density,self.bs)

        #return I


    def T_mesh(self):
        simulate.update_params(self)
        (M,C) = Thermal_model.T_model(self.n_rows,self.n_cols,self.R,self.L,self.k,self.C_p,self.rho,self.h,self.dt,self.T_inf,self.T_init,self.RS,self.power_density,self.bs)
        
        r = np.linspace(0, self.R, self.n_cols + 1)  # radius values at cell boundaries
        r_half = (r[1:] + r[:-1]) / 2                # radius values at cell centers
        z = np.linspace(0, self.L, self.n_rows + 1)  # height values at cell boundaries
        
        changes = []
        N = self.n_rows * self.n_cols
        T = self.T_init * np.ones(N)  # vector of temperatures

        
        for i in tqdm(range(2000)):
            deltaT = M @ T + C
            T += deltaT
            changes.append(deltaT.max())
        
        
        T_mesh = T.reshape(self.n_rows, self.n_cols) - 273.15
        
        return T_mesh   


    def button_pressed(self,*args): 
        
        with simulation_output:
            simulation_output.clear_output(wait=True)
            simulate.update_params(self)
            simulate.plot(self)
    
    
        display(simulation_output)


    def update_params(self):

        def file_change_R(change):
            self.R = change.new

        def file_change_L(change):
            self.L = change.new

        def file_change_n_rows(change):
            self.n_rows = change.new

        def file_change_n_cols(change):
            self.n_cols = change.new

        def file_change_T_init(change):
            self.T_init = change.new

        def file_change_k(change):
            self.k = change.new

        def file_change_C_p(change):
            self.C_p = change.new

        def file_change_rho(change):
            self.rho = change.new

        def file_change_h(change):
            self.h = change.new

        def file_change_RS(change):
            self.RS = change.new           

        def file_change_T_inf(change):
            self.T_inf = change.new

        def file_change_power_density(change):
            self.power_density = change.new

        def file_change_dt(change):
            self.dt = change.new

        def file_change_bs(change):
            self.bs = change.new           


        R.observe(file_change_R, names = 'value')          
        L.observe(file_change_L, names = 'value') 
        n_rows.observe(file_change_n_rows, names = 'value')   
        n_cols.observe(file_change_n_cols, names = 'value') 
        k.observe(file_change_k, names = 'value')
        C_p.observe(file_change_C_p, names = 'value')
        rho.observe(file_change_rho, names = 'value')
        h.observe(file_change_h, names = 'value')
        RS.observe(file_change_RS, names = 'value') 
        T_inf.observe(file_change_T_inf, names = 'value')
        T_init.observe(file_change_T_init, names = 'value') 
        power_density.observe(file_change_power_density, names = 'value') 
        dt.observe(file_change_dt, names = 'value') 
        bs.observe(file_change_bs, names = 'value') 
        

    def plot(self):

        with simulation_output:               
            
            T_mesh = simulate.T_mesh(self)
            #I = simulate.beam_intensity(self)
            
            # create the figure environment
            fig = plt.figure(figsize = (18,18), constrained_layout=False)
            gs = fig.add_gridspec(9, 9)
            ax1 = fig.add_subplot(gs[2:9, 2:6])  # center subplot - plot the mesh
            ax2 = fig.add_subplot(gs[3:7, 0:2])  # left subplot - plot temperature along the depth
            ax3 = fig.add_subplot(gs[0:2, 2:6])  # top centered subplot - plot temperature at the surface
            #ax4 = fig.add_subplot(gs[0:2, 0:2])
            ax5 = fig.add_subplot(gs[0:4, 6:9])  # top right subplot - contour plot           
            ax6 = fig.add_subplot(gs[4:9, 6:9])  # bottom right subplot - write values
            
            # general parameters        
            fig.patch.set_facecolor([0.85,0.85,0.85])
            fs = 18       

            # plot ax1 - the mesh   
            res1 = self.R/ self.n_cols * 1e6
            res2 = self.L/ self.n_rows * 1e6                   
            
            im = plot_utils.imshow_bar(T_mesh, ax=ax1, extent=(0, res1 * T_mesh.shape[1], res2 * T_mesh.shape[0], 1))
            #im = ax1.imshow(T_mesh, extent=(0, res1 * T_mesh.shape[1], res2 * T_mesh.shape[0], 1))
            #plt.colorbar(im, ax = ax1)  
            #cb = ax1.colorbar(im, ax=ax1, shrink=0.78, pad = 0.01)
            #fs = 20
            #

            # plot ax2 - T along the depth
            x1 = np.linspace(0,self.L*1e6,len(T_mesh[:,0]))
            ax2.plot(T_mesh[:,0],x1, ls='--')

            # plot ax3 - T at the surface
            x2 = np.linspace(0,self.R*1e6,len(T_mesh[0,:]))
            ax3.plot(x2,T_mesh[0,:])

            # plot ax4 - the fading beam
            #ax4.plot(I)

            # plot ax5 - contour plot
            r = np.linspace(0, self.R, self.n_cols + 1)  # radius values at cell boundaries
            r_half = (r[1:] + r[:-1]) / 2                # radius values at cell centers
            z = np.linspace(0, self.L, self.n_rows + 1)  # height values at cell boundaries

            r_plot = r[None].repeat(self.n_rows, axis=0)[:, :-1] * 1e6
            z_plot = z[:, None].repeat(self.n_cols, axis=1)[:-1, :] * 1e6

            

            cp = ax5.contourf(r_plot , z_plot, T_mesh, extent=(0, res1 * T_mesh.shape[1], res2 * T_mesh.shape[0], 1))
            ax5.clabel(cp, inline=True, fontsize=10, colors = 'black')                     

            #cp = ax1.contourf(r_plot, z_plot, T_mesh)          
            
            #ax2.plot(T_mesh[:,0],x1, ls='--', label='Depth T')  
            
            depth_T_values = np.round(T_mesh[:,0],2)
            depth_T_min = list(depth_T_values >= T_mesh.min() + 0.1).index(False) * res2
            ax6.annotate(f'max T = {np.round(T_mesh.max(),3)} C', (0.1, 0.9), fontsize = fs)
            ax6.annotate(f'min T = {np.round(T_mesh.min(),3)} C', (0.1, 0.8), fontsize = fs)
            #ax6.annotate(f'depth at which T = T_min +0.1 C is {depth_T_min} µm', (0.1, 0.7), fontsize = fs)
            
            ax2.set_xlim(np.max(T_mesh[:,0]), np.min(T_mesh[:,0]))
            ax2.set_ylim(np.max(x1), np.min(x1))

            ax3.set_xlim(0)
            #ax3.set_ylim(np.max(x2)/5, np.min(x2))


            # remove axis for the text box
            #ax6.axes.get_xaxis().set_visible(False)
            #ax6.axes.get_yaxis().set_visible(False)
            
            # axis limits
            #ax1.set_xlim(0,10)
            #ax1.set_ylim(np.max(x1), np.min(x1))

                    
            # plot the labels
            ax1.set_xlabel('Radial coordinate $r$ (microns)',fontsize = fs)
            ax1.set_ylabel('Depth $z$ (microns)',fontsize = fs)        
            
            ax2.set_xlabel('Temperature (C)', fontsize = fs)
            ax2.set_ylabel('Depth $z$ (microns)', fontsize = fs)

            ax3.set_xlabel('Radial coordinate $r$ (microns)', fontsize = fs)
            ax3.set_ylabel('Temperature (C)', fontsize = fs)

            ax5.set_xlabel('Radial coordinate $r$ (microns)', fontsize = fs)
            ax5.set_ylabel('Radial coordinate $r$ (microns)', fontsize = fs)


            ax1.xaxis.set_tick_params(labelsize=fs)
            ax1.yaxis.set_tick_params(labelsize=fs)

            ax2.xaxis.set_tick_params(labelsize=fs)
            ax2.yaxis.set_tick_params(labelsize=fs)

            ax3.xaxis.set_tick_params(labelsize=fs)
            ax3.yaxis.set_tick_params(labelsize=fs)

            ax5.xaxis.set_tick_params(labelsize=fs)
            ax5.yaxis.set_tick_params(labelsize=fs)

            ax1.spines['bottom'].set_visible(False)
            ax1.spines['top'].set_visible(True)
            ax2.spines['left'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['top'].set_visible(False)  

            ax1.xaxis.set_ticks_position('top')  
            ax2.xaxis.set_ticks_position('top')   
            ax2.yaxis.set_ticks_position('right')   

            ax1.xaxis.set_label_position('top') 
            ax2.xaxis.set_label_position('top') 
            ax2.yaxis.set_label_position('right')    
 
            plt.tight_layout()
            plt.show()

        
            
