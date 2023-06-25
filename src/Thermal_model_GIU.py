####### IMPORT PACKAGES #######

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import ndimage
from ipywidgets import Layout, Button, Box, interact, interactive, fixed, interact_manual
import ipywidgets as wg
from IPython.display import display, clear_output
import Thermal_simulation_class
import plot_utils

####### DEFINE FUNCTION #######

def T_model(folder=''):
    """
    Function that models the diffusion of heat through a paint layer.

    Inputs
    ====================================================
    folder: a str
    Path folder where to save the figures. 
     
    Returns
    ====================================================
            
    """
    ###### IMPORT PACKAGES ######
    
    import Thermal_model
    
    
    ###### GENERAL PARAMETERS ######
    style = {"description_width": "initial"}
    
    
    ###### CREATE WIDGETS ######
  
    R = wg.FloatSlider(
        value=0.01, 
        min=0,
        max=10,
        step=0.01,
        description="Radius (m)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

    L = wg.FloatSlider(
        value=0.0004,
        min=0,
        max=1,
        step=0.0001,
        description="Height (m)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
       
    n_rows = wg.IntSlider(
        value = 80,
        min=1,
        max=1000,
        description="Rows nb",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

    n_cols = wg.IntSlider(
        value = 120,
        min=1,
        max=1000,
        description="Columns nb",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
    k = wg.FloatSlider(
        value=0.2,
        min=0.01,
        max=10,
        step=0.01,
        description="Therm. cond. (W/(m K))",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

    C_p = wg.IntSlider(
        value = 2500,
        min=1,
        max=5000,
        description="Sp. heat cap. (J/(kg K))",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
    rho = wg.FloatSlider(
        value=1500,
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
        value = 0.1,
        min=0,
        max=1,
        step=0.01,
        description="Reflectance spectrum",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
    power = wg.FloatSlider(
        value = 5,
        min=0,
        max=100,
        step=0.1,
        description="Power (mW)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

    bs = wg.FloatSlider(
        value = 5,
        min=0,
        max=20,
        step=0.1,
        description="Beam size (um)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
    
    output = wg.Output()
    
    list_widgets = [wg.VBox([R,L,n_rows,n_cols,k,C_p,rho,dt,T_inf,T_init,RS,power,bs])]
    
    accordion = wg.Accordion(children = list_widgets)
    accordion.set_title(0, 'Parameters')
    
    
    ###### THERMAL DATA ######

    

    
    ###### PLOT FUNCTION ######
    
    
    
    def plot_data(n_rows,n_cols,R,L,k,C_p,rho,h,dt,T_inf,T_init,RS,power,bs):
        T = Temp_class.Temp(n_rows,n_cols,R,L,k,C_p,rho,h,dt,T_inf,T_init,RS,power,bs)
        #(M,C) = Thermal_model.T_model(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, power.value)
                      
        
        T_mesh = T.T_mesh()
        
        
        # create the figure environment
        fig = plt.figure(figsize = (16,8), constrained_layout=False)
        gs = fig.add_gridspec(6, 6)
        ax1 = fig.add_subplot(gs[0:6, 0:3])
        ax2 = fig.add_subplot(gs[0:3, 3:5]) 
        ax3 = ax2.twinx()
        ax4 = fig.add_subplot(gs[3:6, 3:5])
        
        # general parameters        
        fig.patch.set_facecolor([0.85,0.85,0.85])
        fs = 18       

        # plot the data
        
        res1 = R/ n_cols
        res2 = L/ n_rows
        #im = ax1.imshow(T_mesh,extent=(0, res1 * T_mesh.shape[1], res2 * T_mesh.shape[0], 1))
        #im = ax1.imshow(T_mesh)
        #plt.colorbar(im, ax = ax1)
        im = plot_utils.imshow_bar(T_mesh, ax=ax1)
        
        x1 = np.linspace(0,L*1e6,len(T_mesh[:,0]))
        x2 = np.linspace(0,R*1e3,len(T_mesh[0,:]))
        ax2.plot(T_mesh[:,0],x1, ls='--', label='Depth T')
        ax3.plot(T_mesh[0,:],x2, label='Surface T')
        
        ax4.annotate(f'max T = {np.round(T_mesh.max(),3)}', (0.1, 0.9), fontsize = fs)
        ax4.annotate(f'min T = {np.round(T_mesh.min(),3)}', (0.1, 0.8), fontsize = fs)
        
        ax2.set_xlim(np.max(T_mesh[:,0]), np.min(T_mesh[:,0]))
        ax2.set_ylim(np.max(x1), np.min(x1))

        ax3.set_xlim(np.max(T_mesh[:,0]), np.min(T_mesh[:,0]))
        ax3.set_ylim(np.max(x2)/5, np.min(x2))


        # remove axis for the text box
        ax4.axes.get_xaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)
        
        # axis limits
        #ax1.set_xlim(0,10)
        #ax1.set_ylim(11, 1)
                
        # labels
        ax1.set_xlabel('$x$ (microns)',fontsize = fs)
        ax1.set_ylabel(' Depth $z$ (microns)',fontsize = fs)        
        
        ax2.set_xlabel('Temperature (C)', fontsize = fs)
        ax2.set_ylabel('Depth (microns)', fontsize = fs)
        ax3.set_ylabel('Width (mm)', fontsize = fs)
                
        ax2.legend(loc='upper right')
        ax3.legend(loc='lower left')
        plt.tight_layout()
        plt.show()
    

    ###### DISPLAY GIU ######
    
    def file_change_R(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, change.new, L.value, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, power.value,bs.value)
     

    def file_change_L(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, change.new, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, power.value,bs.value)
     

    def file_change_n_rows(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(change.new, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, power.value,bs.value)
     

    def file_change_n_cols(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, change.new, R.value, L.value, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, power.value,bs.value)
     

    def file_change_k(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, change.new, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, power.value,bs.value)
     

    def file_change_C_p(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, k.value, change.new, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, power.value,bs.value)
     

    def file_change_rho(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, change.new, h.value, dt.value, T_inf.value, T_init.value, RS.value, power.value,bs.value)

    def file_change_h(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, change.new, dt.value, T_inf.value, T_init.value, RS.value, power.value,bs.value)
     
    def file_change_dt(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, h.value, change.new, T_inf.value, T_init.value, RS.value, power.value,bs.value)
     
    def file_change_T_inf(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, h.value, dt.value, change.new, T_init.value, RS.value, power.value,bs.value)
     
    def file_change_RS(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, change.new, power.value,bs.value)
    
    def file_change_power(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, change.new, bs.value)
     
    def file_change_T_init(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, change.new, RS.value, power.value,bs.value)
 
    def file_change_bs(change):
        output.clear_output(wait = True)
     
        with output:
            plot_data(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, power.value,change.new)
         


    R.observe(file_change_R, names = 'value')   
    L.observe(file_change_L, names = 'value') 
    n_rows.observe(file_change_n_rows, names = 'value')   
    n_cols.observe(file_change_n_cols, names = 'value') 
    T_init.observe(file_change_T_init, names = 'value')
    k.observe(file_change_k, names = 'value')
    C_p.observe(file_change_C_p, names = 'value')
    rho.observe(file_change_rho, names = 'value')
    h.observe(file_change_h, names = 'value')
    RS.observe(file_change_RS, names = 'value') 
    T_inf.observe(file_change_T_inf, names = 'value')
    T_init.observe(file_change_T_init, names = 'value') 
    power.observe(file_change_power, names = 'value') 
    dt.observe(file_change_dt, names = 'value') 
    bs.observe(file_change_bs, names = 'value') 
    
    ###### DISPLAY GIU ######
    
    display(accordion)
    display(output)



































