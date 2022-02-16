####### IMPORT PACKAGES #######

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import ndimage
from ipywidgets import Layout, Button, Box, interact, interactive, fixed, interact_manual
import ipywidgets as wg
from IPython.display import display, clear_output


####### DEFINE FUNCTION #######

def T_model(folder):
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
  
    R = wg.IntSlider(
        value=23, 
        min=1,
        max=10000,
        description="Radius (µm)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

    L = wg.IntSlider(
        value=23,
        min=1,
        max=100000,
        description="Height (µm)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
       
    n_rows = wg.IntSlider(
        value = 100,
        min=1,
        max=1000,
        description="Rows nb",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )

    n_cols = wg.IntSlider(
        value = 200,
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
        description="Sp. heat cap. (W/(m^2 K))",
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
        value=0.1,
        min=0,
        max=1,        
        description="Time step (s)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
    T_inf = wg.FloatSlider(
        value=300,
        min=0,
        max=500,
        step=0.01,
        description="Far field Temp.(K)",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )    
    
    T_init = wg.FloatSlider(
        value=300,
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
    
    power_density = wg.IntSlider(
        value = 40000,
        min=0,
        max=100000,
        step=1000,
        description="Power density",
        layout=Layout(width="50%", height="30px"),
        style=style,
    )
    
    
    output = wg.Output()
    
    list_widgets = [wg.VBox([R,L,n_rows,n_cols,k,C_p,rho,dt,T_inf,T_init,RS,power_density])]
    
    accordion = wg.Accordion(children = list_widgets)
    accordion.set_title(0, 'Parameters')
    
    
    ###### THERMAL DATA ######

    

    
    ###### PLOT FUNCTION ######
    
    
    
    def plot_data():
        
        (M,C) = Thermal_model.T_model(n_rows.value, n_cols.value, R.value, L.value, k.value, C_p.value, rho.value, h.value, dt.value, T_inf.value, T_init.value, RS.value, power_density.value)
        
        changes = []
        N = n_rows.value * n_cols.value
        T = T_init.value * np.ones(N)  # vector of temperatures

        
        for i in tqdm(range(500)):
            deltaT = M @ T + C
            T += deltaT
            changes.append(deltaT.max())
        
        
        T_mesh = T.reshape(n_rows.value, n_cols.value) - 273.15
        
        
        # create the figure environment
        fig = plt.figure(figsize = (16,8), constrained_layout=False)
        gs = fig.add_gridspec(6, 6)
        ax1 = fig.add_subplot(gs[0:6, 0:3])
        ax2 = fig.add_subplot(gs[0:3, 3:5]) 
        ax3 = fig.add_subplot(gs[3:6, 3:5])
        
        # general parameters        
        fig.patch.set_facecolor([0.85,0.85,0.85])
        fs = 18       

        # plot the data
        
        res1 = R.value/ n_cols.value
        res2 = L.value/ n_rows.value
        im = ax1.imshow(T_mesh,extent=(0, res1 * T_mesh.shape[1], res2 * T_mesh.shape[0], 1))
        #im = ax1.imshow(T_mesh)
        plt.colorbar(im, ax = ax1)
        
        ax3.annotate(f'max T = {np.round(T_mesh.max(),3)}', (0.1, 0.9), fontsize = fs)
        ax3.annotate(f'min T = {np.round(T_mesh.min(),3)}', (0.1, 0.8), fontsize = fs)
        
        
        # remove axis for the text box
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)
        
        # axis limits
        ax1.set_xlim(0,10)
        ax1.set_ylim(11, 1)
                
        # labels
        ax1.set_xlabel('$x$ (microns)',fontsize = fs)
        ax1.set_ylabel(' Depth $z$ (microns)',fontsize = fs)
        
        ax2.set_xlabel('Time (seconds)', fontsize = fs)
        ax2.set_ylabel('Temperature (C)', fontsize = fs)
        
                
        plt.tight_layout()
        plt.show()
    

    ###### DISPLAY GIU ######
    
    def file_change(change):
        output.clear_output(wait = True)
        
        R = change.new
        L = change.new
        n_rows = change.new
        n_cols = change.new
        T_init = change.new
        RS = change.new
        with output:
            plot_data()
     


    R.observe(file_change, names = 'value')   
    L.observe(file_change, names = 'value') 
    n_rows.observe(file_change, names = 'value')   
    n_cols.observe(file_change, names = 'value') 
    T_init.observe(file_change, names = 'value')
    k.observe(file_change, names = 'value')
    C_p.observe(file_change, names = 'value')
    rho.observe(file_change, names = 'value')
    h.observe(file_change, names = 'value')
    RS.observe(file_change, names = 'value') 
    T_inf.observe(file_change, names = 'value')
    T_init.observe(file_change, names = 'value') 
    power_density.observe(file_change, names = 'value') 
    
    ###### DISPLAY GIU ######
    
    display(accordion)
    display(output)



































