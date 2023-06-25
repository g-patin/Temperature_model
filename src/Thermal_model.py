####### IMPORT PACKAGES #######

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import ndimage
from ipywidgets import Layout, Button, Box, interact, interactive, fixed, interact_manual
import ipywidgets as wg
from IPython.display import display, clear_output


####### DEFINE FUNCTION #######

def T_model(n_rows, n_cols, R, L, k, C_p, rho, h, dt, T_inf, T_init, RS, power,bs):
    """
    Function that models the diffusion of heat through a paint layer when exposed to a light source.

    Inputs
    ====================================================
    n_rows: an int
    Number of rows in the finite volume mesh
    
    n_cols: an int
    Number of columns in the finite volume mesh
    
    R: a float
    Radius of the cylindrical system (m)

    L: a float
    Height of the cylindrical system (m)

    k: a float
    Thermal conductivity of the substrate (paint) (W/(m K))

    C_p: a float
    Specific heat capacity of the substrate (paint) (J/(kg K))

    rho: a float
    Mass density of the substrate (paint) (kg/m^3)

    h: a float
    Heat transfer coefficient from the top-facing surface of the substrate (paint) (W/(m^2 K))

    dt: a float
    Timestep (s)

    T_inf: a float
    Far-field temperature (°C)

    T_init: a float
    Initial temperature of the substrate (paint) (°C)

    RS: a float
    Mean reflectance spectrum of the sample
    
    power_density: a int
    Power density of the light source illuminating the surface (define unit !)

    bs: a float
    Influence the size of the fading beam. It corresponds precisely to the sigma coefficient attached to the gaussian filter of the fading beam data.

    fwhm : a float
    Full width at half maximum (microns)

    
    Returns
    ====================================================
    It returns a tuple (M,C) where M is an N x N matrix and C a constant term arising from Dirichlet boundary conditions.
    """

    #R = R/1000
    #L = L/1000
    #R = R * 1.e-6
    #L = L * 1.e-6

    dr = R / n_cols                       # width of cells
    dz = L / n_rows                       # height of cells
    alpha = k / (rho * C_p)               # thermal diffusivity m^2/s
    N = n_rows * n_cols                   # number of degrees of freedom

    res1 = R/ n_cols * 1e6
    res2 = L/ n_rows * 1e6 

    area = np.pi*((bs/ 1e6 )/2)**2        # area in m²
    power_density = (power/1e3) / area
    print(f'power density = {power_density} W/m²')

    s = np.zeros(n_cols*2)
    s[n_cols] = 1
    s = ndimage.gaussian_filter(s, (bs/res1)/2.355)
    s /= s.max()    
    s *= power_density

    I = np.zeros(n_cols)                  # power density values (W/m^2) of the light source as a function of radius along the top
    I[:] = s[n_cols:]        

    r = np.linspace(0, R, n_cols + 1)     # radius values at cell boundaries
    r_half = (r[1:] + r[:-1]) / 2         # radius values at cell centers
    z = np.linspace(0, L, n_rows + 1)     # height values at cell boundaries        
        
    M = np.zeros((N, N))                  # coefficients in above equation
    C = np.zeros(N)                       # constant term arising from Dirichlet BCs

    counter = np.zeros(N)


    for row in tqdm(np.arange(n_rows)):
        for col in np.arange(n_cols):            
            if row != 0 and row != n_rows - 1: # neither top nor bottom

                
                ##### AREA 1 #####
                # also not left or right wall, we are in a typical internal cell
                
                if col != 0 and col != n_cols - 1:                    
                    coeff = alpha * dt
                    
                    # which DOF number are we considering now; also which M row
                    i = row * n_cols + col
                    counter[i] += 1
                    i_E = i + 1  # dof for east cell
                    i_W = i - 1  # west cell
                    i_N = i - n_cols
                    i_S = i + n_cols
                    
                    # coefficient for each surrounding cell                     
                    M[i, i_E] = coeff * (r[col + 1] / r_half[col]) / dr ** 2   # coeff on east cell                    
                    M[i, i_W] = coeff * (r[col] / r_half[col]) / dr ** 2       # coeff on west cell
                    M[i, i_N] = coeff * 1 / dz ** 2                            # coeff on north cell
                    M[i, i_S] = coeff * 1 / dz ** 2                            # coeff on south cell

                    # coefficient for this cell
                    M[i, i] = -(M[i, i_E] + M[i, i_W] + M[i, i_N] + M[i, i_S])
                    
                
                ##### AREA 2 #####
                # we are not top or bottom, but we are on the right wall    

                elif (col == n_cols - 1):  
                    coeff = alpha * dt
                    
                    # which DOF number are we considering now; also which M row
                    i = row * n_cols + col
                    counter[i] += 1                    
                    i_W = i - 1            # DOF for west cell
                    i_N = i - n_cols       # DOF for north cell
                    i_S = i + n_cols       # DOF for south cell   
                    # there is no east cell
                    
                    # coefficient for each surrounding cell                   
                    M[i, i_W] = coeff * (r[col] / r_half[col]) / dr ** 2  # coeff on west cell
                    M[i, i_N] = coeff * 1 / dz ** 2                       # coeff on north cell
                    M[i, i_S] = coeff * 1 / dz ** 2                       # coeff on south cell
                    # M[i, i_E] = coeff * (r[col + 1] / r_half[col]) / dr ** 2
                    
                    # coefficient for this cell
                    C[i] = (coeff * (r[col + 1] / r_half[col]) * T_inf / dr ** 2)  # Dirichlet BC
                    M[i, i] = -(
                        (coeff * (r[col + 1] / r_half[col]) / dr ** 2)
                        + M[i, i_W]
                        + M[i, i_N]
                        + M[i, i_S]
                    )
                
                
                ##### AREA 3 #####
                # then we are on the left wall
                
                else:  
                    coeff = alpha * dt
                    
                    # which DOF number are we considering now; also which M row
                    i = row * n_cols + col
                    counter[i] += 1
                    i_E = i + 1            # DOF for east cell                  
                    i_N = i - n_cols       # DOF for north cell
                    i_S = i + n_cols       # DOF for south cell
                    # there is no west cell
                    
                    # coefficient for each surrounding cell                    
                    M[i, i_E] = coeff * (r[col + 1] / r_half[col]) / dr ** 2    # coeff on east cell
                    M[i, i_N] = coeff * 1 / dz ** 2                             # coeff on north cell
                    M[i, i_S] = coeff * 1 / dz ** 2                             # coeff on south cell
                    # coeff on west cell would vanish due to r[col]=0 here:
                    # M[i, i_W] = coeff * (r[col] / r_half[col]) / dr ** 2

                    # coefficient for this cell
                    M[i, i] = -(M[i, i_E] + M[i, i_N] + M[i, i_S])


            # we are top or bottom, but we need to treat them separately
            else:                       
                                              
                if row == n_rows - 1:   #  we are in the bottom areas
                    
                    
                    ##### AREA 4 #####
                    # we are at the bottom but not the bottom right corner or the bottom left corner
                    
                    if (col != 0 and col != n_cols - 1):  
                        coeff = alpha * dt
                        
                        # which DOF number are we considering now; also which M row
                        i = row * n_cols + col
                        counter[i] += 1
                        i_E = i + 1        # DOF for east cell
                        i_W = i - 1        # DOF for west cell
                        i_N = i - n_cols   # DOF for  north cell
                        # we have no south cell, but we assume a phantom with T=T_inf
                        
                        # coefficient for each surrounding cell
                        M[i, i_E] = coeff * (r[col + 1] / r_half[col]) / dr ** 2   # coeff on east cell
                        M[i, i_W] = coeff * (r[col] / r_half[col]) / dr ** 2       # coeff on west cell                        
                        M[i, i_N] = coeff * 1 / dz ** 2                            # coeff on north cell
                        # M[i, i_S] = coeff * 1 / dz ** 2                          # coeff on south cell

                        # coefficient for this cell
                        C[i] = coeff * T_inf / dz ** 2  # Dirichlet BC for bottom face
                        M[i, i] = -(M[i, i_E] + M[i, i_W] + M[i, i_N] + (coeff * 1 / dz ** 2))
                        
                        
                    ##### AREA 5 #####  
                    # we are on the bottom right corner
                    
                    elif col == n_cols - 1:  
                        coeff = alpha * dt
                        
                        # which DOF number are we considering now; also which M row
                        i = row * n_cols + col
                        counter[i] += 1
                        i_W = i - 1        # DOF for west cell
                        i_N = i - n_cols   # DOF for north cell
                        # we have no south cell, but we assume a phantom with T_S=T_inf
                        # we also have no east cell, but we assume a phantom with T_E=T_inf
                        
                        # coefficient for each surrounding cell                      
                        M[i, i_W] = coeff * (r[col] / r_half[col]) / dr ** 2        # coeff on west cell
                        M[i, i_N] = coeff * 1 / dz ** 2                             # coeff on north cell
                        # coeff on south cell
                        # coeff on east cell
                        
                        # coefficient for this cell
                        C[i] = (coeff * T_inf / dz ** 2) + (  # Dirichlet BC for bottom face
                            coeff * (r[col + 1] / r_half[col]) * T_inf / dr ** 2
                        )  # east face
                        M[i, i] = -(
                            (coeff * (r[col + 1] / r_half[col]) / dr ** 2)
                            + M[i, i_W]
                            + M[i, i_N]
                            + (coeff * 1 / dz ** 2)
                        )
                    
                    
                    ##### AREA 6 #####  
                    # we are on the bottom left corner
                    
                    else:  
                        coeff = alpha * dt
                        
                        # which DOF number are we considering now; also which M row
                        i = row * n_cols + col
                        counter[i] += 1
                        i_E = i + 1        # DOF for east cell
                        i_W = i - 1        # DOF for west cell
                        i_N = i - n_cols   # DOF for north cell
                        # we have no south cell, but we assume a phantom with T=T_inf
                        
                        # coefficient for each surrounding cell                        
                        M[i, i_E] = coeff * (r[col + 1] / r_half[col]) / dr ** 2    # coeff on east cell
                        M[i, i_N] = coeff * 1 / dz ** 2                             # coeff on north cell
                        # we have no west cell
                        # M[i, i_S] = coeff * 1 / dz ** 2                           # coeff on south cell

                        # coefficient for this cell
                        C[i] = coeff * T_inf / dz ** 2                              # Dirichlet BC for bottom face
                        M[i, i] = -(M[i, i_E] + M[i, i_N] + (coeff * 1 / dz ** 2))

                
                # we are on the top areas
                else:
                    
                    ##### AREA 7 #####
                    # we are on the top areas but not the upper left or upper right corner
                    
                    if (col != 0 and col != n_cols - 1):  
                        coeff = alpha * dt
                        
                        # which DOF number are we considering now; also which M row
                        i = row * n_cols + col
                        counter[i] += 1
                        i_E = i + 1        # DOF for east cell
                        i_W = i - 1        # DOF for west cell                        
                        i_S = i + n_cols   # DOF for south cell 
                        # we don't have a north cell
                        
                        # coefficient for each surrounding cell                         
                        M[i, i_E] = coeff * (r[col + 1] / r_half[col]) / dr ** 2    # coeff on east cell                        
                        M[i, i_W] = coeff * (r[col] / r_half[col]) / dr ** 2        # coeff on west cell
                        M[i, i_S] = coeff * 1 / dz ** 2                             # coeff on south cell
                        # there is no north cell M[i, i_N] = coeff * 1 / dz ** 2    # coeff on north cell

                        # coefficient for this cell
                        C[i] = (
                            coeff
                            * (
                                (h * T_inf)  # heat xfer top BC
                                + ((1 - RS) * I[col])  # luminuous power BC
                            )
                            / (k * dz)
                        )
                        M[i, i] = -(M[i, i_E] + M[i, i_W] + (coeff * h / (k * dz)) + M[i, i_S])
                        
                        
                    ##### AREA 9 #####
                    # the upper left corner cell
                    elif col == 0:  
                        coeff = alpha * dt
                        
                        # which DOF number are we considering now; also which M row
                        i = row * n_cols + col
                        counter[i] += 1
                        i_E = i + 1        # DOF for east cell
                        i_S = i + n_cols   # DOF for south cell
                        # there is no west cell
                        # we don't have a north cell
                        
                        # coefficient for each surrounding cell                        
                        M[i, i_E] = coeff * (r[col + 1] / r_half[col]) / dr ** 2    # coeff on east cell
                        M[i, i_S] = coeff * 1 / dz ** 2                             # coeff on south cell
                        # coeff on west cell
                        # there is no north cell M[i, i_N] = coeff * 1 / dz ** 2    # coeff on north cell
                        
                        # coefficient for this cell
                        C[i] = (
                            coeff
                            * (
                                (h * T_inf)            # heat transfer top BC
                                + ((1 - RS) * I[col])  # luminuous power BC
                            )
                            / (k * dz)
                        )
                        M[i, i] = -(M[i, i_E] + (coeff * h / (k * dz)) + M[i, i_S])
                        
                        
                    ##### AREA 8 #####
                    # the upper right corner cell
                    else: 
                        coeff = alpha * dt
                        # which DOF number are we considering now; also which M row
                        
                        i = row * n_cols + col
                        counter[i] += 1                        
                        i_W = i - 1        # DOF for west cell                        
                        i_S = i + n_cols   # DOF for south cell
                        # we don't have a north cell
                        # we don't have a east cell
                        
                        # coefficient for each surrounding cell                   
                        M[i, i_W] = coeff * (r[col] / r_half[col]) / dr ** 2        # coeff on west cell
                        M[i, i_S] = coeff * 1 / dz ** 2                             # coeff on south cell
                        # M[i, i_N] = coeff * 1 / dz ** 2                           # coeff on north cell                        
                        # M[i, i_E] = coeff * (r[col + 1] / r_half[col]) / dr ** 2  # coeff on east cell
                        
                        # coefficient for this cell
                        C[i] = (
                            coeff
                            * (
                                (h * T_inf)            # heat transfer top BC
                                + ((1 - RS) * I[col])  # luminuous power BC
                            )
                            / (k * dz)
                        ) + (coeff * T_inf * (r[col + 1] / r_half[col]) / dr ** 2)
                        M[i, i] = -((coeff * (r[col + 1] / r_half[col]) / dr ** 2) + M[i, i_W] + (coeff * h / (k * dz)) + M[i, i_S])

    return M,C
                    











































