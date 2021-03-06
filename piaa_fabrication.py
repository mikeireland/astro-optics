"""Outputs (saves) files for each PIAA lens in the Light Forge .dat format

Author: Adam Rains
"""
import numpy as np
import piaa
import os
from itertools import cycle
import pylab as pl
import ft_filter

def generate_light_forge_file(grid_size=4):
    """Saves files in Light Forge file format (GridXYZ), surface height (z) on
    a rectangular x-y grid (15 x 15 mm). 
     -  x-y are given in 10 um steps
     -  z is given in um with 1 nm resolution (3 decimal places)
     -  First row and column are the x and y values respectively, in ascending
        order (with [0,0] being 0.0)
     -  Files are saved in a .dat format and later zipped
    
    Returns
    -------
    surf_output: numpy array
        Light Forge format output of tessellated lenses
    piaa_lens1: numpy.array
        The physical representation of the first PIAA lens, units in mm.
    piaa_lens2: numpy.array
        The physical representation of the second PIAA lens, units in mm.
    """
    #Constants
    alpha = 2.0                 # Gaussian exponent term
    r0 = 0.40                   # Fractional size of secondary mirror
    n_med = 1.5                 # Refraction index of the medium
    thickness = 15.0            # Physical thickness between the PIAA lenses
    piaa_radius_in_mm = 1.0     # Physical radius
    real_heights = True         # ...
    dx = 10.0/1000.0            # Resolution/sampling in mm/pixel
    npix = 352                  # Number of pixels for the simulation
    focal_length_1 = 94         # Focal length of wavefront before PIAA lens #1
    wavelength_in_mm = 0.52/1000.0  # Wavelength of light in mm (0.52um)
    frac_to_focus = thickness / focal_length_1
    
    surf_width = 15            # Width of the square surface clear aperture (mm)
    n_vals = surf_width / dx   # Number of xy values
    cell_width = n_vals / grid_size # Number width of each grid cell
    
    # Space not taken up by lenses (mm)
    FACTOR = 8 # Accounts for missing factor somewhere. Possibly wrong units
    remaining_space = 15 - grid_size*piaa_radius_in_mm*2
    space_per_side = int(remaining_space/grid_size/dx/2*FACTOR) # Buffer per side(px)

    # If odd, make even
    if space_per_side % 2 != 0:
        space_per_side -= 1
    
    # Use a relative filepath
    dir = os.path.dirname(__file__)
    filename = dir + ("\\piaa_grid_" + str(grid_size) + ".dat")
    
    # Generate PIAA surfaces and slopes
    # Require access to 2D lens shape (height as a function of radius)
    # Much easier to remove discontinuities in 2D than 3D
    r_surf, surf_1, surf_2 = piaa.s_annulus_gauss(alpha, r0, frac_to_focus, 
                                                 n_med, thickness, 
                                                 piaa_radius_in_mm, 
                                                 real_heights)
    
    x_in_pix = np.arange(npix, dtype=int)
    xy = np.meshgrid(x_in_pix, x_in_pix)
    r = np.sqrt((xy[0] - npix/2.0)**2 + (xy[1] - npix/2.0)**2)
    r_normalised = r * dx / (piaa_radius_in_mm) #R going from 0 to 1
    
    # Save line lengths - Radius of surface 1 and 2, not including ramps. 
    LS1 = len(surf_1)
    LS2 = len(surf_2)
    
    # Construct new array with spline component, using the available space on 
    # either side of the lens
    s1_ramped = np.zeros(LS1 + space_per_side)
    s2_ramped = np.zeros(LS2 + space_per_side)
    
    # Add the existing lens
    s1_ramped[0:LS1] = surf_1
    s2_ramped[0:LS2] = surf_2
    
    # Add spline. The x values are normalised with 1=grid spacing.
    # The derivative is normalised in the same way.
    # As per PIAA/Fresnel convention, surface #1 is < 0 (that is to say shifted
    # below 0, not merely negative 
    # (TODO: shift the curve > 0 earlier in the code, rather accounting for it
    # multiple times and decreasing readability)
    x_values = np.arange(0,space_per_side+1)
    surf_1_pos = [(i + np.abs(np.min(surf_1))) for i in surf_1]
    dy_dx_1 = surf_1_pos[-2] - surf_1_pos[-1]
    s1_spline = single_spline_interpolation(x_values, surf_1_pos[-1], dy_dx_1)
    s1_ramped[LS1-1:] = [(i - np.max(surf_1_pos)) for i in s1_spline] 

    x_values = np.arange(0,space_per_side+1)    
    dy_dx_2 = surf_2[-2] - surf_2[-1]
    s2_spline = single_spline_interpolation(x_values, surf_2[-1], dy_dx_2)
    s2_ramped[LS2-1:] = s2_spline
    
    # Fill in the lens heights. The negative sign in front of "surf_1" and 
    # "surf_2" are needed to match the Fresnel diffraction sign convention.
    piaa_lens1 = np.zeros( (npix,npix) )
    piaa_lens2 = np.zeros( (npix,npix) )
    
    piaa_lens1[xy] = np.interp(r_normalised, np.arange(0.0005, 
                             (LS1 + 1.0*space_per_side)/LS1, 0.001), -s1_ramped)
    piaa_lens2[xy] = np.interp(r_normalised, np.arange(0.0005, 
                             (LS2 + 1.0*space_per_side)/LS2, 0.001), -s2_ramped)
    
    # Flip the lenses so that z=0 is the flat surface
    piaa_lens1 = np.abs(piaa_lens1 - np.max(piaa_lens1))
    piaa_lens2 = np.abs(piaa_lens2 - np.max(piaa_lens2))
    
    # Insert each lens into a larger array of zeroes of size 15/0.01/grid_size
    p1 = np.zeros( [cell_width, cell_width] )
    p2 = np.zeros( [cell_width, cell_width] )
    edge = (cell_width - npix) / 2
    p1[edge:(edge + npix), edge:(edge + npix)] = piaa_lens1
    p2[edge:(edge + npix), edge:(edge + npix)] = piaa_lens2
    
    # Set up the cycler to alternate between each lens
    piaa_array = [p1*1000, p2*1000]
    lenses = cycle(piaa_array)
    
    # Create output array, width = 15/0.01 + 1 (+1 to account for axes)
    surf_output = np.zeros( [n_vals + 1, n_vals + 1] )
    
    # Give x/y in um, with [0,0] being at the centre of the lens
    xy_vals = (np.arange(0, surf_width, dx) - 0.5*surf_width) * 1000 #mm -> um
    
    # Construct the output array, where the first row/col are x/y respectively
    surf_output[0,1:] = xy_vals
    surf_output[1:,0] = xy_vals

    # Tessellate lens #1 and #2 over the grid
    # Shift the outer row/column of lenses to edge to have more space between
    for x in xrange(0, grid_size):
        # Determine x roll
        if x == 0:
            x_roll = -1
        elif x == (grid_size - 1):
            x_roll = 1
        else:
            x_roll = 0
            
        for y in xrange(0, grid_size):
            # Determine y roll
            if y == 0:
                y_roll = -1
            elif y == (grid_size - 1):
                y_roll = 1   
            else:
                y_roll = 0
             
            # Set up min and max x/y
            x1 = 1 + x*cell_width
            x2 = x1 + cell_width
            y1 = 1 + y*cell_width
            y2 = y1 + cell_width
            
            # Shift exterior lenses to outside of grid
            # Will allow for fewer cuts
            y_shift = np.roll(next(lenses), int(y_roll*edge),axis=1)
            xy_shift = np.roll(y_shift, int(x_roll*edge), axis=0)

            surf_output[x1:x2, y1:y2] = xy_shift
    
    # Save (Where the z values are in decimal form 1 nm resolution)
    np.savetxt(filename, surf_output, fmt='%10.3f', delimiter="\t")     
    
    # Return the physical lenses (eg - for the purpose of visualisation)
    return surf_output, piaa_lens1, piaa_lens2, 
 
def single_spline_interpolation(x_values_in, A, B):
    """
    Performs a single 3rd order polynomial spline interpolation from the edge of
    the constructed lens to a height of 0.0.
    
    Parameters
    ----------
    x_values: float array 
        Values for the x axis including end points. Must be in ascending order.
    A: float
        Height at the edge of the lens where the spline is to start
    B: float
        Derivative at the edge of the lens where the spline is to start  

    Returns
    -------
    y_values: float array
        Array of heights of the original lens + spline segment. Reversed such 
        that zero is on the right.
    """
    # Initialise output
    y_values = []
    
    #Scale the x-axis.
    x_values = x_values_in.astype(float)
    x_offset = x_values[0]
    x_length = x_values[-1] - x_values[0]
    x_scaled = (x_values - x_offset)/x_length
    B_scaled = B*x_length
    
    # Solve simultaneous equations for a and b terms (c = d = 0)
    # Use the terms to solve the 3rd order polynomial for each x values
    for x in x_scaled:
        ans     = np.array([A, B_scaled])
        coeff   = np.array([[1, 1],
                            [3, 2]])            
        
        ab = np.linalg.solve(coeff, ans)
        
        y = ab[0]*x**3 + ab[1]*x**2
        y_values.append(y)
        
    # Return the result, but reverse so that the lens is on the left of the 
    # spline segment
    return y_values[::-1]
    
def ft_filter_and_save(surface, grid_size=4):
    """
    """
    # Fourier Filter (Don't filter the axes required as part of the Light Forge
    # requirements)
    ft_filtered_surf = np.zeros( [len(surface), len(surface)])
    ft_filtered_surf[0,:] = surface[0,:]
    ft_filtered_surf[:,0] = surface[:,0]
    ft_filtered_surf[1:,1:] = ft_filter.ft_filter(surface[1:,1:])

    # Save
    dir = os.path.dirname(__file__)
    filename = dir + ("\\piaa_grid_" + str(grid_size) + "_ft_filtered" + ".dat")
    np.savetxt(filename, ft_filtered_surf, fmt='%10.3f', delimiter="\t")  

    # Return the filtered surface
    return ft_filtered_surf