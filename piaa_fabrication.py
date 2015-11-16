"""Outputs (saves) files for each PIAA lens in the Light Forge .dat format

Author: Adam Rains
"""
import numpy as np
import piaa
import os
from itertools import cycle

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
    npix = 256                  # Number of pixels for the simulation
    focal_length_1 = 94         # Focal length of wavefront before PIAA lens #1
    wavelength_in_mm = 0.52/1000.0  # Wavelength of light in mm (0.52um)
    frac_to_focus = thickness / focal_length_1
    
    surf_width = 15            # Width of the square surface clear aperture (mm)
    n_vals = surf_width / dx   # Number of xy values
    cell_width = n_vals / grid_size # Number width of each grid cell
    
    # Use a relative filepath
    dir = os.path.dirname(__file__)
    filename = dir + ("\\piaa_grid_" + str(grid_size) + ".dat")
    
    # Create PIAA lenses for the above parameters
    piaa_lens1, piaa_lens2 = piaa.create_piaa_lenses(alpha, r0, frac_to_focus, 
                                 n_med, thickness, piaa_radius_in_mm, 
                                 real_heights, dx, npix, wavelength_in_mm)
    
    # Get the physical lenses back from the wavefront distortions
    piaa_lens1 /= (2 * np.pi / wavelength_in_mm * (n_med - 1.0))
    piaa_lens2 /= (2 * np.pi / wavelength_in_mm * (n_med - 1.0))
    
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
    for x in xrange(0, grid_size):
        for y in xrange(0, grid_size):
            x1 = 1 + x*cell_width
            x2 = x1 + cell_width
            y1 = 1 + y*cell_width
            y2 = y1 + cell_width
            
            surf_output[x1:x2, y1:y2] = next(lenses)
    
    # Save (Where the z values are in decimal form 1 nm resolution)
    np.savetxt(filename, surf_output, fmt='%10.3f', delimiter="\t")     
    
    # Return the physical lenses (eg - for the purpose of visualisation)
    return piaa_lens1, piaa_lens2
    