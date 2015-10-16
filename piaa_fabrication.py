"""Outputs (saves) files for each PIAA lens in the Light Forge .dat format

Author: Adam Rains
"""
import numpy as np
import piaa
import os

def generate_light_forge_file():
    """Saves files in Light Forge file format (GridXYZ), surface height (z) on
    a rectangular x-y grid. 
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
    delta = 1e-2                # Radius of annular dead zone for second pupil
    dt = 1e-3                   # Integration step size
    n_med = 1.5                 # Refraction index of the medium
    thickness = 15.0            # Physical thickness between the PIAA lenses
    piaa_radius_in_mm = 1.0     # Physical radius
    real_heights = True         # ...
    dx = 10.0/1000.0             # Resolution/sampling in mm/pixel
    npix = 256                 # Number of pixels for the simulation
    focal_length_1 = 94         # Focal length of wavefront before PIAA lens #1
    wavelength_in_mm = 0.52/1000.0  # Wavelength of light in mm (0.52um)
    
    frac_to_focus = thickness / focal_length_1
    
    # Use a relative filepath
    dir = os.path.dirname(__file__)
    filename_1 = dir + "\\p1.dat"
    filename_2 = dir + "\\p2.dat"
    
    # Create PIAA lenses for the above parameters
    piaa_lens1, piaa_lens2 = piaa.create_piaa_lenses(alpha, r0, frac_to_focus, 
                                 delta, dt, n_med, thickness, piaa_radius_in_mm, 
                                 real_heights, dx, npix, wavelength_in_mm)
    
    # Get the physical lenses back from the wavefront distortions
    piaa_lens1 /= (2 * np.pi / wavelength_in_mm * (n_med - 1.0))
    piaa_lens2 /= (2 * np.pi / wavelength_in_mm * (n_med - 1.0))
    
    # Create output array, with room for x and y values
    p1_output = np.zeros( [npix + 1, npix + 1] )
    p2_output = np.zeros( [npix + 1, npix + 1] )
    
    # Give x/y in um, with [0,0] being at the centre of the lens
    xy_vals = (np.arange(0, npix*dx, dx) - 0.5*npix*dx) * 1000 # x1000: mm -> um
    
    # Construct the output array, where the first row/col are x/y respectively
    p1_output[0,1:] = xy_vals
    p1_output[1:,0] = xy_vals
    p1_output[1:,1:] = piaa_lens1 * 1000    # x1000: mm -> um
    p2_output[0,1:] = xy_vals
    p2_output[1:,0] = xy_vals
    p2_output[1:,1:] = piaa_lens2 * 1000    # x1000: mm -> um
    
    # Save (Where the z values are in decimal form 1 nm resolution)
    np.savetxt(filename_1, p1_output, fmt='%10.3f', delimiter="\t")
    np.savetxt(filename_2, p2_output, fmt='%10.3f', delimiter="\t")             
    
    # Return the physical lenses (eg - for the purpose of visualisation)
    return piaa_lens1, piaa_lens2
    