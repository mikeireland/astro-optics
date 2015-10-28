"""This function models Phase Induced Amplitude Apodization (PIAA), a beam 
shaping technique.

PIAA is the technique of apodizing a pupil without loss, so that the beam better 
matches e.g. a coronagraph or a fibre input. This module contains a collection 
of routines to help with PIAA calculations.

Authors:
Dr Michael Ireland
Adam Rains
"""

import numpy as np
import opticstools as optics_tools
import pdb
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import utils
import math
import csv
import pylab as pl
import time

def ds_annulus_gauss(r, s, alpha, r0):
    """An annulus morphing into a (truncated) Gaussian. Only
    the first slope is given. The equations are:
    
    I_1(u) = exp(-alpha * u**2) / (pi/alpha [1-exp(-alpha)])
    I_0(r) = 1.0 / [pi (1-r_0^2)] for r0<r<1
    
    With a second radial co-ordinate u, we have for a "slope" s:
    
    u = r + s
    s'(r) = [ I_0 * r / I_1 * (r + s) ]  -  1
    
    The second slope is found directly
    from interpolation. For a second radial co-ordinate u and a
    second slope v, we have: 
    
    v(u) = -s(r), with
    u(r) = r + s(r)
    
    Parameters
    ----------
    alpha: float
        In the formula for I_1 - the exponent of the Gaussian intensity.
    r0: float
        The fractional radius of the telescope "secondary obstruction" in the 
        annulus. e.g. for a 40% obstruction, this would be 0.4. 
    r: float
        The radial co-ordinate of the input wavefront.
    s: float
        The slope of the wavefront at r.
        
    Returns
    -------
    ds: float
        ds/dr at the input co-ordinate r
    
    """
    
    # Equations for intensity, I_0(r) and I_1(u) respectively
    I_0 = 1.0 / (np.pi * (1 - r0**2))
    I_1 = np.exp(-alpha * (r + s)**2) / (np.pi/alpha * (1 - np.exp(-alpha)))
    
    # s'(r) = ds/dr
    ds_dr = (I_0 / I_1) * r / (r + s)  -  1
    
    # In full (*very* slightly different result, on the order of 10^-13)
    # (1 - np.exp(-alpha))*np.exp(alpha*(r + s)**2)*r/alpha/(1-r0**2)/(r+s) - 1
    
    return ds_dr
    
def s_annulus_gauss(alpha, r0, frac_to_focus, n_med, thickness, radius_in_mm, 
                    real_heights=True):
    """Everything is scaled so that a slope of 1 unit corresponds to an angle
    of 1 radius per length. To convert to real units of slope, this has to be 
    multiplied by (radius/(thickness / n_med)). To convert to real heights, we 
    then have to divide by (n-1).
    
    Parameters
    ----------
    alpha: float
        In the formula for I_1 - the exponent of the Gaussian intensity.
    r0: float
        The fractional radius of the telescope "secondary obstruction" in the 
        annulus. e.g. for a 40% obstruction, this would be 0.4. 
    frac_to_focus: float
        The fraction (2nd surface z coord - 1st surface z coord)/
                     (focus z coord       - 1st surface z coord)
        The idea of this variable is that the PIAA should be able to be placed 
        in the optical train and not change the position of the first lens focus
    n_med: float
        Refractive index of the medium. This is used to compensate for the glass
        and give extra power to the new optic as well as to estimate the height.
    thickness: float
        Physical thickness/distance between the realised PIAA lenses
    radius_in_mm: float
        Physical radius for realised PIAA lens
    real_heights: Boolean
        Does nothing at present...
    Returns
    -------
    (r_surface,surface1,surface2,r_slope,slope_1,slope_2): float arrays
        TODO: Fill out for each return variable
    """
    # Define constants
    delta = 1e-2       # Radius of annular dead zone for second pupil
    dt = 1e-3          # Integration step size
    
    # Set up an "integrator" to integrate the differential equation defined by 
    # ds_annulus_gauss (thus obtaining s in terms of r)
    s_integrator = integrate.ode(ds_annulus_gauss) \
                   .set_integrator('vode', method='bdf', with_jacobian=False)
    
    # At the inner edge of the annulus (r = r0), the slope is -r0 + delta. 
    s_integrator.set_initial_value(-r0 + delta, r0).set_f_params( alpha,r0 )
    
    # By Mike's conventional shorthand, rs is and array of r values, and
    rs = np.array([])
    # ss is an array of s values.
    ss = np.array([])
    
    # Integrate ds/dr to find values of s and r
    while s_integrator.successful() and s_integrator.t < 1.0:
        s_integrator.integrate(s_integrator.t+dt)
        rs = np.append(rs,s_integrator.t)
        ss = np.append(ss,s_integrator.y)
    
    # Create an array of u's of the same size and spacing as the r values    
    us = np.linspace(0,1,1.0/dt + 1)
    
    # Slope at the 2nd surface is v, which is something that un-does the slope s
    # Interpolate v from u (r + s) and -s values
    vs = np.interp(us,rs + ss,-ss,left=np.nan, right=0)
    
    ss_full = np.interp(us,rs,ss, left=np.nan, right=0)
    
    # Now fill in the remaining slopes in a smooth way 
    # Replace 'nan' entries at the start of vs and ss_full with evenly spaced 
    # values extrapolated from the first non-'nan' value
    wbad = np.where(vs != vs)[0]
    wgood = np.where(vs == vs)[0]
    vs[wbad] = vs[wgood[0]]*np.arange(wgood[0])/wgood[0]
    
    wbad = np.where(ss_full != ss_full)[0]
    wgood = np.where(ss_full == ss_full)[0]
    ss_full[wbad] = ss_full[wgood[0]]*np.arange(wgood[0])/wgood[0]
    
    # Also integrate the slopes. For surface 1, un-do the convergence that we 
    # would have had on the way to focus...

    # Also convert to real heights. For the PIAA, the whole setup has to be 
    # scaled by the radius of the optic (imagine just rescaling a lens diagram). 
    # The effective optical thickness of the glass is thickness/n_med, and the 
    # aspect ratio is then radius_in_mm/(thickness/n_med)
    
    #For the wavefront curvatures associated with "frac_to_focus" (i.e. undoing 
    # and re-doing the wavefront curvatures), the actual glass thickness is 
    # needed, rather than thickness/n_med

    if (real_heights):
        s1 = integrate.cumtrapz(ss_full*n_med + frac_to_focus*us,us)
        s1 *= radius_in_mm**2/thickness/(n_med-1.0)
    else:
        s1 = integrate.cumtrapz(ss_full + frac_to_focus*us,us)
    
    #Now we need to restore the converging beam. This just means adding a 
    # curvature that gives us a focal length of (1-frac_to_focus)/frac_to_focus
    if (real_heights):
        s2 = integrate.cumtrapz(vs*n_med-frac_to_focus/(1-frac_to_focus)*us,us)
        s2 *= radius_in_mm**2/thickness/(n_med-1.0)
    else:
        s2 = integrate.cumtrapz(vs - frac_to_focus/(1-frac_to_focus)*us,us)

    return 0.5*(us[1:]+us[:-1]), s1, s2 #, us, ss_full, vs (not necessary)
    
def create_piaa_lenses(alpha, r0, frac_to_focus, n_med, thickness, radius_in_mm, 
                       real_heights, dx, npix, wavelength_in_mm):
    """Constructs a pair of PIAA lenses given the relevant parameters.
    
    Parameters
    ----------
    alpha: float
        In the formula for I_1 - the exponent of the Gaussian intensity.
    r0: float
        The fractional radius of the telescope "secondary obstruction" in the 
        annulus. e.g. for a 40% obstruction, this would be 0.4. 
    frac_to_focus: float
        The fraction (2nd surface z coord - 1st surface z coord)/
                     (focus z coord       - 1st surface z coord)
    n_med: float
        Refractive index of the medium. This is used to compensate for the glass
        and give extra power to the new optic as well as to estimate the height.
    thickness: float
        Physical thickness/distance between the realised PIAA lenses
    radius_in_mm: float
        Physical radius for realised PIAA lens
    real_heights: Boolean
        Does nothing at present...
    dx: float
        Resolution/sampling in um/pixel
    npix: int
        The number of pixels.
    wavelength_in_mm: float
        The wavelength of the light in mm.
        
    Returns
    -------
    piaa_lens1: np.array([[...]...])
        The phase aberrations introduced by the first PIAA lens.
    piaa_lens2: np.array([[...]...])
        The phase aberrations introduced by the second PIAA lens, correcting 
        those of the first lens.
    """
    # Generate PIAA surfaces and slopes
    r_surf, surf_1, surf_2 = s_annulus_gauss(alpha, r0, frac_to_focus, n_med, 
                                          thickness, radius_in_mm, real_heights)
    
    x_in_pix = np.arange(npix, dtype=int)
    xy = np.meshgrid(x_in_pix, x_in_pix)
    r = np.sqrt((xy[0] - npix/2.0)**2 + (xy[1] - npix/2.0)**2)
    
    r_normalised = r * dx / (radius_in_mm) #R going from 0 to 1
    # Fill in the lens heights. The negative sign in front of "surf_1" and 
    # "surf_2" are needed to match the Fresnel diffraction sign convention.
    piaa_lens1 = np.zeros( (npix,npix) )
    piaa_lens2 = np.zeros( (npix,npix) )
    piaa_lens1[xy] = np.interp(r_normalised, r_surf, -surf_1)
    piaa_lens2[xy] = np.interp(r_normalised, r_surf, -surf_2)
    
    #Convert these to radians of phase. 
    # As the surface of a lens was returned, we have to multiply by (n-1) to get 
    # the wavefront.
    piaa_lens1 *= 2 * np.pi / wavelength_in_mm * (n_med - 1.0)
    piaa_lens2 *= 2 * np.pi / wavelength_in_mm * (n_med - 1.0)
    
    # PIAA lenses are constructed, return the result
    return piaa_lens1, piaa_lens2
