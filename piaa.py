"""Phase induced amplitude apodisation (PIAA) is the technique of apodising a pupil
without loss, so that the beam better matches e.g. a coronagraph or a fiber input.
This module contains a collection of routines to help with PIAA calculations

e.g.:

(r_surface,surface1,surface2,r_slope,slope_1,slope_2) = piaa.s_annulus_gauss(2.0, 0.35, 0.01)
plot(r_surface, surface1)
plot(r_surface, surface2)
"""

import numpy as np
import optics_tools
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
    
    # Alternatively, in full (results in *very* slightly different result, on the order of 10^-13)
    # (1 - np.exp(-alpha)) * np.exp(alpha*(r + s)**2) * r / alpha / (1-r0**2) / (r+s) - 1
    
    return ds_dr
    
def s_annulus_gauss(alpha,r0,frac_to_focus,delta=1e-2,dt=1e-3, n_med=1.5, thickness=15.0, radius_in_mm=1.0, real_heights=True):
    """Everything is scaled so that a slope of 1 unit corresponds to an angle
    of 1 radius per length. To convert to real units of slope, this has to be multiplied
    by (radius/(thickness / n_med)). To convert to real heights, we then have to divide 
    by (n-1).
    
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
    delta: float
        As the slope can not be infinite... delta describes
        the radius of the annular dead zone in the second pupil. 
    dt: float
        Step size for integration.
    n_med: float
        Refractive index of the medium. This is used to compensate for the glass
        (and give extra power to the new optic) as well as to estimate the height.
    thickness: float
        Physical thickness/distance between the realised PIAA lenses
    radius_in_mm: float
        Physical radius for realised PIAA lens
    real_heights: Boolean
        Does nothing at present...
    Returns
    -------
    (r_surface,surface1,surface2,r_slope,slope_1,slope_2): float arrays
        um is the radial
    """
    #Set up an "integrator" to integrate the differential equation defined by ds_annulus_gauss (thus obtaining s in terms of r)
    s_integrator = integrate.ode(ds_annulus_gauss).set_integrator('vode', method='bdf', with_jacobian=False)
    
    #At the inner edge of the annulus (r = r0), the slope is -r0 + delta. 
    s_integrator.set_initial_value(-r0 + delta, r0).set_f_params( alpha,r0 )
    
    #By Mike's conventional shorthand, rs is and array of r values, and
    rs = np.array([])
    #ss is an array of s values.
    ss = np.array([])
    
    # Integrate ds/dr to find values of s and r
    while s_integrator.successful() and s_integrator.t < 1.0:
        s_integrator.integrate(s_integrator.t+dt)
        rs = np.append(rs,s_integrator.t)
        ss = np.append(ss,s_integrator.y)
    
    # Create an array of u's of the same size and spacing as the r values    
    us = np.linspace(0,1,1.0/dt + 1)
    
    #The slope at the 2nd surface is "v", which is something that un-does the slope "s".
    #Interpolate v from u (r + s) and -s values
    vs = np.interp(us,rs + ss,-ss,left=np.nan, right=0)
    
    #Now fill in the remaining slopes in a smooth way.
    ss_full = np.interp(us,rs,ss, left=np.nan, right=0)
    wbad = np.where(vs != vs)[0]
    wgood = np.where(vs == vs)[0]
    vs[wbad] = vs[wgood[0]]*np.arange(wgood[0])/wgood[0]
    wbad = np.where(ss_full != ss_full)[0]
    wgood = np.where(ss_full == ss_full)[0]
    ss_full[wbad] = ss_full[wgood[0]]*np.arange(wgood[0])/wgood[0]
    
    #Inserting a glass section frac_to_focus thick changes the target 
    #focus position behind the second surface from 
    # (1.0-frac_to_focus)/frac_to_focus to 
    #(1.0-frac_to_focus/frac_to_focus) + (n_med-1)
    power1 = frac_to_focus/(1.0-frac_to_focus) - 1.0/( (1.0-frac_to_focus)/frac_to_focus + (n_med-1.0))
    
    #Also integrate the slopes.
    s1 = integrate.cumtrapz(ss_full + frac_to_focus*us,us)
    s2 = integrate.cumtrapz(vs - (frac_to_focus/(1-frac_to_focus) + power1)*us,us)
    
    #Convert to real heights
    s1 *= radius_in_mm/(thickness/n_med) / (n_med - 1)
    s2 *= radius_in_mm/(thickness/n_med) / (n_med - 1)
    
    return 0.5*(us[1:]+us[:-1]), s1,s2,us,ss_full, vs
    
def create_piaa_lenses(alpha, r0, frac_to_focus, delta, dt, n_med, thickness, radius_in_mm, real_heights, dx, npix, wavelength_in_mm):
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
    delta: float
        As the slope can not be infinite... delta describes
        the radius of the annular dead zone in the second pupil. 
    dt: float
        Step size for integration.
    n_med: float
        Refractive index of the medium. This is used to compensate for the glass
        (and give extra power to the new optic) as well as to estimate the height.
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
    
    piaa_lens1, piaa_lens2: 
    
    """
    # Generate PIAA surfaces and slopes
    r_surface, surface1, surface2, r_slope, slope_1, slope_2 = s_annulus_gauss(alpha, r0, frac_to_focus, delta, dt, n_med, thickness, radius_in_mm, real_heights)
    
    x_in_pix = np.arange(npix, dtype=int)
    xy = np.meshgrid(x_in_pix, x_in_pix)
    r = np.sqrt((xy[0] - npix/2.0)**2 + (xy[1] - npix/2.0)**2)
    
    r_normalised = r * dx / (radius_in_mm) #R going from 0 to 1
    # Fill in the lens heights. The negative sign in front of "surface1" and 
    # "surface2" are needed to match the Fresnel diffraction sign convention.
    piaa_lens1 = np.zeros( (npix,npix) )
    piaa_lens2 = np.zeros( (npix,npix) )
    piaa_lens1[xy] = np.interp(r_normalised, r_surface, -surface1)
    piaa_lens2[xy] = np.interp(r_normalised, r_surface, -surface2)
    
    #Convert these to radians of phase. 
    # As the surface of a lens was returned, we have to multiply by (n-1) to get the wavefront.
    piaa_lens1 *= 2 * np.pi / wavelength_in_mm * (n_med - 1.0)
    piaa_lens2 *= 2 * np.pi / wavelength_in_mm * (n_med - 1.0)
    
    # PIAA lenses are constructed, return the result
    return piaa_lens1, piaa_lens2

def propagate_and_save(directory, distance_step, dz, seeing=1.0, gaussian_alpha=2.0):
    """Creates and saves plots for the PIAA system in increments of the specified distance step.
    
    Parameters
    ----------
    directory: string
        The directory to save the plots to.
    distance_step: integer
        The distance to propagate the wavefront forward for each plot
        Should be a fraction of the thickness and focal length
    dz: float   
        Free parameter, determines the magnification of the lenslet
    seeing: float
        The seeing for the telescope.        
    gaussian_alpha: float
        Exponent constant from Gaussian
    Returns
    -------
    electric_field: 2D numpy.ndarray
        The electric field entering the optical fibre
    """
    
    # Create the lens object
    lens = PIAA_System()
    
    # Initialise seeing and alpha
    lens.seeing_in_arcsec = seeing
    lens.alpha = gaussian_alpha
    
    # Create the PIAA components and the annulus
    lens.create_piaa()
    lens.create_annulus()
    
    # The image count to uniquely identify each image created
    file_number = 0
    
    # [1, 2] - Compute the input electric field (annulus x distorted wavefront from atmosphere) 
    print "[1, 2] - Compute Input"
    electric_field = lens.calculate_input_e_field()
    utils.save_plot(np.abs(electric_field)**0.5, 'Electric Field: Input', directory, ("%05d" % file_number))
    file_number += 1
    
    # [3] - Pass the electric field through the first PIAA lens
    print "[3] - PIAA #1"
    electric_field = lens.apply_piaa_lens(lens.piaa_lens1, electric_field)
    
    # [4] - Propagate the electric field through glass to the second lens
    print "[4] - PIAA #1 - #2"
    distance_to_piaa_2 = lens.thickness
    remaining_distance_to_piaa_2 = distance_to_piaa_2
    while remaining_distance_to_piaa_2 > 0:
        # Propagate the field by the distance step (or the distance remaining if it is smaller than the step)
        if remaining_distance_to_piaa_2 > distance_step:
            electric_field = lens.propagate(electric_field, distance_step / lens.n_med)
            remaining_distance_to_piaa_2 -= distance_step   
        else:
            electric_field = lens.propagate(electric_field, remaining_distance_to_piaa_2 / lens.n_med)
            remaining_distance_to_piaa_2 = 0
        
        # Save the plot at each step
        title = 'Electric Field: ' + str(distance_to_piaa_2 - remaining_distance_to_piaa_2) + ' of ' + str(distance_to_piaa_2) + " mm from PIAA Lens #2"
        utils.save_plot(np.abs(electric_field)**0.5, title, directory, ("%05d" % file_number))
        file_number += 1 
       
    
    # [5] - Pass the electric field through the second PIAA lens
    print "[5] - PIAA #2"
    electric_field = lens.apply_piaa_lens(lens.piaa_lens2, electric_field)
    electric_field = lens.curved_wavefront(electric_field, lens.focal_length_1)
    
    # [6] - Propagate the electric field to the 3.02mm lens
    print "[6] - PIAA #2 - 3.02mm Lens"
    distance_to_lens = lens.focal_length_1 + lens.focal_length_2 + dz
    remaining_distance_to_lens = distance_to_lens
    while remaining_distance_to_lens > 0:
        # Propagate the field by the distance step (or the distance remaining if it is smaller than the step)
        if remaining_distance_to_lens > distance_step:
            electric_field = lens.propagate(electric_field, distance_step)
            remaining_distance_to_lens -= distance_step   
        else:
            electric_field = lens.propagate(electric_field, remaining_distance_to_lens)
            remaining_distance_to_lens = 0
        
        # Save the plot at each step
        title = 'Electric Field: ' + str(distance_to_lens - remaining_distance_to_lens) + ' of ' + str(distance_to_lens) + " mm from 3.02mm Lens After PIAA Lens #2"
        utils.save_plot(np.abs(electric_field)**0.5, title, directory, ("%05d" % file_number))
        file_number += 1  
        
    # [7] - Pass the electric field through the 3.02mm lens
    print "[7] - 3.02mm Lens"
    electric_field = lens.apply_circular_lens(electric_field, lens.circular_lens_diameter)
    electric_field = lens.curved_wavefront(electric_field, lens.focal_length_2)    
    utils.save_plot(np.abs(electric_field)**0.5, "Electric Field: At 3.02mm Lens", directory, ("%05d" % file_number))
    file_number += 1
    
    # [8] - Propagate to the 1mm square lenslet
    print "[8] - 3.02mm Lens - Square Lenslet"
    distance_to_lenslet = 1 / ( 1/lens.focal_length_2 - 1/(lens.focal_length_2 + dz) )  #From thin lens formula
    remaining_distance_to_lenslet = distance_to_lenslet
    while remaining_distance_to_lenslet > 0:
        # Propagate the field by the distance step (or the distance remaining if it is smaller than the step)
        if remaining_distance_to_lenslet > distance_step:
            electric_field = lens.propagate(electric_field, distance_step)
            remaining_distance_to_lenslet -= distance_step  
        else:
            electric_field = lens.propagate(electric_field, remaining_distance_to_lenslet)
            remaining_distance_to_lenslet = 0
        
        # Save the plot at each step
        title = 'Electric Field: ' + str(distance_to_lenslet - remaining_distance_to_lenslet) + ' of ' + str(distance_to_lenslet) + " mm from Square Lenslet"
        utils.save_plot(np.abs(electric_field)**0.5, title, directory, ("%05d" % file_number))
        file_number += 1    
    
    # [9] - Multiply by square lenslet, crop/interpolate and multiply by curved wf
    print "[9] - Square Lenslet"
    electric_field = lens.apply_lenslet(electric_field)
    electric_field = lens.crop_and_interpolate(electric_field)
    #electric_field = lens.curved_wavefront(electric_field, lens.focal_length_3)   
    utils.save_plot(np.abs(electric_field)**0.5, "Electric Field: At Square Lenslet", directory, ("%05d" % file_number))
    file_number += 1
    
    # [10] - Propagate to the fibre optic cable
    print "[10] - Square Lenslet - Optical Fibre"
    distance_to_fibre = 1.0/(1.0/lens.focal_length_3 - 1.0/(distance_to_lenslet - lens.focal_length_2))
    remaining_distance_to_fibre = distance_to_fibre
    while remaining_distance_to_fibre > 0:
        # Propagate the field by the distance step (or the distance remaining if it is smaller than the step)
        if remaining_distance_to_fibre > distance_step:
            electric_field = lens.propagate(electric_field, distance_step)
            remaining_distance_to_fibre -= distance_step  
        else:
            electric_field = lens.propagate(electric_field, remaining_distance_to_fibre)
            remaining_distance_to_fibre = 0
        
        # Save the plot at each step
        title = 'Electric Field: ' + str(distance_to_fibre - remaining_distance_to_fibre) + ' of ' + str(distance_to_fibre) + " mm from Optical Fibre"
        utils.save_plot(np.abs(electric_field)**0.5, title, directory, ("%05d" % file_number))
        file_number += 1
        
    # [11] - Compute the near field profile and the fibre coupling
    coupling = lens.compute_coupling(electric_field)
    return coupling, electric_field    
    
        
