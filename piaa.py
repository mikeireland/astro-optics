"""Phase induced amplitude apodisation (PIAA) is the technique of apodising a pupil
without loss, so that the beam better matches e.g. a coronagraph or a fiber input.
This module contains a collection of routines to help with PIAA calculations

e.g.:

(r_surface,surface1,surface2,r_slope,slope_1,slope_2) = piaa.s_annulus_gauss(2.0, 0.35, 0.01)
plot(r_surface, surface1)
plot(r_surface, surface2)
"""

import numpy as np
import optics
import pdb
from scipy import integrate
import matplotlib.pyplot as plt

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
    
    return (1 - np.exp(-alpha)) * np.exp(alpha*(r + s)**2) * r / alpha / (1-r0**2) / (r+s) - 1
    
def s_annulus_gauss(alpha,r0,frac_to_focus,delta=1e-2,dt=1e-3, n_med=1.5, thickness=15.0, radius_in_mm=1.0, real_heights=True):
    """Everything is scaled so that a slope of 1 unit corresponds to an angle
    of 1 radius per length. To convert to real units of slope, this has to be multiplied
    by (radius/(thickness / n_med)). To convert to real heights, we then have to divide 
    by (n-1).
    
    Parameters
    ----------
    alpha: float
        The 
    frac_to_focus: float
        The fraction (2nd surface z coord - 1st surface z coord)/
                     (focus z coord       - 1st surface z coord)
    delta: float
        As the slope can not be infinite... delta describes
        the radius of the annular dead zone in the second pupil. 
    n_med: float
        Refractive index of the medium. This is used to compensate for the glass
        (and give extra power to the new optic) as well as to estimate the height.
        
    Returns
    -------
    (r_surface,surface1,surface2,r_slope,slope_1,slope_2): float arrays
        um is the radial
    """
    #Set up an "integrator" to integrate the differential equation defined by ds_annulus_gauss
    s_integrator = integrate.ode(ds_annulus_gauss).set_integrator('vode', method='bdf', with_jacobian=False)
    #At the inner edge of the annulus (r = r0), the slope is -r0 + delta. 
    s_integrator.set_initial_value(-r0 + delta, r0).set_f_params( alpha,r0 )
    #By Mike's conventional shorthand, rs is and array of r value, and
    #ss is an array of s values.
    rs = np.array([])
    ss = np.array([])
    while s_integrator.successful() and s_integrator.t < 1.0:
        s_integrator.integrate(s_integrator.t+dt)
        rs = np.append(rs,s_integrator.t)
        ss = np.append(ss,s_integrator.y)
    us = np.linspace(0,1,1.0/dt + 1)
    #The slope at the 2nd survace is "v", which is something that un-does
    #the slope "s".
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
    
def annulus_gauss_fresnel(alpha,r0,ignore_piaa=False):
    """Use Fresnel diffraction to verify that s_annulus_gauss does roughly the right thing.
    NB Parameters are HARDWIRED so this is for TESTING only.
    
    Adam's notes 2 though 6. Note that to apply an atmosphere, you need to go:
    electric_field = exp(1j * wf), 
    with the wf normalised so that it is in radians of phase. See web pages:
    http://www.ing.iac.es/Astronomy/development/hap/dimm.html
    http://community.dur.ac.uk/james.osborn/thesis/thesisse3.html
    
    e.g. Scale a wavefront like this (with "seeing" in radians):
    r_0 = seeing / 0.98 / wavelength
    wf_in_radians = kmf(STUFF) * np.sqrt(6.88*(pixel_size/r_0)**(5.0/3.0))
    """
    n_med = 1.5
    thickness = 15.0
    
    wavelength_in_mm = 0.5/1000.0
    #Simulate something not going to focus.
    frac_to_focus = 1e-6 
    
    r_surface,surface1,surface2,dummy1,dummy2,dummy3 = s_annulus_gauss(alpha,r0,frac_to_focus,n_med=n_med,thickness=thickness)
    dx = 5.0/1000.0           #Single pixel sampling in mm
    pupil_diam = 2.0           #Diameter in mm.
    secondary_obstruction = 0.8 #Obstruction in mm
    
    npix = 1024
    
    x_in_pix = np.arange(npix,dtype=int)
    xy = np.meshgrid(x_in_pix,x_in_pix)
    r = np.sqrt((xy[0] - npix/2.0)**2 + (xy[1] - npix/2.0)**2)
    r_normalised = r*dx/(pupil_diam/2) #R going from 0 to 1
    #Fill in the lens heights. The negative sign in front of "surface1" and 
    #"surface2" are needed to match the fresnel diffraction sign convention.
    piaa_lens1 = np.zeros( (npix,npix) )
    piaa_lens2 = np.zeros( (npix,npix) )
    piaa_lens1[xy] = np.interp(r_normalised, r_surface, -surface1)
    piaa_lens2[xy] = np.interp(r_normalised, r_surface, -surface2)
    
    #Convert these to radians of phase at 500nm. As the surface of a lens was returned,
    #we have to multiply by (n-1) to get the wavefront.
    piaa_lens1 *= 2*np.pi/wavelength_in_mm*(n_med - 1.0)
    piaa_lens2 *= 2*np.pi/wavelength_in_mm*(n_med - 1.0)
    
    #For a test, try to ignore the PIAA and see what happens.
    if (ignore_piaa):
        piaa_lens1 *= 0
        piaa_lens2 *= 0
    
    #Compute the input electric fields... 
    efield_in = optics.circle(npix,pupil_diam/dx) - optics.circle(npix,secondary_obstruction/dx)
    efield_lens1 = efield_in * np.exp(1j * piaa_lens1)
    #Propagate...
    efield_lens2_before = optics.fresnel(efield_lens1, dx, thickness/n_med, wavelength_in_mm)
    efield_lens2_after = efield_lens2_before * np.exp(1j * piaa_lens2)

    
    #Show the electric field at the second lens
    plt.clf()
    plt.imshow(np.abs(efield_lens2_before)**0.5 )
    plt.draw()
    plt.title('E field at 2nd lens (click to continue)')
    dummy = plt.ginput(1)
    
    #Propagate to a distant focus...
    efield_lens2_to_focus = efield_lens2_after*optics.curved_wf(npix,dx,200,wavelength_in_mm)
    efield_200mm_focus = optics.fresnel(efield_lens2_to_focus, dx, 200, wavelength_in_mm)
    plt.imshow(np.abs(efield_200mm_focus))
    
    

