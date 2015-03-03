"""Phase induced amplitude apodisation (PIAA) is the technique of apodising a pupil
without loss, so that the beam better matches e.g. a coronagraph or a fiber input.
This module contains a collection of routines to help with PIAA calculations

e.g.:

(r_surface,surface1,surface2,r_slope,slope_1,slope_2) = piaa.s_annulus_gauss(2.0, 0.35, 0.01)
plot(r_surface, surface1)
plot(r_surface, surface2)
"""

import numpy as np
from scipy import integrate

def ds_annulus_gauss(r, s, alpha, r0):
    """An annulus morphing into a (truncated) Gaussian. Only
    the first slope is given. The equations are:
    I_1 = exp(-alpha(r+s)**2) / (pi/alpha [1-exp(-alpha)])
    I_0 = 1.0 / [pi (1-r_0^2)] for r0<r<1
    
    With a second radial co-ordinate u, we have:
    u = r + s
    s'(r) = [ I_0 * x / I_1 * y ]  -  1
    
    The second slope is found directly
    from interpolation. For a second radial co-ordinate u and a
    second slope v, we have: 
    
    v(u) = -s(r), with
    u(r) = r + s(r)
    """
    
    return (1 - np.exp(-alpha)) * np.exp(alpha*(r + s)**2) * r / alpha / (1-r0**2) / (r+s) - 1
    
def s_annulus_gauss(alpha,r0,frac_to_focus,delta=1e-2,dt=1e-3, n_med=1.5, thickness=15.0, radius_in_mm=1.0, real_heights=True):
    """Everything is scaled so that a slope of 1 unit corresponds to an angle
    of 1 radius per length. To convert to real units of slope, this has to be multiplied
    by (radius/(thickness / n_med)). To convert to real heights, we then have to divide 
    by (n-1).
    
    Parameters
    ----------
    alpha: The 
    frac_to_focus: float
        The fractional distance of the distance between surfaces and the first surface
        to focus
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
    aa = integrate.ode(ds_annulus_gauss).set_integrator('vode', method='bdf', with_jacobian=False)
    aa.set_initial_value(-r0 + delta, r0).set_f_params( alpha,r0 )
    rs = np.array([])
    ss = np.array([])
    while aa.successful() and aa.t < 1.0:
        aa.integrate(aa.t+dt)
        rs = np.append(rs,aa.t)
        ss = np.append(ss,aa.y)
    us = np.linspace(0,1,1.0/dt + 1)
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