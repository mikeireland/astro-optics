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
import utils

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
    # dummy = plt.ginput(1)
    
    # #Propagate to a distant focus...
    # efield_lens2_to_focus = efield_lens2_after*optics.curved_wf(npix,dx,200,wavelength_in_mm)
    # efield_200mm_focus = optics.fresnel(efield_lens2_to_focus, dx, 200, wavelength_in_mm)
    # plt.imshow(np.abs(efield_200mm_focus))
    
    

def generate_atmosphere(npix, wavelength, pixel_size, seeing):
    """ Applies an atmosphere in the form of Kolmogorov turbulence to an initial wavefront and scales
    Parameters
    ----------
    
    npix: integer
        The size of the square of Kolmogorov turbulence generated
    
    wavelength: float
        The wavelength in mm. Amount of atmospheric distortion depends on the wavelength. 
    pixel_size: float
    
    
    Returns
    -------
    
    electric_field: 
    
    """
    
    #!!! MJI This formula was wrong - you used arcsec not radians
    seeing_in_radians = np.radians(seeing/3600.)
    # Generate the Kolmogorov turbulence
    turbulence = optics.kmf(npix)
    
    # Calculate r0 (Fried's parameter), which is a measure of the strength of seeing distortions
    #!!! MJI This formula was wrong
    r0 = 0.98 * wavelength / seeing_in_radians 
    
    # Apply the atmosphere and scale
    wf_in_radians = turbulence * np.sqrt(6.88*(pixel_size/r0)**(5.0/3.0))
        
    # Convert the wavefront to an electric field
    electric_field = np.exp(1j * wf_in_radians)
    
    return wf_in_radians # electric_field
    
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

    
class PIAA_System:
    """A class to model the behaviour of the PIAA system, including all lenses, propagation and fibre fitting.
    """
    def __init__(self):
        """Constructs the class with the following list of default parameters.
        """        
        self.alpha = 2.0                         # Gaussian exponent term
        self.r0 = 0.40                           # Fractional size of secondary mirror/obstruction
        self.frac_to_focus = 1e-6                # Fraction (z_s2 - z_s1)/(z_f - z_s1)
        self.delta = 1e-2                        # Radius of annular dead zone for second pupil
        self.dt = 1e-3                           # Integration step size
        self.n_med = 1.5                         # Refraction index of the medium
        self.thickness = 15.0                    # Physical thickness between the PIAA lenses
        self.radius_in_mm = 1.0                  # Physical radius
        self.telescope_magnification = 250./2.0  #Telescope magnification prior to the exit pupil plane
        self.seeing_in_arcsec = 1.0              #Seeing in arcseconds before magnification
        self.real_heights = True                 # ...
        self.dx = 5.0/1000.0                     # Resolution/sampling in mm/pixel
        self.npix = 1024                         # Number of pixels for the simulation
        self.wavelength_in_mm = 0.5/1000.0       # Wavelength of light in mm
        self.focal_length_1 = 200                # Focal length of wavefront after PIAA lens #2
        self.focal_length_2 = 3.02               # Focal length of 3.02mm lens
        self.focal_length_3 = 4.62               # Focal length of 4.64mm square lenslet
        self.lenslet_width = 1.0                 # Width of the square lenslet in mm
        self.fibre_radius = 0.0017               # Width of the optical fibre in mm
        
        
        
    def create_piaa(self):
        """Creates the PIAA lenses using the stored physical parameters
        """ 
        self.piaa_lens1, self.piaa_lens2 = create_piaa_lenses(self.alpha, self.r0, self.frac_to_focus, self.delta, self.dt, self.n_med, self.thickness, self.radius_in_mm, self.real_heights, self.dx, self.npix, self.wavelength_in_mm)
 
    def create_annulus(self):
        """Compute the input electric field (annulus x distorted wavefront)
        """ 
        self.annulus = (optics.circle(self.npix,(self.radius_in_mm * 2)/self.dx) - optics.circle(self.npix, (self.radius_in_mm * 2 * self.r0)/self.dx))
        
    def calculate_input_e_field(self):
        """Compute the input electric field using the annulus and Kolmogorov turbulence
        
        Returns
        -------
        efield_in: 2D numpy.ndarray
            The electric field.
        """
        self.efield_in = self.annulus * np.exp(1j * generate_atmosphere(self.npix, self.wavelength_in_mm, self.dx, self.seeing_in_arcsec * self.telescope_magnification))
    
        return self.efield_in
        
    def apply_piaa_lens(self, piaa_lens, efield_before):
        """Pass an electric field through a PIAA lens
        
        Parameters
        ----------
        piaa_lens: 2D numpy.ndarray
            The PIAA lens to pass the electric field through.
        efield_before: 2D numpy.ndarray
            The electric field before the PIAA lens   
            
        Returns
        -------
        efield_after: 2D numpy.ndarray
            The electric field after the PIAA lens
        """
        efield_after = efield_before * np.exp(1j * piaa_lens)
        
        return efield_after
        
    def curved_wavefront(self, efield_before, f):
        """Convert the electric field into a curved wavefront
        
        Parameters
        ----------
        efield_before: 2D numpy.ndarray
            The electric field before the curved wavefront    
        f: float    
            The focal length in mm of the wavefront
            
        Returns
        -------
        efield_after: 2D numpy.ndarray
            The electric field after the PIAA lens
        """    
        # Propagate the electric field to a distant focus
        efield_after = efield_before * optics.curved_wf(self.npix, self.dx, f, self.wavelength_in_mm)
        
        return efield_after
    
    def propagate(self, efield_before, distance):
        """Propagate the electric field a certain distance
        
        Parameters
        ----------
        efield_before: 2D numpy.ndarray
            The electric field before being propagated  
        distance: integer
            The distance to propagate the electric field
        Returns
        -------
        efield_after: 2D numpy.ndarray
            The electric field after the propagation
        """            
        efield_after = optics.fresnel(efield_before, self.dx, distance, self.wavelength_in_mm)
        
        return efield_after
        
    def apply_lenslet(self, efield_before):
        """Propagate the electric field through a square lenslet 
        
        Parameters
        ----------
        efield_before: 2D numpy.ndarray
            The electric field before the lenslet
       
       Returns
        -------
        efield_after: 2D numpy.ndarray
            The electric field after the lenslet
        """            
        efield_after = optics.square(self.npix, self.lenslet_width/self.dx) * efield_before
        
        return efield_after
    
def propagate_and_save(directory, distance_step):
    """Creates and saves plots for the PIAA system in increments of the specified distance step.
    
    Parameters
    ----------
    directory: string
        The directory to save the plots to.
    distance_step: integer
        The distance to propagate the wavefront forward for each plot
        Should be a fraction of the thickness and focal length
    """
    
    # Create the lens and its annulus and PIAA components
    lens = PIAA_System()
    lens.create_piaa()
    lens.create_annulus()
    lens.create_piaa()
    
    # The image count to uniquely identify each image created
    file_number = 0
    
    # Compute the input electric field (annulus x distorted wavefront)
    electric_field = lens.calculate_input_e_field()
    
    # Save the initial plot
    utils.save_plot(np.abs(electric_field)**0.5, 'Electric Field: Input', directory, ("%05d" % file_number))
    file_number += 1
    
    # Pass the electric field through the first PIAA lens
    electric_field = lens.apply_piaa_lens(lens.piaa_lens1, electric_field)
    
    #Propagate the electric field through glass to the second lens
    for step in xrange(1, int(lens.thickness) + 1, distance_step):
        # Propagate the field by the distance step
        electric_field = lens.propagate(electric_field, distance_step / lens.n_med)
        
        # Save the plot at each step
        title = 'Electric Field: ' + str(distance_step * step) + ' / ' + str(lens.thickness) + " mm from PIAA Lens #1"
        utils.save_plot(np.abs(electric_field)**0.5, title, directory, ("%05d" % file_number))
        file_number += 1
    
    # Pass the electric field through the second lens
    electric_field = lens.apply_piaa_lens(lens.piaa_lens2, electric_field)
    
    # Propagate the electric field to a distant focus
    electric_field = lens.curved_wavefront(electric_field, lens.focal_length_1)
    
    #Propagate the electric field to focus beyond the second lens
    for step in xrange(1, int(lens.focal_length) + 1, distance_step):
        # Propagate the field by the distance step
        electric_field = lens.propagate(electric_field, distance_step)
        
        # Save the plot at each step
        title = 'Electric Field: ' + str(distance_step * step) + ' / ' + str(lens.focal_length) + " mm from Focus After PIAA Lens #2" 
        utils.save_plot(np.abs(electric_field)**0.5, title, directory, ("%05d" % file_number))
        file_number += 1
    
    
    
        
def propagate_to_fibre(dz, alpha):
    """Propagates the wavefront to the fibre, passing through the following stages:
        1 - Apply atmospheric turbulence
        2 - Pass the wavefront through the telescope pupil (with annulus)
        3 - Apply PIAA lens #1
        4 - Propagate through the intervening glass to PIAA lens #2 (15mm)
        5 - Apply PIAA lens #2
        6 - Propagate to focus (200mm) and then propagate further (3.02mm + dz) to image on the 3.02mm lens
        7 - Apply the 3.02mm lens
        8 - Propagate to the 1mm square lenslet (distance given by thin film equation, ~3.02^2/dz mm)
        9 - Apply the 1mm square lenslet
        10 - Propagate to the fibre (4.64mm + alpha)
        11 - Compute the coupling to the fibre
    
    Parameters
    ----------
    dz: float   
        Free parameter, determines the magnification of the lenslet
    alpha: float
        Free parameter for propagation from the lenslet to the fibre
    """
    
    # Create the lens and its annulus and PIAA components
    lens = PIAA_System()
    lens.create_piaa()
    lens.create_annulus()
    lens.create_piaa()
    
    # [1, 2] - Compute the input electric field (annulus x distorted wavefront from atmosphere) 
    electric_field = lens.calculate_input_e_field()
    
    # [3] - Pass the electric field through the first PIAA lens
    electric_field = lens.apply_piaa_lens(lens.piaa_lens1, electric_field)
    
    # [4] - Propagate the electric field through glass to the second lens
    electric_field = lens.propagate(electric_field, lens.thickness / lens.n_med)
    
    # [5] - Pass the electric field through the second PIAA lens
    electric_field = lens.apply_piaa_lens(lens.piaa_lens2, electric_field)
    electric_field = lens.curved_wavefront(electric_field, lens.focal_length_1)   
    
    # [6] - Propagate the electric field to the 3.02mm lens
    distance_to_lens = lens.focal_length_1 + lens.focal_length_2 + dz
    electric_field = lens.propagate(electric_field, distance_to_lens)
    
    # [7] - Pass the electric field through the 3.02mm lens
    electric_field = lens.curved_wavefront(electric_field, lens.focal_length_2)    
    
    # [8] - Propagate to the 1mm square lenslet
    distance_to_lenslet = 1 / ( 1/lens.focal_length_2 - 1/(lens.focal_length_2 + dz) )  #From thin lens formula
    electric_field = lens.propagate(electric_field, distance_to_lenslet)    
    
    # [9] - Multiply by square lenslet, propagate through glass (forget for the moment) and multiply by curved wf
    electric_field = lens.apply_lenslet(electric_field)
    electric_field = lens.curved_wavefront(electric_field, lens.focal_length_3)   
    
    # [10] - Propagate to the fibre optic cable
    distance_to_fibre = lens.focal_length_3 + alpha
    electric_field = lens.propagate(electric_field, distance_to_fibre)     
    
    # [11] - Compute the near field profile and the fibre coupling
    return electric_field
    