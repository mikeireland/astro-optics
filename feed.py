"""
"""
import numpy as np
import optics
import optics_tools
import utils
import pylab as pl
import time
import multiprocessing as mp
import csv

class Feed:
    """A class to model the behaviour of the RHEA fibre feed and simulate it in operation
    """
    def __init__(self, stromlo_variables=True, use_piaa=True, use_microlens_array=True):
        """Constructs the class with the following list of default parameters.
        """        
        self.alpha = 2.0                         # Gaussian exponent term
        self.r0 = 0.40                           # Fractional size of secondary mirror/obstruction
        self.frac_to_focus = 1e-6                # Fraction (z_s2 - z_s1)/(z_f - z_s1)
        self.delta = 1e-2                        # Radius of annular dead zone for second pupil
        self.dt = 1e-3                           # Integration step size
        self.n_med = 1.5                         # Refraction index of the medium
        self.thickness = 15.0                    # Physical thickness between the PIAA lenses
        self.piaa_radius_in_mm = 1.0             # Physical radius
        self.telescope_magnification = 250./2.0  #Telescope magnification prior to the exit pupil plane
        self.seeing_in_arcsec = 1.0              #Seeing in arcseconds before magnification
        self.real_heights = True                 # ...
        self.dx = 2.0/1000.0                     # Resolution/sampling in mm/pixel
        self.npix = 2048                         # Number of pixels for the simulation
        self.focal_length_1 = 94                 # Focal length of wavefront after PIAA lens #2
        self.focal_length_2 = 3.02               # Focal length of 3.02mm lens
        self.focal_length_3 = 4.64               # Focal length of 4.64mm square lenslet
        self.circular_lens_diameter = 1.0        # Diameter of the 3.02mm circular lens
        self.lenslet_width = 1.0                 # Width of the square lenslet in mm
        self.numerical_aperture = 0.13           # Numerical aperture of the optical fibre 
        self.mode = None
        
        # Stromlo/Subaru value set
        if stromlo_variables:
            self.wavelength_in_mm = 0.52/1000.0    # Wavelength of light in mm (0.52um for Stromlo)
            self.fibre_core_radius = 0.00125       # Width of the optical fibre in mm (0.00125mm for Stromlo)
        else:
            self.wavelength_in_mm = 0.7/1000.0   # Wavelength of light in mm (0.7um for Subaru)
            self.fibre_core_radius = 0.00175     # Width of the optical fibre in mm (0.00175mm for Subaru)
        
        # Boolean variables determining feed layout
        self.use_piaa = use_piaa
        self.use_microlens_array = use_microlens_array
        
    def create_optical_elements(self):
        """
        """
        # Create telescope pupil
        r_pupil = (self.piaa_radius_in_mm * 2)/self.dx
        r_obstruction = (self.piaa_radius_in_mm * 2 * self.r0)/self.dx
        self.telescope_pupil = utils.annulus(self.npix, r_pupil, r_obstruction)
        
        # Create PIAA
        if self.use_piaa:
            self.piaa_optics = optics.PIAAOptics(self.alpha, self.r0, self.frac_to_focus, self.delta, self.dt, self.n_med, self.thickness, self.piaa_radius_in_mm, self.real_heights, self.dx, self.npix, self.wavelength_in_mm)
            
        # Create lens #1
        self.lens_1 = optics.CircularLens(self.focal_length_1, self.piaa_radius_in_mm * 2)
        
        # Create lens #2
        self.lens_2 = optics.CircularLens(self.focal_length_2, self.circular_lens_diameter)
        
        # Create microlens array
        if self.use_microlens_array:
            self.microlens_array = optics.MicrolensArray_3x3(self.focal_length_3, self.lenslet_width)
            
    def propagate_to_fibre(self, dz, offset, turbulence, seeing=1.0, alpha=2.0, use_tip_tilt=True):
        """
        """
        # Initialise seeing and alpha
        self.seeing_in_arcsec = seeing
        self.alpha = alpha
        
        # Create optics
        self.create_optical_elements()
        
        # Initialise list to store electric fields from each step
        #print "Seeing:", self.seeing_in_arcsec
        electric_field= []
        electric_field.append(self.telescope_pupil)

        # Apply tip/tilt
        if use_tip_tilt:
            #print "Applying tip/tilt"
            turbulence = optics_tools.calculate_tip_tilt(turbulence, self.telescope_pupil, self.npix)
        
        # Apply effect of turbulence at telescope pupil (phase distortions)
        tef = optics_tools.apply_and_scale_turbulent_ef(turbulence, self.npix, self.wavelength_in_mm, self.dx, self.seeing_in_arcsec * self.telescope_magnification)
        #print "Passing through turbulence"
        electric_field.append( (self.telescope_pupil * tef) )
        
        # Apply PIAA
        if self.use_piaa:
            #print "Applying PIAA"
            electric_field.append( self.piaa_optics.apply(electric_field[-1]) )
        
        # Apply Lens #1
        #print "Applying lens #1"
        electric_field.append( self.lens_1.apply(electric_field[-1], self.npix, self.dx, self.wavelength_in_mm) )
        
        # Propagate to lens #2
        #print "Propagating to lens #2"
        distance_to_lens = self.focal_length_1 + self.focal_length_2 + dz
        electric_field.append( optics_tools.propagate_by_fresnel(electric_field[-1], self.dx, distance_to_lens, self.wavelength_in_mm) )
        
        # Apply Lens #2
        #print "Applying lens #2"
        electric_field.append( self.lens_2.apply(electric_field[-1], self.npix, self.dx, self.wavelength_in_mm) )
        
        # Propagate to microlens array
        #print "Propagating to microlens array"
        distance_to_microlens_array = 1 / ( 1/self.focal_length_2 - 1/(self.focal_length_2 + dz) )  #From thin lens formula
        electric_field.append( optics_tools.propagate_by_fresnel(electric_field[-1], self.dx, distance_to_microlens_array, self.wavelength_in_mm) )
        
        # Interpolate by 2x and update dx for future use
        #print "Interpolate"
        electric_field.append( utils.interpolate_by_2x(electric_field[-1], self.npix) )
        new_dx =  self.dx / 2.0
        new_npix = self.npix * 2.0
        
        # Compute Fibre mode
        #print "Compute mode"
        self.fibre_mode = optics_tools.calculate_fibre_mode(self.wavelength_in_mm, self.fibre_core_radius, self.numerical_aperture, new_npix, new_dx)
        
        # Apply microlens array, propagate to fibre and compute coupling
        #print "Apply microlens array and couple"
        distance_to_fibre = 1.0/(1.0/self.focal_length_3 - 1.0/(distance_to_microlens_array - self.focal_length_2))
        input_field = np.sum(np.abs(electric_field[0])**2)
        ef_at_fibre, coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9 = self.microlens_array.apply_propagate_and_couple(electric_field[-1], new_npix, new_dx, self.wavelength_in_mm, distance_to_fibre, input_field, offset, self.fibre_mode)
        electric_field.append(ef_at_fibre)
        
        # All finished, return
        return coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9, electric_field, turbulence, tef

def test(seeing,  use_piaa=True, use_tip_tilt=True):
    tm = []
    tm.append( (time.time(), "Start") )
    turbulence = optics_tools.kmf(2048)
    rhea_feed = Feed(use_piaa=use_piaa)
    c, a, eta, ef, t, tef = rhea_feed.propagate_to_fibre(0.152, 84, turbulence, seeing=seeing, use_tip_tilt=use_tip_tilt)
    tm.append( (time.time(), "End") )
    print "Total time: ", tm[-1][0] - tm[0][0] 
    return c, a, eta, ef, turbulence, t, tef