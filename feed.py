"""Models a the RHEA fibre feed composed of OpticalElements.

Authors:
Adam Rains
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
    
    TODO: Make Feed the base class and RHEA Feed a specific instance of it
    """
    def __init__(self, stromlo_variables=True, use_piaa=True, use_microlens_array=True, n=1.0):
        """Constructs the class with the following list of default parameters.
        """        
        self.alpha = 2.0                         # Gaussian exponent term
        self.r0 = 0.40                           # Fractional size of secondary mirror/obstruction
        self.delta = 1e-2                        # Radius of annular dead zone for second pupil
        self.dt = 1e-3                           # Integration step size
        self.n_med = 1.5                         # Refraction index of the medium
        self.thickness = 15.0                    # Physical thickness between the PIAA lenses
        self.piaa_radius_in_mm = 1.0             # Physical radius
        self.telescope_magnification = 250./2.0  #Telescope magnification prior to the exit pupil plane
        self.seeing_in_arcsec = 1.0              #Seeing in arcseconds before magnification
        self.real_heights = True                 # ...
        self.dx = 2.0/1000.0                     # Resolution/sampling in mm/pixel
        self.npix = 4096                         # Number of pixels for the simulation
        self.focal_length_1 = 94                 # Focal length of wavefront before PIAA lens #1
        self.focal_length_2 = 3.02               # Focal length of 3.02mm lens
        self.focal_length_3 = 4.64               # Focal length of 4.64mm square lenslet
        self.circular_lens_diameter = 0.8        # Diameter of the 3.02mm circular lens. Maybe make it small due to glue etc? Should be 1mm
        self.lenslet_width = 1.0                 # Width of the square lenslet in mm
        self.numerical_aperture = 0.13           # Numerical aperture of the optical fibre 
        self.mode = None
        
        #self.frac_to_focus = 1e-6                # Fraction (z_s2 - z_s1)/(z_f - z_s1)
        self.frac_to_focus = n * self.thickness / (self.thickness + self.focal_length_1)
        
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
        """Creates each of the OpticalElements composing the RHEA feed"""
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
        """Simulate light propagating through a turbulent atmosphere and through the entire fibre feed, before being coupled to optical fibre/s
        
        Parameters
        ----------
        dz: float   
            Free parameter, determines the magnification of the microlens array (changed to optimise coupling)
        offset: int
            The x/y distance that each of the outer fibres is off-centre in a radially outwards direction
        seeing: float
            The seeing for the telescope in arcseconds.
        gaussian_alpha: float
            Exponent constant from the Gaussian distribution achieved by the PIAA optics    
        use_tip_tilt: boolean
            Whether to apply Tip/Tilt correction to the incident wavefront
            
        Returns
        -------

        coupling_1_to_9: [float, float, float]
            The coupling of the wave at the fibre plane with the fibre mode for c1 (the central fibre), c5 (the central row and column) and c9 (entire 3x3 array)
        aperture_loss_1_to_9
            The aperture loss of the wave at the fibre plane with the fibre mode for a1 (the central fibre), a5 (the central row and column) and a9 (entire 3x3 array)
        eta_1_5_9    
            The efficiency/throughput (eta = coupling * aperture loss) of the wave at the fibre plane with the fibre mode for a1 (the central fibre), a5 (the central row and column) and a9 (entire 3x3 array).  
        electric_field: np.array([[[...]...]...])
            Array of the electric field at various points throughout the length of the feed.
        turbulence: np.array([[...]...])
            The turbulence that was applied to the wave initially, which may or may not be Tip/Tilt corrected
        tef: np.array([[...]...])
            The phase aberrations introduced by the turbulence
        """
        # Initialise seeing and alpha
        self.seeing_in_arcsec = seeing
        self.alpha = alpha
        
        # Create optics
        self.create_optical_elements()
        
        # Initialise list to store electric fields from each step
        electric_field= []
        electric_field.append(self.telescope_pupil)

        # Apply tip/tilt (Only if seeing > 0.0)
        if use_tip_tilt and seeing > 0.0:
            turbulence = optics_tools.correct_tip_tilt(turbulence, self.telescope_pupil, self.npix)
        
        # Apply effect of turbulence at telescope pupil (phase distortions)
        tef = optics_tools.apply_and_scale_turbulent_ef(turbulence, self.npix, self.wavelength_in_mm, self.dx, self.seeing_in_arcsec * self.telescope_magnification)
        electric_field.append( (self.telescope_pupil * tef) )
        
        # Apply Lens #1
        electric_field.append( self.lens_1.apply(electric_field[-1], self.npix, self.dx, self.wavelength_in_mm) )     
        
        # Apply PIAA
        if self.use_piaa:
            electric_field.append( self.piaa_optics.apply(electric_field[-1]) )
            distance_to_lens = self.focal_length_1 - self.thickness + self.focal_length_2 + dz
        else:
            distance_to_lens = self.focal_length_1 + self.focal_length_2 + dz

        # Propagate to lens #2
        electric_field.append( optics_tools.propagate_by_fresnel(electric_field[-1], self.dx, distance_to_lens, self.wavelength_in_mm) )
        
        """
        distance_step = 1.0
        remaining_distance_to_lens = distance_to_lens
        file_number = 0
        while remaining_distance_to_lens > 0:
            # Propagate the field by the distance step (or the distance remaining if it is smaller than the step)
            if remaining_distance_to_lens > distance_step:
                electric_field[-1] = optics_tools.propagate_by_fresnel(electric_field[-1], self.dx, distance_step, self.wavelength_in_mm)
                remaining_distance_to_lens -= distance_step   
            else:
                electric_field[-1] = optics_tools.propagate_by_fresnel(electric_field[-1], self.dx, remaining_distance_to_lens, self.wavelength_in_mm)
                remaining_distance_to_lens = 0
            
            # Save the plot at each step
            title = str(distance_to_lens - remaining_distance_to_lens) + ' of ' + str(distance_to_lens) + " mm from 3.02mm Lens After PIAA Lens #2"
            utils.save_plot(np.abs(electric_field[-1])**0.5, title, "C:\\Users\\Adam Rains\\Dropbox\\University\\2015\\ASTR8010\\Code\\astro-optics\\Saves\\Layout\\", ("%05d" % file_number))
            file_number += 1  
        """ 
        # Apply Lens #2
        electric_field.append( self.lens_2.apply(electric_field[-1], self.npix, self.dx, self.wavelength_in_mm) )
        
        # Propagate to microlens array
        distance_to_microlens_array = 1 / ( 1/self.focal_length_2 - 1/(self.focal_length_2 + dz) ) #-30.0 #From thin lens formula
        print distance_to_microlens_array
        electric_field.append( optics_tools.propagate_by_fresnel(electric_field[-1], self.dx, distance_to_microlens_array, self.wavelength_in_mm) )
              
        # Interpolate by 2x and update dx for future use
        electric_field.append( utils.interpolate_by_2x(electric_field[-1], self.npix) )
        new_dx =  self.dx / 2.0
        new_npix = self.npix * 2.0
        
        # For the purpose of having an image of the light at the lenslet
        electric_field.append( self.microlens_array.apply( electric_field[-1], new_npix, new_dx, self.wavelength_in_mm) )
        
        # Compute Fibre mode
        self.fibre_mode = optics_tools.calculate_fibre_mode(self.wavelength_in_mm, self.fibre_core_radius, self.numerical_aperture, new_npix, new_dx)
        
        # Apply microlens array, propagate to fibre and compute coupling
        distance_to_fibre = 1.0/(1.0/self.focal_length_3 - 1.0/(distance_to_microlens_array - self.focal_length_2))
        input_field = np.sum(np.abs(electric_field[0])**2)
        ef_at_fibre, coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9 = self.microlens_array.apply_propagate_and_couple(electric_field[-2], new_npix, new_dx, self.wavelength_in_mm, distance_to_fibre, input_field, offset, self.fibre_mode)
        electric_field.append(ef_at_fibre)
        
        # All finished, return
        return coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9, electric_field, turbulence, tef

def test(dz=0.152, offset=84, seeing=0.0, alpha=2.0, use_piaa=True, use_tip_tilt=True, n=1.0):
    """Method testing the functionality of propagate_to_fibre"""
    tm = []
    tm.append( (time.time(), "Start") )
    turbulence = optics_tools.kmf(2048)
    rhea_feed = Feed(use_piaa=use_piaa, n=n)
    c, a, eta, ef, t, tef = rhea_feed.propagate_to_fibre(dz, offset, turbulence, seeing=seeing, alpha=alpha, use_tip_tilt=use_tip_tilt)
    tm.append( (time.time(), "End") )
    print "Total time: ", tm[-1][0] - tm[0][0] 
    return c, a, eta, ef, t, tef