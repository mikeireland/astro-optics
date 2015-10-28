"""Models a the RHEA fibre feed composed of OpticalElements.

Authors:
Adam Rains
"""
import numpy as np
import optics
import opticstools as optics_tools
import opticstools.utils as utils
import pylab as pl
import time
import multiprocessing as mp
import csv

class Feed:
    """A class to model and simulate the behaviour of the RHEA fibre feed 
    
    TODO: Make Feed the base class and RHEA Feed a specific instance of it
    """
    def __init__(self, stromlo_variables=True, use_piaa=True, 
                 use_microlens_array=True):
        """Constructs the class with the following list of default parameters.
        """        
        self.alpha = 2.0            # Gaussian exponent term
        self.r0 = 0.40              # Fractional size of secondary mirror
        self.n_med = 1.5            # Refraction index of the medium
        self.thickness = 15.0       # Physical thickness between the PIAA lenses
        self.piaa_r_in_mm = 1.0     # Physical radius
        self.tele_mag = 250./2.0    # Telescope mag prior to exit pupil plane
        self.seeing_in_arcsec = 1.0 # Seeing in arcseconds before magnification
        self.real_heights = True    # ...
        self.dx = 2.0/1000.0        # Resolution/sampling in mm/pixel
        self.npix = 2048            # Number of pixels for the simulation
        self.focal_length_1 = 94    # Focal length of wf before PIAA lens #1
        self.focal_length_2 = 3.02  # Focal length of 3.02mm lens
        self.focal_length_3 = 4.64  # Focal length of 4.64mm square lenslet
        self.lens_2_diameter = 0.8  # Diameter  3.02mm circular lens. 
        self.lenslet_width = 1.0    # Width of the square lenslet in mm
        self.num_aperture = 0.13    # Numerical aperture of the optical fibre 
        self.mode = None
        self.frac_to_focus = self.thickness / self.focal_length_1
        
        # Stromlo/Subaru value set
        if stromlo_variables:
            self.wavelength_in_mm = 0.52/1000.0  # Wavelength of light in mm 
            self.fibre_core_radius = 0.00125     # Width of fibre in mm 
        else:
            self.wavelength_in_mm = 0.7/1000.0   # Wavelength of light in mm 
            self.fibre_core_radius = 0.00175     # Width of fibre in mm 
        
        # Boolean variables determining feed layout
        self.use_piaa = use_piaa
        self.use_microlens_array = use_microlens_array
        
    def create_optical_elements(self):
        """Creates each of the OpticalElements composing the RHEA feed"""
        # Create telescope pupil
        r_pupil = (self.piaa_r_in_mm * 2)/self.dx
        r_obstruction = (self.piaa_r_in_mm * 2 * self.r0)/self.dx
        self.telescope_pupil = utils.annulus(self.npix, r_pupil, r_obstruction)
        
        # Create PIAA
        if self.use_piaa:
            self.piaa_optics = optics.PIAAOptics(self.alpha, self.r0, 
                                                 self.frac_to_focus, self.n_med, 
                                                 self.thickness, 
                                                 self.piaa_r_in_mm, 
                                                 self.real_heights, self.dx, 
                                                 self.npix, 
                                                 self.wavelength_in_mm)
            
        # Create lens #1
        self.lens_1 = optics.CircularLens(self.focal_length_1, 
                                          self.piaa_r_in_mm * 2)
        
        # Create lens #2
        self.lens_2 = optics.CircularLens(self.focal_length_2, 
                                          self.lens_2_diameter)
        
        # Create microlens array
        if self.use_microlens_array:
            self.m_array = optics.MicrolensArray_3x3(self.focal_length_3
                                                           , self.lenslet_width)
            
    def propagate_to_fibre(self, dz, offset, turbulence, seeing, alpha=2.0, 
                           use_tip_tilt=True):
        """Simulate light propagating through a turbulent atmosphere and through 
        the entire fibre feed, before being coupled to optical fibre/s
        
        Parameters
        ----------
        dz: float   
            Free parameter, determines the magnification of the microlens array 
            (changed to optimise coupling)
        offset: int
            The x/y distance that each of the outer fibres is off-centre in a 
            radially outwards direction
        seeing: float
            The seeing for the telescope in arcseconds.
        gaussian_alpha: float
            Exponent constant from the Gaussian distribution achieved by the 
            PIAA optics    
        use_tip_tilt: boolean
            Whether to apply Tip/Tilt correction to the incident wavefront
            
        Returns
        -------

        coupling_1_to_9: [float, float, float]
            The coupling of the wave at the fibre plane with the fibre mode for 
            c1 (the central fibre), c5 (the central row and column) and c9 
            (entire 3x3 array)
        aperture_loss_1_to_9
            The aperture loss of the wave at the fibre plane with the fibre mode 
            for a1 (the central fibre), a5 (the central row and column) and a9 
            (entire 3x3 array)
        eta_1_5_9    
            The efficiency/throughput (eta = coupling * aperture loss) of the
            wave at the fibre plane with the fibre mode for a1 (the central 
            fibre), a5 (the central row and column) and a9 (entire 3x3 array).  
        electric_field: np.array([[[...]...]...])
            Array of the electric field at various points throughout the length 
            of the feed.
        turbulence: np.array([[...]...])
            The turbulence that was applied to the wave initially, which may or 
            may not be Tip/Tilt corrected
        tef: np.array([[...]...])
            The phase aberrations introduced by the turbulence
        """
        # Initialise seeing and alpha
        self.seeing_in_arcsec = seeing
        self.alpha = alpha
        
        # Create optics
        self.create_optical_elements()
        
        # Initialise list to store electric fields from each step
        ef = []
        ef.append(self.telescope_pupil)

        # Apply tip/tilt (Only if seeing > 0.0)
        if use_tip_tilt and seeing > 0.0:
            turbulence = optics_tools.correct_tip_tilt(turbulence, 
                                                       self.telescope_pupil, 
                                                       self.npix)
        
        # Apply effect of turbulence at telescope pupil (phase distortions)
        tef = optics_tools.apply_and_scale_turbulent_ef(turbulence, self.npix, 
                                                        self.wavelength_in_mm, 
                                                        self.dx, 
                                                        self.seeing_in_arcsec * 
                                                        self.tele_mag)
                                                        
        ef.append( (self.telescope_pupil * tef) )
        
        # Apply Lens #1
        ef.append(self.lens_1.apply(ef[-1], self.npix, self.dx, 
                                    self.wavelength_in_mm))     
        
        # Apply PIAA
        if self.use_piaa:
            ef.append( self.piaa_optics.apply(ef[-1]) )
            distance_to_lens = (self.focal_length_1 - self.thickness + 
                                self.focal_length_2 + dz)
        else:
            distance_to_lens = self.focal_length_1 + self.focal_length_2 + dz

        # Propagate to lens #2
        ef.append(optics_tools.propagate_by_fresnel(ef[-1], self.dx, 
                                                    distance_to_lens, 
                                                    self.wavelength_in_mm) )
        
        # Apply Lens #2
        ef.append(self.lens_2.apply(ef[-1], self.npix, self.dx, 
                  self.wavelength_in_mm) )
        
        # Propagate to microlens array
        distance_to_microlens_array = (1 / ( 1/self.focal_length_2 - 
                                       1/(self.focal_length_2 + dz) )) 
                                       #From thin lens formula
        ef.append(optics_tools.propagate_by_fresnel(ef[-1], self.dx, 
                                                    distance_to_microlens_array, 
                                                    self.wavelength_in_mm) )
              
        # Interpolate by 2x and update dx for future use
        ef.append( utils.interpolate_by_2x(ef[-1], self.npix) )
        new_dx =  self.dx / 2.0
        new_npix = self.npix * 2.0
        
        # For the purpose of having an image of the light at the lenslet 
        # (This presently notably slows down the code though)
        #ef.append(self.m_array.apply( ef[-1], new_npix, new_dx, 
                  #self.wavelength_in_mm) )
        
        # Compute Fibre mode
        self.mode = optics_tools.calculate_fibre_mode(self.wavelength_in_mm, 
                                                      self.fibre_core_radius, 
                                                      self.num_aperture, 
                                                      new_npix, new_dx)
        
        # Apply microlens array, propagate to fibre and compute coupling
        distance_to_fibre = (1.0/(1.0/self.focal_length_3 - 
                             1.0/(distance_to_microlens_array - 
                                  self.focal_length_2)))
        input_field = np.sum(np.abs(ef[0])**2)
        ef_at_fibre, coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9 = \
            self.m_array.apply_propagate_and_couple(ef[-1], new_npix, new_dx, 
                                                    self.wavelength_in_mm, 
                                                    distance_to_fibre, 
                                                    input_field, offset, 
                                                    self.mode)
        ef.append(ef_at_fibre)
        
        # All finished, return
        return coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9, ef, turbulence

def test(dz=0.152, offset=84, seeing=0.0, alpha=2.0, use_piaa=True, 
         use_tip_tilt=True):
    """Method testing the functionality of propagate_to_fibre"""
    tm = []
    tm.append( (time.time(), "Start") )
    turbulence = optics_tools.kmf(2048)
    rhea_feed = Feed(use_piaa=use_piaa)
    c, a, eta, ef, t = rhea_feed.propagate_to_fibre(dz, offset, turbulence, 
                                                    seeing=seeing, alpha=alpha, 
                                                    use_tip_tilt=use_tip_tilt)
    tm.append( (time.time(), "End") )
    print "Total time: ", tm[-1][0] - tm[0][0] 
    return c, a, eta, ef, t
