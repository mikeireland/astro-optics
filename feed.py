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
        #self.focal_length_1 = 94    # Focal length of wf before PIAA lens #1
        #self.focal_length_2 = 3.02  # Focal length of 3.02mm lens
        self.focal_length_3 = 4.64  # Focal length of 4.64mm square lenslet
        self.lens_2_diameter = 0.8  # Diameter  3.02mm circular lens. 
        self.lenslet_width = 1.0    # Width of the square lenslet in mm
        self.num_aperture = 0.13    # Numerical aperture of the optical fibre 
        self.mode = None
        
        
        # Stromlo/Subaru value set
        if stromlo_variables:
            self.wavelength_in_mm = 0.52/1000.0  # Wavelength of light in mm 
            self.fibre_core_radius = 0.00125     # Width of fibre in mm 
            self.focal_length_1 = 94    # Focal length of wf before PIAA lens #1
        else:
            self.wavelength_in_mm = 0.7/1000.0  # Wavelength of light in mm 
            self.fibre_core_radius = 0.00175    # Width of fibre in mm 
            self.focal_length_1 = 2/7.2*300     # 7.2 mm pupil via 300 mm lens
            self.focal_length_2 = 1.45          # Re-focusing lens 2
        
        self.frac_to_focus = self.thickness / self.focal_length_1
        
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
            self.m_array = optics.MicrolensArray_3x3(self.focal_length_3, 
                                                     self.lenslet_width)
            
    def propagate_to_fibre(self, dz, offset, turbulence, seeing, alpha=2.0, 
                           use_tip_tilt=True, do_interpolate=False):
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
        
        print "Distance to lens #2: %0.2f mm" % distance_to_lens
        
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
                                           
        print "Distance to MLA: %0.2f mm" % distance_to_microlens_array    
        
        ef.append(optics_tools.propagate_by_fresnel(ef[-1], self.dx, 
                                                    distance_to_microlens_array, 
                                                    self.wavelength_in_mm) )
              
        # Interpolate by 2x and update dx for future use
        if do_interpolate:
            ef.append( utils.interpolate_by_2x(ef[-1], self.npix) )
            new_dx =  self.dx / 2.0
            new_npix = self.npix * 2.0
        else:
            new_dx =  self.dx
            new_npix = self.npix
        
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
        self.distance_to_fibre = distance_to_fibre                      
                                
        input_field = np.sum(np.abs(ef[0])**2)
        ef_at_fibre, coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9 = \
            self.m_array.apply_propagate_and_couple(ef[-1], new_npix, new_dx, 
                                                    self.wavelength_in_mm, 
                                                    distance_to_fibre, 
                                                    input_field, offset, 
                                                    self.mode)
        ef.append(ef_at_fibre)
        
        print "Distance to fibre: %0.2f mm" % distance_to_fibre
        
        # All finished, return
        return coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9, ef, turbulence

def test(dz=0.152, offset=84, seeing=0.0, alpha=2.0, use_piaa=True, 
         use_tip_tilt=True, stromlo_variables=True):
    """Method testing the functionality of propagate_to_fibre"""
    tm = []
    tm.append( (time.time(), "Start") )
    turbulence = optics_tools.kmf(2048)
    rhea_feed = Feed(use_piaa=use_piaa, stromlo_variables=stromlo_variables)
    c, a, eta, ef, t = rhea_feed.propagate_to_fibre(dz, offset, turbulence, 
                                                    seeing=seeing, alpha=alpha, 
                                                    use_tip_tilt=use_tip_tilt)
    tm.append( (time.time(), "End") )
    print "Total time: ", tm[-1][0] - tm[0][0] 
    return c, a, eta, ef, t
    
    
    
def propagate_subaru():
    """May 2018 Method to accurately compute Subaru cable throughput compared
    to the throughput observed in the lab and at the summit."""
    # Initialise Turbulence  (seeing=0, so not actually applied)
    turbulence = optics_tools.kmf(2048)
    
    # Initialise Feed
    subaru_feed = Feed(use_piaa=False, stromlo_variables=False)
    subaru_feed.npix = 2048 #4096
    subaru_feed.dx = 2.0/1000.0         # Pixel scale in mm/px
    offset = int(0.15 / subaru_feed.dx) # From 1.15 mm pitch mask
    
    # Calculate dz using known fabricated distance to MLA
    d_mla = 36.39
    f_2 = subaru_feed.focal_length_2
    dz = d_mla*f_2 / (d_mla - f_2) - f_2
    
    # Propagate through all optics and couple
    c, a, eta, ef, t = subaru_feed.propagate_to_fibre(dz, offset, turbulence, 
                                                    seeing=0.0, alpha=2.0, 
                                                    use_tip_tilt=False, 
                                                    do_interpolate=False)

    input_field = np.sum(np.abs(ef[0]**2))
    al_train = [np.sum(np.abs(ef_i**2))/input_field for ef_i in ef]
    
    print "\n-----------------\nAperture Losses\n-----------------"
    print "%-22s: %-2.3f" % ("Pupil plane", al_train[0])
    print "%-22s: %-2.3f" % ("Pupil (Turbulent)", al_train[1])
    print "%-22s: %-2.3f" % ("Apply Lens #1", al_train[2])
    print "%-22s: %-2.3f" % ("Surface Lens #2", al_train[3])
    print "%-22s: %-2.3f" % ("Apply Lens #2", al_train[4])
    print "%-22s: %-2.3f" % ("MLA Surface", al_train[5])
    print "%-22s: %-2.3f" % ("Fibre Surface", al_train[6])
    
    print "\n-----------------\nThroughput\n-----------------"
    print "%-10s: %-2.3f" % ("1 fibre", eta[0])
    print "%-10s: %-2.3f" % ("5 fibres", eta[1])
    print "%-10s: %-2.3f" % ("9 fibres", eta[2])
    
    # Now we want to compute the predicted throughput using the *known* fibre
    # positional offsets - i.e. compute the overlap integrals for each fibre,
    # offset from its 'ideal' position by the calculated alignment offset. This
    # should be able to serve as a useful sanity check figure for the 
    # throughputs measured in the lab. It will be critical however to get the 
    # orientation right such that the fibres are offset appropriately.
    #
    # Replacement Subaru cable IFU orientation:
    #  | 1 | 2 | 5 |
    #  | 6 | 9 | 7 |
    #  | 8 | 3 | 4 |
    #
    # Fibre order for coupling:
    #  | 1 | 2 | 3 |
    #  | 4 | 5 | 6 |
    #  | 7 | 8 | 9 |
    #
    # Need to get the alignments from each fibre in the order of the coupling 
    # computation above. Modify the propagation routine to accept additional
    # fibre offsets as an array, and apply each.
    fibre_alignment = np.loadtxt("subaru2_alignment.csv", delimiter=",")
    offset_dict = {}
    
    # Construct dictionary with key being fibre number, and values being x/y
    # offsets in pixels
    for fibre in fibre_alignment:
        offset_dict[int(fibre[0])] = fibre[1:] / 1000 / subaru_feed.dx
        
    subaru_ifu = np.array([[1, 2, 5],
                           [6, 9, 7],
                           [8, 3, 4]])
    #import pdb
    #pdb.set_trace()
    
    ef_real, c_real, al_real, eta_real = \
        subaru_feed.m_array.apply_propagate_and_couple(ef[-2], 
                                            subaru_feed.npix, 
                                            subaru_feed.dx, 
                                            subaru_feed.wavelength_in_mm, 
                                            subaru_feed.distance_to_fibre, 
                                            input_field, offset, 
                                            subaru_feed.mode, 
                                            real_offsets=offset_dict, 
                                            offset_ifu=subaru_ifu, test=True)
    
    print "\n-----------------\nReal Throughput\n-----------------"
    print "%-10s: %-2.3f" % ("1 fibre", eta_real[0])
    print "%-10s: %-2.3f" % ("5 fibres", eta_real[1])
    print "%-10s: %-2.3f" % ("9 fibres", eta_real[2])
    
    return c, a, eta, ef, t