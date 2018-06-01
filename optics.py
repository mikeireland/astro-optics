"""This contains classes modelling optical elements such as lenses.

Each optical element has a focal length and width and implements the method 
"apply", which enables it to modify an incident wave

Authors:
Adam Rains
"""
from __future__ import division, print_function
import numpy as np
import opticstools as optics_tools
import opticstools.utils as utils
import piaa

class OpticalElement:
    """The base class for all optical elements."""
    
    def __init__(self, focal_length, width):
        """Initialisation for an OpticalElement.
        
        Parameters
        ----------
        focal_length: float
            The focal length in mm.
        width: float
            The width in mm.
        """
        self.focal_length = focal_length
        self.width = width
    
    def apply(self, input_ef, npix, dx, wavelength_in_mm):
        """Used to modify the incident wave as it passes through the 
        OpticalElement. The simplest implementation does not modify the incident 
        wave.
        
        Parameters
        ----------
        input_ef: np.array([[...]...])
            2D square incident wave consisting of complex numbers.
        npix: int
            Size of input_wf per side, preferentially a power of two (npix=2**n)
        dx: float
            Resolution of the wave in mm/pixel
        wavelength_in_mm: float
            Wavelength of the wave in mm
            
        Return
        ------
        input_ef: np.array([[...]...])
            Unchanged incident wave.
        """
        return input_ef
        
class CircularLens(OpticalElement):
    """A CircularLens OpticalElement - application of a curved wavefront and 
    circular mask"""
    
    def apply(self, input_ef, npix, dx, wavelength_in_mm):
        """Used to modify the incident wave as it passes through the 
        CircularLens.
        
        Parameters
        ----------
        input_ef: np.array([[...]...])
            2D square incident wave consisting of complex numbers.
        npix: int
            Size of input_wf per side, preferentially a power of two (npix=2**n)
        dx: float
            Resolution of the wave in mm/pixel
        wavelength_in_mm: float
            Wavelength of the wave in mm
            
        Return
        ------
        output_ef: np.array([[...]...])
            The modified wave after application of the curved wavefront and 
            circular mask.
        """
        # Apply a circular mask
        masked_ef = utils.circle(npix, (self.width / dx)) * input_ef
        
        # Apply the curved wavefront of the lens
        curved_wf = optics_tools.curved_wf(npix, dx, self.focal_length, 
                                          wavelength_in_mm)
        output_ef = masked_ef * curved_wf
        
        return output_ef
       
        
class MicrolensArray_3x3(OpticalElement):
    """A 3x3 microlens array OpticalElement - a 3x3 square grid of lenses, each 
    coming to its own focus.
    """
    def apply(self, input_ef, npix, dx, wavelength_in_mm):
        """Applies 3x3 curved wavefronts and square masks to the incident wave.

        Parameters
        ----------
        input_ef: np.array([[...]...])
            2D square incident wave consisting of complex numbers.
        npix: int
            Size of input_wf per side, preferentially a power of two (npix=2**n)
        dx: float
            Resolution of the wave in mm/pixel
        wavelength_in_mm: float
            Wavelength of the wave in mm
            
        Return
        ------
        output_ef: np.array([[...]...])
            The modified wave after application of the 3x3 curved wavefronts and 
            square mask.
        """           
        # Initialise the resultant electric field 
        # (that will be the addition of each of the 9 micro-lenslets)
        output_ef = np.zeros((npix, npix))
        
        # Create the square window that will shifted around to represent the 
        # light getting into each micro-lenslet
        square = utils.square(npix, self.width/dx) + 0j
        
        # Starting in the top left, track the window over each row
        for x in range(-1,2):
            for y in range(-1,2):
                # Calculate the base shift (the actual shift requires the -1, 
                # 0 or 1 multiplier to get direction) 
                shift = int(self.width/dx)
                
                # Have the window be a curved wavefront and shift as required
                curved_wf = optics_tools.curved_wf(npix, dx, self.focal_length, 
                                                   wavelength_in_mm)
                y_shift = np.roll((square*curved_wf), shift*y, axis=1)
                xy_shift = np.roll(y_shift, shift*x, axis=0)
                
                # Add the resulting wavefront to a field containing the 
                # wavefronts of all 9 lenses
                output_ef = output_ef + xy_shift * input_ef
 
        return output_ef
    
    def apply_propagate_and_couple(self, input_ef, npix, dx, wavelength_in_mm, 
                                   distance_to_fibre, input_field, offset, 
                                   fibre_mode, real_offsets=None,
                                   offset_ifu=None, test=False):
        """Applies 3x3 curved wavefronts and square masks to the incident wave.

        Parameters
        ----------
        input_ef: np.array([[...]...])
            2D square incident wave consisting of complex numbers.
        npix: int
            Size of input_wf per side, preferentially a power of two (npix=2**n)
        dx: float
            Resolution of the wave in mm/pixel
        wavelength_in_mm: float
            Wavelength of the wave in mm
        distance_to_fibre: float
            The distance to the optical fibre plane from the front of the 
            microlens array.
        input_field: float
            Sum of the EF at the telescope pupil (used to calculate the 
            throughput and any losses)
        offset: integer
            Number of pixels that the 8 non-central microlenses are offset 
            radially outwards by at the fibre plane.
        fibre_mode: np.array([[...]...])
            2D square mode of the optical fibre, constructed from a combination 
            of Bessel functions
            
        Return
        ------
        output_ef: np.array([[...]...])
            The modified wave after application of the 3x3 curved wavefronts and 
            square mask.
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
        """ 
        npix = int(npix)
        
        # Initialise the resultant electric field (that will be the addition of 
        # each of the 9 micro-lenslets)
        out_ef = np.zeros((npix, npix)) + 0j
        
        # Initialise the displacement of each square from the centre
        shift = int(self.width / dx)
        
        # Create the square window that will be used to represent the light 
        # getting into each micro-lenslet
        square = utils.square(npix, self.width/dx) + 0j
        curved_square = optics_tools.curved_wf(shift, dx, self.focal_length, 
                                               wavelength_in_mm)
        
        # Variable to store the coupling for the central fibre (1), centre and 
        # horizontal/vertical fibres (5) and all, including diagonals (9)
        coupling_1_to_9 = []
        aperture_loss_1_to_9 = []
        eta_1_5_9 = [0,0,0]
        
        # For each of the 9 microlenses 
        for x in range(-1,2):
            for y in range(-1,2):
                # Shift the desired square to the centre
                y_shifted_ef = np.roll(input_ef, -shift*y, axis=1)
                xy_shifted_ef = np.roll(y_shifted_ef, -shift*x, axis=0)
                
                #--------------------------------------------------------------
                # Computing coupling with *real* offsets
                #--------------------------------------------------------------
                #if type(real_offsets) != None and type(offset_ifu) != None:
                if test:
                    #print "Computing coupling with real fibre offsets"
                    

                    
                    fibre_num = int(offset_ifu[x+1, y+1])
                    fibre_x = int(np.round(real_offsets[fibre_num][0]))
                    fibre_y = int(np.round(real_offsets[fibre_num][2]))
                    
                    # Positive shift in y (axis 0) is down
                    # Positive shift in x (axis 1) is right
                    xy_shifted_ef = np.roll(xy_shifted_ef, -fibre_x, axis=1)
                    xy_shifted_ef = np.roll(xy_shifted_ef, -fibre_y, axis=0)
                    
                    #import pdb
                    #pdb.set_trace()
                #--------------------------------------------------------------
                
                # Apply the window and curved wavefront
                sub_ef = xy_shifted_ef[(npix//2 - shift//2):(npix//2 + shift//2),
                                       (npix//2 - shift//2):(npix//2 + shift//2)]
                curved_ef = curved_square * sub_ef
                
                # Propagate the light passing through the microlens to the fibre
                ef_at_fibre = optics_tools.propagate_by_fresnel(curved_ef, dx, 
                                                              distance_to_fibre, 
                                                              wavelength_in_mm)

                # Add the resulting wavefront to a field containing the 
                # wavefronts of all 9 lenses
                side = int(npix/2 - 1.5*shift)
                out_ef[(side + shift*(x+1)):(side + shift*(x+2)), 
                       (side + shift*(y+1)):(side + shift*(y+2))] = ef_at_fibre
            
                # Compute the coupling, applying an offset to the fibre mode as 
                # required (To account for the outer fibres being off centre and 
                # displaced outwards)
                coupling = optics_tools.compute_coupling(npix, dx, ef_at_fibre, 
                                                         self.width, fibre_mode, 
                                                         -offset*x, -offset*y)
                coupling_1_to_9.append(coupling)
                
                # [11] - Compute aperture loss and eta
                # Calculate aperture loss for the microlens by calculating the 
                # light that gets into the fibre (use a circular mask)
                aperture_loss_fibre = np.sum(np.abs(ef_at_fibre)**2)/input_field
                aperture_loss_1_to_9.append(aperture_loss_fibre)
                
                # Calculate and combine the eta for each set of lenses (1, 5, 9)
                # 1 Fibre
                if (x == 0) and (y == 0):
                    eta_1_5_9[0] += coupling * aperture_loss_fibre
                # 5 Fibres   
                if not ((np.abs(x) == 1) and (np.abs(y) == 1)):
                    eta_1_5_9[1] += coupling * aperture_loss_fibre               
                # 9 Fibres
                eta_1_5_9[2] += coupling * aperture_loss_fibre
        
        return out_ef, coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9

class PIAAOptics(OpticalElement): 
    """A set of PIAAOptics as an OpticalElement - two lenses separated by a 
    known difference. Implements functions in piaa.py."""
    
    def __init__(self, alpha, r0, frac_to_focus, n_med, thickness, radius_in_mm, 
                 real_heights, dx, npix, wavelength_in_mm):
        """Generates the phase aberrations introduced by each of a pair of PIAA 
        lenses given the relevant parameters.
        
        Parameters
        ----------
        alpha: float
            In the formula for I_1 - the exponent of the Gaussian intensity.
        r0: float
            The fractional radius of the telescope secondary obstruction in the 
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
            Refractive index of the medium. Used to compensate for the glass and
            give extra power to the new optic as well as to estimate the height.
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
        """
        # Store parameters
        self.alpha = alpha
        self.r0 = r0
        self.frac_to_focus = frac_to_focus
        self.n_med = n_med
        self.thickness = thickness
        self.radius_in_mm = radius_in_mm
        self.real_heights = real_heights
        self.dx = dx
        self.npix = npix
        self.wavelength_in_mm = wavelength_in_mm
        
        # Generate PIAA lenses #1 and #2
        self.piaa_lens1, self.piaa_lens2 = piaa.create_piaa_lenses(self.alpha, 
                                           self.r0, self.frac_to_focus, 
                                           self.n_med, self.thickness, 
                                           self.radius_in_mm, self.real_heights, 
                                           self.dx, self.npix, 
                                           self.wavelength_in_mm)
    
    def apply(self, input_ef):
        """Applies the first PIAA lens to the incident wave, propagates the 
        result to the second PIAA lens before applying it too.

        Parameters
        ----------
        input_ef: np.array([[...]...])
            2D square incident wave consisting of complex numbers.
            
        Return
        ------
        output_ef: np.array([[...]...])
            The modified wave after application of the PIAA optics
        """
        # Pass the electric field through the first PIAA lens
        ef_1 = input_ef * np.exp(1j * self.piaa_lens1) 
        
        # Propagate the electric field through glass to the second lens
        ef_2 = optics_tools.propagate_by_fresnel(ef_1, self.dx, 
                                                 self.thickness / self.n_med, 
                                                 self.wavelength_in_mm)
        
        # Pass the electric field through the second PIAA lens
        output_ef = ef_2 * np.exp(1j * self.piaa_lens2)

        return output_ef