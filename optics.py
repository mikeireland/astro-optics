"""
"""

import numpy as np
import optics_tools
import piaa
import utils

class OpticalElement:
    """
    """
    def __init__(self, focal_length, width):
        """
        """
        self.focal_length = focal_length
        self.width = width
    
    def apply(self, input_ef, npix, dx, wavelength_in_mm):
        """
        """
        return input_ef
        
class CircularLens(OpticalElement):
    """
    """
    def apply(self, input_ef, npix, dx, wavelength_in_mm):
        """
        """
        # Apply a circular mask
        masked_ef = utils.circle(npix, (self.width / dx)) * input_ef
        
        # Apply the curved wavefront of the lens
        output_ef = masked_ef * optics_tools.curved_wf(npix, dx, self.focal_length, wavelength_in_mm)
        
        return output_ef
       
        
class MicrolensArray_3x3(OpticalElement):
    """
    """
    def apply(self, input_ef, npix, dx, wavelength_in_mm):
        """Propagate the electric field through a 3 x 3 square lenslet array
        
        Parameters
        ----------
        input_ef: 2D numpy.ndarray
            The electric field before the lenslet
       
       Returns
        -------
        efield_after: 2D numpy.ndarray
            The electric field after the lenslet
        """            
        # Initialise the resultant electric field (that will be the addition of each of the 9 micro-lenslets)
        output_ef = np.zeros((npix, npix))
        
        # Create the square window that will shifted around to represent the light getting into each micro-lenslet
        square = utils.square(npix, self.width/dx) + 0j
        
        # Starting in the top left, track the window over each row
        for x in xrange(-1,2):
            for y in xrange(-1,2):
                # Calculate the base shift (the actual shift requires the -1, 0 or 1 multiplier to get direction) 
                shift = int(self.width/dx)
                
                # Have the window be a curved wavefront and shift as required
                shifted_square = np.roll(np.roll( (square * optics_tools.curved_wf(npix, dx, self.focal_length, wavelength_in_mm) ), shift*y, axis=1), shift*x, axis=0)
                
                # Add the resulting wavefront to a field containing the wavefronts of all 9 lenses
                output_ef = output_ef + shifted_square * input_ef
 
        return output_ef
    
    def apply_propagate_and_couple(self, input_ef, npix, dx, wavelength_in_mm, distance_to_fibre, input_field, offset, fibre_mode):
        """
        """
        # Initialise the resultant electric field (that will be the addition of each of the 9 micro-lenslets)
        output_ef = np.zeros((npix, npix)) + 0j
        
        # Initialise the displacement of each square from the centre
        shift = int(self.width / dx)
        
        # Create the square window that will be used to represent the light getting into each micro-lenslet
        square = utils.square(npix, self.width/dx) + 0j
        curved_square = optics_tools.curved_wf(shift, dx, self.focal_length, wavelength_in_mm)
        
        # Variable to store the coupling for the central fibre (1), centre and horizontal/vertical fibres (5) and all, including diagonals (9)
        coupling_1_to_9 = []
        aperture_loss_1_to_9 = []
        eta_1_5_9 = [0,0,0]
        
        # For each of the 9 microlenses 
        for x in xrange(-1,2):
            for y in xrange(-1,2):
                # Shift the desired square to the centre
                shifted_ef = np.roll(np.roll(input_ef, -shift*y, axis=1), -shift*x, axis=0)

                # Apply the window and curved wavefront
                curved_ef = curved_square * shifted_ef[(npix/2 - shift/2):(npix/2 + shift/2),(npix/2 - shift/2):(npix/2 + shift/2)]
                
                # [10] - Propagate the light passing through the microlens to the fibre
                ef_at_fibre = optics_tools.propagate_by_fresnel(curved_ef, dx, distance_to_fibre, wavelength_in_mm)

                # Add the resulting wavefront to a field containing the wavefronts of all 9 lenses
                side = int(npix/2 - 1.5*shift)
                output_ef[(side + shift*(x+1)):(side + shift*(x+2)), (side + shift*(y+1)):(side + shift*(y+2))] = ef_at_fibre
            
                # Compute the coupling, applying an offset to the fibre mode as required (To account for the outer fibres being off centre and displaced outwards)
                coupling = optics_tools.compute_coupling(npix, dx, ef_at_fibre, self.width, fibre_mode, -offset*x, -offset*y)
                coupling_1_to_9.append(coupling)
                
                # [11] - Compute aperture loss and eta
                # Calculate aperture loss for the microlens by calculating the light that gets into the fibre (use a circular mask)
                aperture_loss_fibre = np.sum(np.abs(ef_at_fibre)**2) / input_field
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
      
        return output_ef, coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9

class PIAAOptics(OpticalElement): 
    """
    """
    def __init__(self, alpha, r0, frac_to_focus, delta, dt, n_med, thickness, radius_in_mm, real_heights, dx, npix, wavelength_in_mm):
        """
        """
        # Store parameters
        self.alpha = alpha
        self.r0 = r0
        self.frac_to_focus = frac_to_focus
        self.delta = delta
        self.dt = dt
        self.n_med = n_med
        self.thickness = thickness
        self.radius_in_mm = radius_in_mm
        self.real_heights = real_heights
        self.dx = dx
        self.npix = npix
        self.wavelength_in_mm = wavelength_in_mm
        
        # Generate PIAA lenses #1 and #2
        self.piaa_lens1, self.piaa_lens2 = piaa.create_piaa_lenses(self.alpha, self.r0, self.frac_to_focus, self.delta, self.dt, self.n_med, self.thickness, self.radius_in_mm, self.real_heights, self.dx, self.npix, self.wavelength_in_mm)
    
    def apply(self, input_ef):
        """
        """
        # Pass the electric field through the first PIAA lens
        ef_1 = input_ef * np.exp(1j * self.piaa_lens1) 
        
        # Propagate the electric field through glass to the second lens
        ef_2 = optics_tools.propagate_by_fresnel(ef_1, self.dx, self.thickness / self.n_med, self.wavelength_in_mm)
        
        # Pass the electric field through the second PIAA lens
        output_ef = ef_2 * np.exp(1j * self.piaa_lens2)

        return output_ef    