import numpy as np
import optics
import optics_tools
import utils
import pylab as pl
import time
import multiprocessing as mp
import feed
import csv
import time
import datetime

output = mp.Queue()

def simulate_for_x_iterations(dz, offset, seeing, alpha, use_piaa, use_tip_tilt, stromlo_variables, iterations, output):
    """
    Run iterations in multiples of 16
    """
    #Initialise results [1 fibre, 5 fibres, 9 fibres]
    eta_totals = [0, 0, 0]
    
    # Use larger array of turbulence and select subcomponents
    npix = 2048
    n = 4
    
    # Initialise count
    count = 0
    
    # Initialise Feed
    rhea_feed = feed.Feed(stromlo_variables, use_piaa)
    
    while iterations > 0:
        # Generate large turbulence patch
        turbulence = optics_tools.kmf(npix * n)
        
        for x in xrange(0,n):
            for y in xrange(0,n):
                # Run for a single patch of turbulence
                turbulent_patch = turbulence[x*npix:((x+1)*npix),y*npix:((y+1)*npix)]
                coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9, electric_field, turbulence_out, tef = rhea_feed.propagate_to_fibre(dz, offset, turbulent_patch, seeing, alpha, use_tip_tilt)
                
                # Add
                eta_totals[0] += eta_1_5_9[0]
                eta_totals[1] += eta_1_5_9[1]
                eta_totals[2] += eta_1_5_9[2]
                
                # Decrement iterations
                iterations -= 1
                count += 1
                print count, " - ", dz, offset, seeing, alpha, use_piaa, use_tip_tilt, stromlo_variables, str(datetime.datetime.now().time())
    
    # All finished, compute average eta
    eta_avg = [x / count for x in eta_totals]
    
    output.put([eta_avg[0], 1, dz, seeing, alpha, use_piaa, use_tip_tilt])
    output.put([eta_avg[1], 5, dz, seeing, alpha, use_piaa, use_tip_tilt])
    output.put([eta_avg[2], 9, dz, seeing, alpha, use_piaa, use_tip_tilt])
    
    return eta_avg
        
def simulate(results_save_path, offset, iterations, dz_values, seeing_values=[0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], use_piaa=True, use_tip_tilt=True, alpha_values=[2.0], stromlo_variables=True):
    """
    Parameters
    ----------
    results_save_path: string  
        The path to save the .csv to.
    offset: int
        The x/y distance that each of the outer fibres is off-centre by in an outwards direction
    iterations: int
        The number of times to run each simulation in order to average out the effects of turbulence
    dz_values: [float]
        The list of dz values to simulate.
    seeing_values: [float]
        The list of seeing values to simulate.
    alpha_values: [float]
        The list of alpha values to simulate.    
    apply_piaa: boolean
        Whether to apply the PIAA lenses or not            
    use_circular_lenslet: boolean
        Whether to use a singlr circular lens instead of the square microlens array
        
    Returns
    -------
    simulation_results: 2D list
        The result and parameters for each simulation: [eta, dz, seeing, alpha]
    """
    
    # Initialise list to store results
    simulation_results = [['eta', '# Fibres','dz','seeing','alpha','use_piaa','use_tip_tilt']]

    processes = []
    output = mp.Queue()
    
    # For each set of dz-alpha-seeing, run for the given number of iterations
    for dz in dz_values:
        for seeing in seeing_values:
            for alpha in alpha_values:
                # Construct processes
                processes.append(mp.Process(target=simulate_for_x_iterations, args=(dz, offset, seeing, alpha, use_piaa, use_tip_tilt, stromlo_variables, iterations, output)))
                
    # Run processes
    for p in processes:
        p.start()
        
    # Exit the completed processes
    for p in processes:
        p.join()
        
    # Get the results
    for i in xrange(0, len(processes)*3):
        simulation_results.append(output.get())
    
    try:
        # All results obtained, save as file
        with open(results_save_path, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(simulation_results)
    except:
        print "Error when writing to .csv"
    else:
        return simulation_results
        
    return simulation_results    

def construct_simulation_plots(save_path, results, offset, iterations=50, dz_values=[0.145,0.15,0.155], seeing_values=[0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], alpha_values=[2.0], apply_piaa=True, use_circular_lenslet=False, number_of_fibres=[1,5,9]):
    """Constructs and saves plots of the output of piaa.simulate()
    Each plot is of dz (independent) vs eta (dependent) for a given seeing value, with lines of varying alpha being plotted
    
    Parameters
    ----------
    save_path: string  
        The path to save the plots to, ending with //
    results: 2D list
        The result and parameters for each simulation: [eta, dz, seeing, alpha]
    offset: int
        The x/y distance that each of the outer fibres is off-centre by in an outwards direction
    iterations: int
        The number of times to run each simulation in order to average out the effects of turbulence
    dz_values: [float]
        The list of dz values to simulate.
    seeing_values: [float]
        The list of seeing values to simulate.
    alpha_values: [float]
        The list of alpha values to simulate.    
    apply_piaa: boolean
        Whether to apply the PIAA lenses or not            
    use_circular_lenslet: boolean
        Whether to use a singlr circular lens instead of the square microlens array
    """
    # Make sure the results array is in the correct format
    results = np.array(results[1:][:])
    
    # Construct a graph for each seeing value, a line for each alpha value and a point for each eta-dz pair
    for dz in dz_values:
        pl.clf()
        for n in number_of_fibres:
            seeing = []
            eta = []
            
            # Find the entries in results that match the given dz and alpha number of fibre
            for simulation in xrange(0, len(results)):
                if (results[simulation, 2] == dz) and (results[simulation, 1] == n):
                    seeing.append(results[simulation, 3])
                    eta.append(results[simulation, 0])
            # All matching simulations found, plot line
            pl.plot(seeing,eta, label=("# Fibres = " + str(n)))
            
        # All alphas plotted, apply labels/legend and save
        title = "Eta as a Function of Seeing, Averaged Over " + str(iterations) + " Iterations " 
        details = "(dz = " + str(dz) + ", offset = " + str(offset) + ", PIAA = " + str(apply_piaa) + ", Square Microlens Array = " + str(not use_circular_lenslet) + ")"
        pl.title(title + "\n" + details, fontsize=12)
        pl.xlabel("seeing (arcseconds)", fontsize=12)
        pl.ylabel("eta (%)")
        pl.legend(prop={'size':10})
        pl.grid()
        pl.savefig( (save_path + title + details + ".png" ))     