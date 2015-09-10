"""Simulates a Feed for a given list of parameters and number of iterations.

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
import feed
import csv
import time
import datetime

output = mp.Queue()

def simulate_for_x_iterations(dz, offset, seeing, alpha, use_piaa, use_tip_tilt, stromlo_variables, iterations, output):
    """For a given set of parameters, simulate for a certain number of iterations and average the resulting eta.
    
    Run iterations in multiples of 16 to make optimal use of how the turbulence patches are selected (from a larger grid of 4x4)
    
    TODO: Do not compute turbulence in perfect seeing (0.0)
    
    Parameters
    ----------
    dz: float   
        Free parameter, determines the magnification of the microlens array (changed to optimise coupling)
    offset: int
        The x/y distance that each of the outer fibres is off-centre in a radially outwards direction
    seeing: float
        The seeing for the telescope in arcseconds.
    alpha: float
        Exponent constant from the Gaussian distribution achieved by the PIAA optics  
    use_piaa: boolean
        Whether to apply the PIAA optics or not          
    use_tip_tilt: boolean
        Whether to apply Tip/Tilt correction to the incident wavefront
    stromlo_variables: boolean
        Whether to use the Stromlo or Subaru set of fibre and wavelength parameters    
    iterations: int
        The number of times to run each simulation in order to average out the effects of turbulence
    output: queue
        The queue to store the simulation results in for the purposes of multiprocessing
        
    Returns
    -------
    eta_avg: [float, float, float]
        The average eta values for 1, 5 and 9 fibres.
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
    
    # In imperfect seeing, there is a random element, necessitating multiple iterations
    if seeing > 0.0: 
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
    
    # In perfect seeing, no need to iterate as there is no random element --> can run only once
    else:
        coupling_1_to_9, aperture_loss_1_to_9, eta_1_5_9, electric_field, turbulence_out, tef = rhea_feed.propagate_to_fibre(dz, offset, 1.0, seeing, alpha, use_tip_tilt)
        eta_avg = eta_1_5_9
        print dz, offset, seeing, alpha, use_piaa, use_tip_tilt, stromlo_variables, str(datetime.datetime.now().time())
        
    output.put([eta_avg[0], 1, dz, seeing, alpha, float(use_piaa), float(use_tip_tilt), float(stromlo_variables), offset, count])
    output.put([eta_avg[1], 5, dz, seeing, alpha, float(use_piaa), float(use_tip_tilt), float(stromlo_variables), offset, count])
    output.put([eta_avg[2], 9, dz, seeing, alpha, float(use_piaa), float(use_tip_tilt), float(stromlo_variables), offset, count])
    
    return eta_avg
        
def simulate(results_save_path, offset, iterations, max_running_processes, dz_values, seeing_values=[0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], use_piaa=True, use_tip_tilt=True, alpha_values=[2.0], stromlo_variables=True):
    """Simulates fibre propagation for sets of dz, seeing and alpha parameters, averaging the resulting eta over a supplied number of iterations.
    
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
    use_piaa: boolean
        Whether to apply the PIAA optics or not          
    use_tip_tilt: boolean
        Whether to apply Tip/Tilt correction to the incident wavefront
    alpha_values: [float]
        The list of alpha values to simulate.    
    stromlo_variables: boolean
        Whether to use the Stromlo or Subaru set of fibre and wavelength parameters   
        
    Returns
    -------
    simulation_results: 2D list
        The result and parameters for each simulation: ['eta', '# Fibres','dz','seeing','alpha','use_piaa','use_tip_tilt', 'stromlo_variables', 'offset', 'iterations']
    """
    
    # Initialise list to store results
    simulation_results = [['eta', '# Fibres','dz','seeing','alpha','use_piaa','use_tip_tilt', 'stromlo_variables', 'offset', 'iterations']]

    processes = []
    output = mp.Queue()
    
    # For each set of dz-alpha-seeing, run for the given number of iterations
    for dz in dz_values:
        for alpha in alpha_values:
            for seeing in seeing_values:
                # Construct processes
                processes.append(mp.Process(target=simulate_for_x_iterations, args=(dz, offset, seeing, alpha, use_piaa, use_tip_tilt, stromlo_variables, iterations, output)))
    
    # Limit the number of active processes at any one time
    total_processes = len(processes)
    finished_processes = 0
    
    while len(processes) > 0:
        # Start a given number of processes
        running_processes = []
        
        for i in xrange(0, max_running_processes):
            if len(processes) > 0:
                running_processes.append(processes.pop())
        
        # Run processes
        for p in running_processes:
            p.start()
            
        # Exit the completed processes
        for p in running_processes:
            p.join()
            finished_processes += 1
        
    # Get the results
    for i in xrange(0, total_processes*3):
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

def plot_simulation_results(csv_path, image_path):
    """Loads simulation results from a csv file and plots corresponding graphs, pulling relevant information regarding the simulations from the file.
    
    TODO: Generalise to allow custom selection of x axis, y axis, legend variables and separate graph variables
    
    Parameters
    ----------
    csv_path: string
        The path to the saved csv of results
    image_path: string
        The save path for the plots.
    """
    # Load the csv
    # [eta, # Fibres, dz, seeing, alpha, use_piaa, use_tip_tilt, stromlo_variables, offset, iterations]
    results = np.loadtxt(open(csv_path, "rb"), delimiter=",",skiprows=1)
    
    # Reconstruct the simulation information
    number_of_fibres = np.sort(np.unique(results[:,1]))
    dz_values = np.sort(np.unique(results[:,2]))
    seeing_values = np.sort(np.unique(results[:,3]))
    alpha_values = np.sort(np.unique(results[:,4]))
    
    use_piaa = bool(results[0,5])
    use_tip_tilt = bool(results[0,6])
    stromlo_variables = bool(results[0,7])
    offset = results[0,8]
    iterations = results[0,9]
    
    # Construct a graph for each seeing value, a line for each alpha value and a point for each eta-dz pair
    for n in number_of_fibres:
        pl.clf()
        for a in alpha_values:
            seeing = []
            eta = []
            
            # Find the entries in results that match the given dz and alpha number of fibre
            for simulation in xrange(0, len(results)):
                if (results[simulation, 4] == a) and (results[simulation, 1] == n):
                    seeing.append(results[simulation, 3])
                    eta.append(results[simulation, 0])
            # All matching simulations found, plot line
            e_sorted = [e for (s,e) in sorted(zip(seeing,eta))]
            pl.plot(seeing_values,e_sorted, label=("Alpha = " + str(int(a))))
            
        # All alphas plotted, apply labels/legend and save
        title = "Eta vs Seeing, " + str(int(iterations)) + " Iterations " 
        details = "(n = " + str(int(n)) + ", offset = " + str(offset) + ", PIAA = " + str(use_piaa) + ", Tip-Tilt = " + str(use_tip_tilt) + ", Stromlo Feed = " + str(stromlo_variables) + ")"
        pl.title(title + "\n" + details, fontsize=12)
        pl.xlabel("seeing (arcseconds)", fontsize=12)
        pl.ylabel("eta")
        pl.legend(prop={'size':10})
        pl.grid()
        pl.ylim([0.0,1.0])
        pl.savefig( (image_path + title + details + ".png" ))     

def construct_simulation_plots(save_path, results, offset, iterations=50, dz_values=[0.145,0.15,0.155], seeing_values=[0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], alpha_values=[2.0], use_piaa=True, use_tip_tilt=True, number_of_fibres=[1,5,9]):
    """Older, more clunky plotting method. Can plot straight from an array, rather than reading in a csv.
    
    Constructs and saves plots of the output of piaa.simulate()
    Each plot is of dz (independent) vs eta (dependent) for a given seeing value, with lines of varying alpha being plotted
    
    TODO: Ensure y axis always is between 0.0 and 1.0, enabling side by side comparisons.
    
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
        details = "(dz = " + str(dz) + ", offset = " + str(offset) + ", PIAA = " + str(apply_piaa) + ", Tip/Tilt = " + str(use_tip_tilt) + ")"
        pl.title(title + "\n" + details, fontsize=12)
        pl.xlabel("seeing (arcseconds)", fontsize=12)
        pl.ylabel("eta")
        pl.legend(prop={'size':10})
        pl.grid()
        #pl.ylim([0.0,1.0])
        pl.savefig( (save_path + title + details + ".png" ))     