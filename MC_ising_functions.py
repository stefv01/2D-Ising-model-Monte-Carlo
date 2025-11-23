"""
This file contains all necessary functions to simulate a 2D ising model by using the Metropolis and Swendsen-Wang algorithms. 
"""

import numpy as np
from numpy import random
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
from scipy.optimize import curve_fit
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def compute_energy(spins, bJ, h):
    """
    Calculate the total energy of the lattice using Numba.

    Parameters
    ----------
    spins : np.ndarray
        Lattice configuration (int8).
    bJ : float
        Coupling constant * inverse temperature (beta * J).
    h : float
        External magnetic field.

    Returns
    -------
    NN_energies : np.ndarray
        Nearest neighbor interaction map (float32).
    total_E : float
        Total energy of the system.
    """
    L = spins.shape[0]
    NN_energies = np.zeros((L, L), dtype=np.float32)
    E_interaction = 0.0
    E_field = 0.0
    
    for i in range(L):
        for j in range(L):
            # Calculate sum of 4 nearest neighbors with periodic boundary conditions
            n_sum = (spins[(i+1)%L, j] + 
                     spins[(i-1)%L, j] + 
                     spins[i, (j+1)%L] + 
                     spins[i, (j-1)%L])
            
            # Interaction term for the specific site
            term = spins[i, j] * n_sum
            NN_energies[i, j] = term
            
            # Accumulate energy terms
            E_interaction += term
            E_field += spins[i, j]
            
    # Total Energy = -J * sum(s_i * s_j) - B * sum(s_i)
    # We divide interaction by 2 because each pair is counted twice in the loop
    total_E = -0.5 * bJ * E_interaction - h * E_field

    return NN_energies, total_E

@njit(fastmath=True)
def metropolis_step(spins, bJ, h):
    """
    Perform one Metropolis update sweep using a checkerboard scheme.

    Parameters
    ----------
    spins : np.ndarray
        Current lattice configuration (int8).
    bJ : float
        Coupling constant * inverse temperature.
    h : float
        External magnetic field.

    Returns
    -------
    spins : np.ndarray
        Updated lattice configuration.
    """
    L = spins.shape[0]
    
    # Checkerboard decomposition: 0 for even sites (Red), 1 for odd sites (Black)
    # Allows updating half the lattice independently
    for parity in range(2):
        for i in range(L):
            for j in range(L):
                if (i + j) % 2 == parity:
                    
                    # Sum of nearest neighbors (Periodic BCs)
                    nn_sum = (spins[(i + 1) % L, j] + 
                              spins[(i - 1) % L, j] + 
                              spins[i, (j + 1) % L] + 
                              spins[i, (j - 1) % L])
                    
                    # Calculate energy change for flipping spin (i,j)
                    # dE = 2 * s_i * (J * sum_neighbors + B)
                    delta_E = 2 * spins[i, j] * (bJ * nn_sum + h)
                    
                    # Metropolis Criterion
                    # 1. If dE <= 0, energy decreases, accept always
                    if delta_E <= 0:
                        spins[i, j] *= -1
                    # 2. If dE > 0, accept with probability exp(-dE)
                    elif np.random.random() < np.exp(-delta_E):
                        spins[i, j] *= -1
                        
    return spins

@njit(fastmath=True)
def sw_step(spins, L, bJ):
    """
    Perform one Swendsen-Wang cluster update step.

    Parameters
    ----------
    spins : np.ndarray
        Current lattice configuration (int8).
    L : int
        Lattice dimension.
    bJ : float
        Coupling constant * inverse temperature.

    Returns
    -------
    spins : np.ndarray
        Updated lattice configuration.
    """
    # Probability to form a bond between aligned neighbors: 1 - exp(-2*beta*J)
    p_bond = 1.0 - np.exp(-2.0 * bJ)
    
    visited = np.zeros((L, L), dtype=np.bool_)
    
    # Pre-allocate stack arrays
    stack_x = np.zeros(L * L, dtype=np.int32)
    stack_y = np.zeros(L * L, dtype=np.int32)
    
    # Neighbors offsets: right, left, down, up
    neighbors_offset = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int8)
    
    # Iterate over every site to find new clusters
    for i in range(L):
        for j in range(L):
            if not visited[i, j]:
                # Found a new unvisited site -> Start a new cluster
                visited[i, j] = True
                
                # Decide spin flip for the entire cluster immediately (50% chance)
                cluster_flip = 1
                if np.random.random() < 0.5:
                    cluster_flip = -1
                
                # Initialize stack
                stack_ptr = 0
                stack_x[stack_ptr] = i
                stack_y[stack_ptr] = j
                stack_ptr += 1
                
                # Process the cluster using explicit stack
                while stack_ptr > 0:
                    stack_ptr -= 1
                    curr_x = stack_x[stack_ptr]
                    curr_y = stack_y[stack_ptr]
                    
                    # Apply the flip decision to the current spin immediately
                    spins[curr_x, curr_y] *= cluster_flip
                    
                    # Check all 4 neighbors
                    for k in range(4):
                        nx = curr_x + neighbors_offset[k, 0]
                        ny = curr_y + neighbors_offset[k, 1]
                        
                        # Manual Periodic Boundary Conditions
                        if nx >= L: nx -= L
                        elif nx < 0: nx += L
                        
                        if ny >= L: ny -= L
                        elif ny < 0: ny += L
                        
                        if not visited[nx, ny]:
                            # Check if neighbors were aligned *before* the flip.
                            old_val_curr = spins[curr_x, curr_y] * cluster_flip
                            
                            if spins[nx, ny] == old_val_curr:
                                # If aligned, form bond with probability p_bond
                                if np.random.random() < p_bond:
                                    visited[nx, ny] = True
                                    stack_x[stack_ptr] = nx
                                    stack_y[stack_ptr] = ny
                                    stack_ptr += 1
                                    
    return spins


# -----------------------------------------------------------------------------
# Class Definitions
# -----------------------------------------------------------------------------

class sys_params:
    """
    Container class for system parameters and lattice initialization in the 2D Ising model.
    
    Stores the main parameters of the Ising system (lattice size, coupling, field)
    and includes methods for lattice initialization and energy calculation.
    
    Parameters
    ----------
    L : int
        Linear dimension of the square lattice (L x L)
    bJ : float
        Product of inverse temperature (β=1/(k_B*T)) and coupling constant (βJ)
    h : float
        External magnetic field strength
    """

    def __init__(self, L, bJ, h):

        self.L = L
        self.bJ = bJ
        self.h = h
        self.spins = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

    def init_lattice(self, p, spin_type):
        """
        Initialize a 2D spin lattice with a given fraction of spins pointing up or down.

        Parameters
        ----------
        p : float
            Probability of assigning the dominant spin type (between 0 and 1).
        spin_type : str
            Either 'up' or 'down'; determines which spin direction dominates.

        Returns
        -------
        None
            Updates self.spins in place.
        """
        rand_dist = np.random.random((self.L, self.L)).astype(np.float32)

        if spin_type == 'up':
            self.spins[rand_dist >= p] = -1
            self.spins[rand_dist < p] = 1
        elif spin_type == 'down':
            self.spins[rand_dist >= p] = 1
            self.spins[rand_dist < p] = -1
            
        self.spins = self.spins.astype(np.int8)

    def lattice_energy(self):
        """
        Compute the total energy of a 2D Ising lattice given spin configuration, coupling J, and external field B.

        Returns
        -------
        NN_energies : np.ndarray
            Array containing the sum of nearest-neighbor contributions for each spin.
        total_energy : float
            Total energy of the lattice including interaction and field contributions.
        """
        return compute_energy(self.spins, self.bJ, self.h)


class MetropolisIsing:
    """
    Implementation of the Metropolis algorithm for the 2D Ising model.
    
    Performs spin flip Monte Carlo updates with a checkerboard scheme for efficient
    sampling of the Boltzmann distribution. Includes methods for both single-temperature
    simulations and temperature sweeps.
    
    Parameters
    ----------
    params : sys_params
        Container object holding system parameters (L, bJ, B) and spin configuration.
    """

    def __init__(self, params):

        self.params = params
        self.L = params.L
        self.bJ = params.bJ
        self.h = params.h
        self.spins = params.spins  
        self.NN_energies, self.energy = params.lattice_energy()


    def run_step(self):
        """
        Perform one Metropolis update sweep using a checkerboard scheme.

        Returns
        -------
        None
            Updates self.spins, self.NN_energies, and self.energy in place.
        """
        self.spins = metropolis_step(self.spins, self.bJ, self.h)
        self.NN_energies, self.energy = self.params.lattice_energy()

    def simulate(self, tsteps, partition=1, save_spins=False):
        """
        Runs a Monte Carlo simulation of the 2D Ising model using the specified update algorithm.

        Parameters
        ----------
        tsteps : int
            Total number of Monte Carlo steps.
        partition : int
            Interval for saving spin configurations.
        save_spins : bool, optional
            If True, saves spin configurations. If False, returns None for spins.

        Returns
        -------
        np.ndarray or None
            Saved spin configurations at partition intervals. None if save_spins is False.
        np.ndarray
            Total energy at each timestep.
        np.ndarray
            Magnetization per spin at each timestep.
        """

        if save_spins:
            list_spins = np.zeros((tsteps//partition, self.L, self.L), dtype=np.int8)
        else:
            list_spins = None
            
        list_energies = np.zeros(tsteps, dtype=np.float32)
        list_avg_spin = np.zeros(tsteps, dtype=np.float32)

        for t in range(tsteps):

            self.run_step()

            if save_spins and t % partition == 0:
                list_spins[t//partition] = self.spins

            list_energies[t] = self.energy
            list_avg_spin[t] = np.sum(self.spins) / (self.L**2)

        return list_spins, list_energies, list_avg_spin

    def temperature_sweep(self, bJ_params, tsteps, partition=1, save_spins=False):
        """
        Evolve the lattice through a range of temperatures (J values).
        
        Parameters
        ----------
        bJ_params : np.ndarray
            Array of coupling constants (βJ) to simulate
        tsteps : int
            Number of Monte Carlo steps per temperature
        partition : int
            Interval for saving configurations
        save_spins : bool, optional
            If True, saves spin configurations. If False, returns None for spins.
            
        Returns
        -------
        tuple
            (lattice_evol, energy_series, mag_series)
            lattice_evol: Saved spin configurations (or None)
            energy_series: Energy at each step for each temperature  
            mag_series: Magnetization at each step for each temperature
        """
        energy_series = np.zeros((len(bJ_params), tsteps), dtype=np.float32)
        mag_series = np.zeros((len(bJ_params), tsteps), dtype=np.float32)
        
        if save_spins:
            spins_series = np.zeros((len(bJ_params) * (tsteps//partition), self.L, self.L), dtype=np.int8)
        else:
            spins_series = None
            

        for i, bJ in enumerate(tqdm(bJ_params, desc="Temperature sweep")):
            self.bJ = bJ
            self.params.bJ = bJ
            
            list_spins, list_energies, list_avg_spin = self.simulate(tsteps, partition, save_spins=save_spins)
            
            energy_series[i] = list_energies
            mag_series[i] = np.abs(list_avg_spin)
            
            if save_spins:
                start_idx = i * (tsteps//partition)
                end_idx = (i+1) * (tsteps//partition)
                spins_series[start_idx:end_idx] = list_spins
        
        return spins_series, energy_series, mag_series


class SWIsing:
    """
    Implementation of the Swendsen-Wang cluster algorithm for the 2D Ising model.
    
    The algorithm identifies clusters of parallel spins and flips them with probability 1/2,
    providing more efficient sampling near critical temperatures compared to single-spin flip methods.
    
    Parameters
    ----------
    params : sys_params
        Container object holding system parameters (L, bJ, B) and spin configuration.
    """
    
    def __init__(self, params):

        self.params = params
        self.L = params.L
        self.bJ = params.bJ
        self.spins = params.spins  
        self.NN_energies, self.energy = params.lattice_energy()
        
    def run_step(self):
        """
        Perform one complete Swendsen-Wang cluster update step.
        """
        self.spins = sw_step(self.spins, self.L, self.bJ)
        self.NN_energies, self.energy = self.params.lattice_energy()

    def simulate(self, tsteps, partition=1, save_spins=False):
        """
        Runs a Monte Carlo simulation of the 2D Ising model using the specified update algorithm.

        Parameters
        ----------
        tsteps : int
            Total number of Monte Carlo steps.
        partition : int
            Interval for saving spin configurations.
        save_spins : bool, optional
            If True, saves spin configurations. If False, returns None for spins.

        Returns
        -------
        np.ndarray or None
            Saved spin configurations at partition intervals. None if save_spins is False.
        np.ndarray
            Total energy at each timestep.
        np.ndarray
            Magnetization per spin at each timestep.
        """
  
        if save_spins:
            list_spins = np.zeros((tsteps//partition, self.L, self.L), dtype=np.int8)
        else:
            list_spins = None

        list_energies = np.zeros(tsteps, dtype=np.float32)
        list_avg_spin = np.zeros(tsteps, dtype=np.float32)

        for t in range(tsteps):

            self.run_step()

            if save_spins and t % partition == 0:
                list_spins[t//partition] = self.spins

            list_energies[t] = self.energy
            list_avg_spin[t] = np.sum(self.spins) / (self.L**2)

        return list_spins, list_energies, list_avg_spin


    def temperature_sweep(self, bJ_params, tsteps, partition=1, save_spins=False):
        """
        Evolve the lattice through a range of temperatures (J values).
        
        Parameters
        ----------
        bJ_params : np.ndarray
            Array of coupling constants (βJ) to simulate
        tsteps : int
            Number of Monte Carlo steps per temperature
        partition : int
            Interval for saving configurations
        save_spins : bool, optional
            If True, saves spin configurations. If False, returns None for spins.
            
        Returns
        -------
        tuple
            (lattice_evol, energy_series, mag_series)
            lattice_evol: Saved spin configurations (or None)
            energy_series: Energy at each step for each temperature  
            mag_series: Magnetization at each step for each temperature
        """
        energy_series = np.zeros((len(bJ_params), tsteps), dtype=np.float32)
        mag_series = np.zeros((len(bJ_params), tsteps), dtype=np.float32)
        
        if save_spins:
            spins_series = np.zeros((len(bJ_params) * (tsteps//partition), self.L, self.L), dtype=np.int8)
        else:
            spins_series = None

        original_bJ = self.bJ

        for i, bJ in enumerate(tqdm(bJ_params, desc="Temperature sweep")):
            self.bJ = bJ
            self.params.bJ = bJ
            
            list_spins, list_energies, list_avg_spin = self.simulate(tsteps, partition, save_spins=save_spins)
            
            energy_series[i] = list_energies
            mag_series[i] = np.abs(list_avg_spin)

            if save_spins:
                start_idx = i * (tsteps//partition)
                end_idx = (i+1) * (tsteps//partition)
                spins_series[start_idx:end_idx] = list_spins
        
        return spins_series, energy_series, mag_series