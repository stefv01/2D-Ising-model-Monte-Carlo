# Monte Carlo Simulation of the 2D Ising Model

A Python-based Monte Carlo (MC) simulation for the 2D Ising model, investigating phase transitions and critical phenomena in magnetic systems. This project implements MC algorithms such as Metropolis and Swendsen-Wang to study thermodynamic behavior, finite-size scaling, and critical behavior near the transition point.

## Physical Model

The simulation models a 2D lattice of spins as a classical statistical mechanics system.

  * **Interactions**: Spins interact via the **Ising Hamiltonian**:
    $$H(\mathbf{s}) = - J \sum_{\langle i, j \rangle} s_i s_j - h \sum_i s_i$$
    where $s_i = \pm 1$ represents the spin state, $J$ is the coupling constant (exchange interaction), and $h$ is the external magnetic field.
      * $J > 0$: Ferromagnetic (neighbors align).
      * $J < 0$: Antiferromagnetic (neighbors anti-align).
  * **Ensemble**: Canonical (NVT) - Temperature, Volume (lattice size), and Number of spins are fixed. The probability of a configuration is given by the Boltzmann distribution $P(\mathbf{s}) \propto e^{-\beta H(\mathbf{s})}$.
  * **Phase Transition**: The system exhibits a second-order phase transition at a critical temperature $T_c = 2 / \ln (1 + \sqrt{2})$ (Onsager's solution), characterized by a divergence in correlation length and susceptibility.

## Features

  * **Algorithms**:
      * **Metropolis**: Single-spin flip local update algorithm using a highly optimized **checkerboard decomposition**.
      * **Swendsen-Wang**: Cluster-flip global update algorithm to combat **critical slowing down** near $T_c$.
  * **Initialization**:
      * Lattices can be initialized as ordered (all up/down) or disordered (random) to study equilibration.
      * Custom spin concentrations available.
  * **Observables**:
      * Real-time tracking of Magnetization ($M$) and Energy ($E$).
      * Calculation of response functions: Magnetic Susceptibility ($\chi$) and Specific Heat ($c_h$).
      * Finite Size Scaling analysis for lattices of varying linear dimension $L$.
  * **Performance**:
      * **Numba JIT Compilation**: Core update loops are compiled to machine code for C-like performance.
      * **Memory Efficient**: Uses `int8` for spins and `float32` for energies to handle large lattices (e.g., $L=1000$).

## File Structure

  * `MC_ising_functions.py`: Contains the core simulation classes (`MetropolisIsing`, `SWIsing`), system parameter container (`sys_params`), and the Numba-optimized update kernels.
  * `MC_ising_simulations.ipynb`: The main executable notebook. It runs temperature sweeps, performs finite size scaling, and generates plots for thermodynamic quantities.
  * `error_analysis.py`: Statistical tools for analyzing MC time series, including autocorrelation time estimation (`iat_error`), data blocking, and bootstrap resampling.
  * `Simulation_Outputs/`: Directory storing generated plots and figures.

## Configuration & Parameters

Key simulation parameters defined in `MC_ising_simulations.ipynb`:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `L` | Linear lattice dimension ($N=L^2$) | 10, 20, 100 |
| `bJ` | Inverse temp * Coupling ($\beta J$) | Varied (Sweep) |
| `h` | External magnetic field | 0.0 |
| `tsteps` | Monte Carlo steps | 20,000+ |
| `partition` | Data saving interval | User defined |

## Usage

1.  **Install Dependencies**:

    ```bash
    pip install numpy scipy matplotlib tqdm numba
    ```

2.  **Run the Simulation**:
    Open `MC_ising_simulations.ipynb` in Jupyter Lab or Notebook.

    ```bash
    jupyter notebook MC_ising_simulations.ipynb
    ```

3.  **Workflow**:

      * **Part 1**: Visualize equilibration at fixed temperatures (low/high) and field effects.
      * **Part 2**: Perform temperature sweeps using Metropolis and Swendsen-Wang.
      * **Part 3**: Analyze finite size scaling effects on observables near $T_c$.

## Methodology Details

### Critical Slowing Down

Near the critical temperature $T_c$, the correlation length $\xi$ diverges. The Metropolis algorithm suffers from **critical slowing down** ($\tau \propto \xi^{2.17}$), requiring huge simulation times to decorrelate samples. The Swendsen-Wang algorithm updates entire clusters scaling with $\xi$, reducing the dynamic critical exponent ($z \approx 0.35$) and vastly improving efficiency near criticality.

## Additional Information

This project was initially developed in May 2025. It demonstrates the application of advanced MC techniques to solve statistical mechanics problems that are analytically intractable (or complex) in finite systems.

## Author

  * **Stefanos Vasileiadis**

## License

This project is open source and available under the **MIT License**.
