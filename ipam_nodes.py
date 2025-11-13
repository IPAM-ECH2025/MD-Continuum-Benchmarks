from pyiron_core.pyiron_nodes.atomistic.calculator.data import InputCalcMD
from pyiron_core.pyiron_workflow import as_function_node
from typing import Optional
import ase.units as units
import numpy as np

@as_function_node("Charge_distribution_Ion")
def PlotChargeDistribution(Data: dict, initial_structure, cation: str = "Na", anion: str = "F"):
    """
    Plots the charge distribution of ions along the z-axis.
    
    Parameters
    ----------
    Data : dict
        A dictionary where keys are element symbols and values are tuples of (x, y) data.
    initial_structure : ase.Atoms
        The initial structure of the system.
    cation : str, optional
        The chemical symbol of the cation (default is "Na").
    anion : str, optional
        The chemical symbol of the anion (default is "F").
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    from matplotlib import pyplot as plt
    import numpy as np
    from ase.units import Bohr

    def get_volume(deltares):
        V = initial_structure.cell[0,0] * initial_structure.cell[1,1] * deltares * (1 / Bohr) ** 3
        return(V)
    
    Sum = np.array(Data[cation][1]) - np.array(Data[anion][1])
    z = np.array(Data[cation][0])
    
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
    ax.plot(z, Sum)
    ax.set_title("Charge Distribution of Ion")

    ax.set_xlabel(r'z - coordinate (Å)')
    
    ax.set_ylabel(r'$\rho_\mathrm{e} $ (e/bohr$^3$)') # \times 10^4

    #fig.tight_layout()
    return fig


@as_function_node("EPD")
def PlotElectricPotentialDistribution(Data: dict, initial_structure, cation: str = "Na", anion: str = "F"):
    """
    Plots the electric potential distribution along the z-axis based on the provided element density data.
    
    Parameters
    ----------
    Data : dict
        A dictionary where keys are element symbols and values are tuples of (x, y) data.
    initial_structure : ase.Atoms
        The initial atomic structure used to determine cell dimensions.
    cation : str, optional
        The chemical symbol of the cation element (default is "Na").
    anion : str, optional
        The chemical symbol of the anion element (default is "F").
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the electric potential distribution plot.
    """
    from matplotlib import pyplot as plt
    import numpy as np
    from ase.units import Bohr

    fig, axs = plt.subplots(4,1, figsize=(3.25,2*4))#, sharex=True)#, sharex=True, sharey=True)
    fig.subplots_adjust(hspace = 0.15)#, wspace=0.5)
    fig.subplots_adjust(hspace = 0.3 / (2 / 3.25), wspace = 0.3)
    
    y = np.array(Data[cation][1]) - np.array(Data[anion][1]) - 0.83 * np.array(Data["O"][1]) + 0.415 * np.array(Data["H"][1])
    z = np.array(Data[cation][0])

    def get_volume(deltares):
        V= initial_structure.cell[0,0] * initial_structure.cell[1,1] * deltares * (1/Bohr) ** 3
        return(V)
        
    electron = -y / get_volume(np.gradient(z)[0])
    
    epsilon_0 = 8.854187817e-12  # Permittivity of free space (in F/m)
    e = 1.602176634e-19  # Elementary charge (in C)

    def e_c_E_potential(x, rho_e, color, axs):
        angstrom_to_meter = 10**(-10)
        axs[0].plot(x, rho_e, color = color)
    
        rho_c = - rho_e * e * (1/Bohr * 1/angstrom_to_meter) ** 3
        axs[1].plot(x, rho_c, color = color)
    
        E = 1/epsilon_0 * np.cumsum(rho_c * np.gradient(x * angstrom_to_meter))
        axs[2].plot(x, E, color = color)
    
        V = np.cumsum( E * np.gradient(x * angstrom_to_meter))
        axs[3].plot(x, V, color = color) 
    
        for ax in axs.flatten():
            ax.set(xlim=[x[0], x[-1]])
        ax.set_xlabel(r'z - coordinate ($\mathrm{\AA}$)')
    
        axs[3].set_ylabel(r'$\phi^{(i)}$ (V)')
        axs[2].set_ylabel(r'$E$ (V/m)')
        axs[1].set_ylabel(r'$\rho_\mathrm{c}$ (C/m$^3$)')
        axs[0].set_ylabel(r'$\rho_\mathrm{e} $ (e/bohr$^3$)') # \times 10^4
    
    e_c_E_potential(z, electron, 'black', axs)
    
    return(fig)

@as_function_node
def GetElementDistributions(trajectory, initial_structure, initial_step: int, n_bins: int = 50):
    """
    Retrieve the spatial distributions of elements along the z-axis.

    Parameters
    ----------
    trajectory : Trajectory-like object
        An object that provides ``positions`` (e.g. a pyiron
        ``Trajectory`` or any object with a ``positions`` attribute).
    initial_structure : Structure-like object
        The reference structure that defines the atomic species,
        lattice vectors, etc.
    initial_step : int
        The step from which to start analyzing the trajectory.
    n_bins : int, optional
        Number of bins to use for the histogram (default is 50).

    Returns
    -------
    bin_centers : numpy.ndarray
        The centers of the bins along the z-axis.
    Data : dict
        A dictionary where keys are element symbols and values are the normalized counts
        of those elements in each bin.
    """
    from pyiron_atomistics.atomistics.job.atomistic import Trajectory
    import numpy as np
    
    # Determine slab boundaries based on Ne and Al positions
    ind_Al = initial_structure.select_index("Al")
    ind_Ne = initial_structure.select_index("Ne")

    slab_bot = np.max(initial_structure.positions[ind_Al, 2])
    slab_top = np.min(initial_structure.positions[ind_Ne, 2])

    # Build a pyiron Trajectory object from the supplied data
    traj = Trajectory(positions=trajectory.positions[int(initial_step):], structure=initial_structure)
    
    elements = initial_structure.get_chemical_symbols()
    elements = set(elements)
    
    # Define bins for histogram along z-axis
    bins = np.linspace(0, slab_top - slab_bot, int(n_bins) + 1)
    
    Data = {}
    for element in elements:
        # find z-coordinate of all atoms of this element from initial_step to end of trajectory
        z_coords = np.array([atom.position[2] for i in range(len(traj)) for atom in traj[i] if atom.symbol == element])
        z_coords = z_coords - slab_bot  # adjust relative to slab bottom
        # compute histogram
        hist, bin_edges = np.histogram(z_coords, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        Data[element] = bin_centers, hist 

    return Data

@as_function_node
def add_ions(structure, cation: str, anion: str, n_ion_pairs: int = 2, seed: int = 42):
    """
    Replace random water moleculeswith ions.

    Parameters:
    structure (ase.Atoms): The input atomic structure.
    cation (str): The cation element symbol.
    anion (str): The anion element symbol.
    n_ion (int): The number of ion pairs to add. Default is 2.

    Returns:
    ase.Atoms: The modified structure with ions.
    """
    import numpy as np

    RNG = np.random.default_rng(int(seed))
    n_ion_pairs = int(n_ion_pairs)

    # find all oxygen indices in the structure
    water_indices = structure.select_index("O")

    if len(water_indices) < n_ion_pairs*2:
        raise ValueError("Not enough water molecules to replace with ions.")

    selected_indices = RNG.choice(water_indices, size=n_ion_pairs*2, replace=False)

    # replace selected oxygen atoms with ions
    for i in range(n_ion_pairs):
        o_index = selected_indices[i*2]
        structure[o_index] = cation
        
        o_index = selected_indices[i*2+1]
        structure[o_index] = anion
        
    # create list of hydrogen indices to remove from the selected water indices
    h_indices = []
    for o_index in selected_indices:
        h_indices.append(o_index + 1)
        h_indices.append(o_index + 2)

    # remove the hydrogen atoms from the structure
    del structure[h_indices]

    return structure

@as_function_node
def IonPotential(
    metal: str = "Al",
    metal_charge: float = 0.0,
    
    neon_charge: float = 0.0,
    electrode_epsilon: float = 0.102,
    electrode_sigma: float = 3.188,
    
    cation: str = "Na",
    cation_charge: float = 1.0,
    cation_epsilon: float = 0.102,
    cation_sigma: float = 1.188,
    
    anion: str = "F",
    anion_charge: float = -1.0,
    anion_epsilon: float = 0.102,
    anion_sigma: float = 1.188,
    
    ion_pair_epsilon: float = 0.102,
    ion_pair_sigma: float = 3.188,
):
    """
    Generate a DataFrame containing LAMMPS potential parameters for ions in water.
    
    Parameters:
    metal (str): The metal cathode element symbol.
    metal_charge (float): The charge of the metal cathode.
    neon_charge (float): The charge of the neon element.
    electrode_epsilon (float): The epsilon parameter for electrode-O interactions.
    electrode_sigma (float): The sigma parameter for electrode-O interactions.
    cation (str): The cation element symbol.
    cation_charge (float): The charge of the cation.
    cation_epsilon (float): The epsilon parameter for cation-O interactions.
    cation_sigma (float): The sigma parameter for cation-O interactions.
    anion (str): The anion element symbol.
    anion_charge (float): The charge of the anion.
    anion_epsilon (float): The epsilon parameter for anion-O interactions.
    anion_sigma (float): The sigma parameter for anion-O interactions.
    ion_pair_epsilon (float): The epsilon parameter for cation-anion interactions.
    ion_pair_sigma (float): The sigma parameter for cation-anion interactions.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the potential parameters.
    """
    import pandas

    ion_potential = pandas.DataFrame(
        {
            "Name": ["H2O_tip3p"],
            "Filename": [[]],
            "Model": ["TIP3P"],
            "Species": [["H", "O", metal, "Ne", cation, anion]],
            "Config": [
                [
                    "# @potential_species H_O  ### species in potential\n",
                    "# W.L. Jorgensen",
                    "The Journal of Chemical Physics 79",
                    "926 (1983); https://doi.org/10.1063/1.445869 \n",
                    "#\n",
                    "\n",
                    "units      real\n",
                    "dimension  3\n",
                    "atom_style full\n",
                    "\n",
                    "# create groups ###\n",
                    "group O type 2\n",         # Oxygen
                    "group H type 1\n",         # Hydrogen
                    f"group {metal} type 3\n",  # Metal Cathode
                    "group Ne type 4\n",        # Neon
                    f"group {cation} type 5\n", # Cation
                    f"group {anion} type 6\n",  # Anion
                    "\n",
                    "## set charges - beside manually ###\n",
                    "set group O charge -0.830\n",
                    "set group H charge 0.415\n",
                    f"set group {metal} charge {metal_charge}\n",
                    f"set group Ne charge {neon_charge}\n",
                    f"set group {cation} charge {cation_charge}\n",
                    f"set group {anion} charge {anion_charge}\n",
                    "\n",
                    "### TIP3P Potential Parameters ###\n",
                    "pair_style lj/cut/coul/long 10.0 \n", # Lang had cutoff of 12Å, why?
                    "pair_coeff * * 0.000 0.000 \n", # set all interactions to zero
                    "# self interactions \n",
                    "pair_coeff 1 1 0.000 0.000 \n",    # H-H
                    "pair_coeff 2 2 0.102 3.188 \n",    # O-O
                    "pair_coeff 3 3 0.350 2.62  \n",    # Al-Al
                    "pair_coeff 4 4 0.010 3.000 \n",    # Ne-Ne
                    "pair_coeff 5 5 0.102 1.188 \n",    # Cation
                    "pair_coeff 6 6 0.102 1.188 \n",    # Anion
                    "\n",
                    "# cross interactions \n",
                    "pair_coeff 2 3 {:.4} {:.4} \n".format(electrode_epsilon, electrode_sigma), # O-cathode
                    "pair_coeff 2 4 {:.4} {:.4} \n".format(electrode_epsilon, electrode_sigma), # O-Neon
                    "pair_coeff 2 5 {:.4} {:.4} \n".format(cation_epsilon, cation_sigma), # O-cation
                    "pair_coeff 2 6 {:.4} {:.4} \n".format(anion_epsilon, anion_sigma), # O-anion
                    "pair_coeff 3 5 {:.4} {:.4} \n".format(cation_epsilon, cation_sigma), # cathode-cation
                    "pair_coeff 3 6 {:.4} {:.4} \n".format(anion_epsilon, anion_sigma), # cathode-anion
                    "pair_coeff 4 5 {:.4} {:.4} \n".format(cation_epsilon, cation_sigma), # neon-cation
                    "pair_coeff 4 6 {:.4} {:.4} \n".format(anion_epsilon, anion_sigma), # neon-anion
                    "pair_coeff 5 6 {:.4} {:.4} \n".format(ion_pair_epsilon, ion_pair_sigma), # cation-anion
                    "\n",
                    "# Tiny H-X soft-core to prevent H-ion/metal collapse (critical!)\n",
                    "pair_coeff 1 3 0.010 1.60 \n", # H-cathode
                    "pair_coeff 1 4 0.010 1.60 \n", # H-anode
                    "pair_coeff 1 5 0.010 1.60 \n", # H-cation
                    "pair_coeff 1 6 0.020 1.60 \n", # H-anion (slightly stronger)
                    "\n",
                    "# bond and angle terms for water molecules ###\n",
                    "bond_style  harmonic\n",
                    "bond_coeff  1 450 0.9572\n",
                    "angle_style harmonic\n",
                    "angle_coeff 1 55 104.52\n",
                    "kspace_style pppm 1.0e-5   # final npt relaxation\n",
                    "\n",
                ]
            ],
        }
    )

    return ion_potential

@as_function_node("fig")
def PlotElementDistribution(element: str, Data: dict):
    """
    Plots the normalized density distribution of a specified element along the z-axis.
    
    Parameters
    ----------
    element : str
        The chemical symbol of the element to plot (e.g., "Na", "F").
    Data : dict
        A dictionary where keys are element symbols and values are tuples of (x, y) data.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """
    
    from matplotlib import pyplot as plt
    
    x, y = Data[element]

    fig, ax = plt.subplots()
    ax.plot(x, y, label=element)
    ax.set_xlabel("Position along z-axis (Å)")
    ax.set_ylabel("Normalized Density (1/Å)")
    return fig