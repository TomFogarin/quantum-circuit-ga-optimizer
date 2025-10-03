
import numpy as np
import itertools
import random
from typing import List, Tuple

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer


class HamiltonianGenerator:
    """
    A class to generate various types of Hamiltonians for quantum algorithms.
    """
    def __init__(self, j_range=(-3.0, 3.0), t_u_range=(0.0, 5.0), step=0.1, seed: int = None):
        """Initializes the HamiltonianGenerator class."""
        # Store a single, reusable mapper instance
        self.mapper = JordanWignerMapper()
        
        # Create and store the coupling constant grids
        # Generate all possible values for J1, J2, J3 on a uniform grid for the Heisn_xyz
        self.j_values = np.arange(j_range[0], j_range[1] + step, step)
        
        # Define the range and spacing for t and U coupling constants for Fermi Hubbard
        # np.arange(0.0, 5.01, 0.1) creates values [0.0, 0.1, ..., 5.0]
        self.t_u_values = np.arange(t_u_range[0], t_u_range[1] + step, step)

        self.seed = seed

    # =============================================================================
    # 1. CONDENSED MATTER SYSTEMS
    # =============================================================================

    def heisenberg_xyz(self, num_qubits: int, num_hamiltonians: int) -> list[SparsePauliOp]:
        """
        Generates a list of Heisenberg XYZ Hamiltonians as Qiskit SparsePauliOp objects.

        The Hamiltonian for the Heisenberg XYZ model is given by:
        H = sum_{i=0}^{n-2} (J1 * Xi*Xi+1 + J2 * Yi*Yi+1 + J3 * Zi*Zi+1)
        where Xi, Yi, Zi are Pauli operators on qubit i, and Xi+1, Yi+1, Zi+1 on qubit i+1.
        J1, J2, J3 are coupling constants.

        Args:
            num_qubits (int): The number of qubits (n) for the Hamiltonian. Must be at least 2.
            num_hamiltonians (int): The desired number of unique Hamiltonians to generate.
                                                This many (J1, J2, J3) tuples will be randomly sampled.

        Returns:
            list[SparsePauliOp]: A list of Qiskit SparsePauliOp objects, each representing
                                 a Heisenberg XYZ Hamiltonian with different (J1, J2, J3) constants.

        Raises:
            ValueError: If num_qubits is less than 2.
        """
        if num_qubits < 2:
            raise ValueError("Number of qubits must be at least 2 for nearest-neighbor interactions in Heisenberg XYZ model.")
        if num_hamiltonians <= 0:
            return []

        # Generate all possible combinations of (J1, J2, J3)
        # This creates a large list of all possible (J1, J2, J3) tuples from the grid.
        all_j_combinations = list(itertools.product(self.j_values, repeat=3))

        # Sample unique combinations if the requested number of Hamiltonians is less than
        # the total number of unique J-combinations available.
        if num_hamiltonians > len(all_j_combinations):
            print(f"Warning: Requested {num_hamiltonians} Hamiltonians, but only "
              f"{len(all_j_combinations)} unique (J1, J2, J3) combinations exist. "
              "Generating all available unique combinations.")
            selected_j_combinations = all_j_combinations
        else:
            # Randomly select the specified number of unique combinations
            rng = random.Random(self.seed) if self.seed is not None else random
            selected_j_combinations = rng.sample(all_j_combinations, num_hamiltonians)

        hamiltonian_list = []

        # Construct a SparsePauliOp for each selected (J1, J2, J3) combination
        for J1, J2, J3 in selected_j_combinations:
            pauli_terms = []

            # Iterate through qubits to build the sum of terms for the Hamiltonian
            # The sum goes from i=0 to num_qubits-2 for nearest-neighbor interactions (i and i+1)
            for i in range(num_qubits - 1):
                # Term: J1 * Xi * Xi+1
                # Create a list of 'I' (Identity) for all qubits, then set 'X' at positions i and i+1
                pauli_x_label = ['I'] * num_qubits
                pauli_x_label[i] = 'X'
                pauli_x_label[i+1] = 'X'
                pauli_terms.append(("".join(pauli_x_label), J1))

                # Term: J2 * Yi * Yi+1
                pauli_y_label = ['I'] * num_qubits
                pauli_y_label[i] = 'Y'
                pauli_y_label[i+1] = 'Y'
                pauli_terms.append(("".join(pauli_y_label), J2))

                # Term: J3 * Zi * Zi+1
                pauli_z_label = ['I'] * num_qubits
                pauli_z_label[i] = 'Z'
                pauli_z_label[i+1] = 'Z'
                pauli_terms.append(("".join(pauli_z_label), J3))

            # Create a SparsePauliOp object from the list of (Pauli string, coefficient) tuples.
            # SparsePauliOp automatically handles summing terms with identical Pauli strings.
            current_hamiltonian = SparsePauliOp.from_list(pauli_terms)
            hamiltonian_list.append(current_hamiltonian)

        return hamiltonian_list

    def twoD_ising(self, num_qubits: int, num_hamiltonians: int) -> list[SparsePauliOp]:
        """
        Generates a list of 2D Transverse-field Ising Hamiltonians as Qiskit SparsePauliOp objects.

        The Hamiltonian for the 2D Ising model on a square lattice is given by:
        H = -j * sum_{<i,j>} ZiZj - μ * sum_k Xk
        where <i,j> sums over nearest-neighbor pairs, Zi, Zj, Xk are Pauli operators,
        j is the interaction strength, and μ is the transverse field strength.

        Args:
            num_qubits (int): The total number of qubits. It must be a perfect square
                              (e.g., 4, 9, 16) to form a square 2D lattice.
            num_hamiltonians_to_generate (int): The desired number of unique Hamiltonians to generate.
                                                This many (j, μ) pairs will be randomly sampled.

        Returns:
            list[SparsePauliOp]: A list of Qiskit SparsePauliOp objects, each representing
                                 a 2D Transverse-field Ising Hamiltonian with different (j, μ) constants.

        Raises:
            ValueError: If num_qubits is not a perfect square (or <= 0), as a square lattice is assumed.
        """
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")

        # Determine the dimensions of the square 2D lattice
        rows = int(np.sqrt(num_qubits))
        if rows * rows != num_qubits:
            raise ValueError(f"num_qubits ({num_qubits}) must be a perfect square to form a square 2D lattice.")
        cols = rows # For a square lattice

        # --- 1. Identify Nearest-Neighbor Pairs on the 2D Grid ---
        nearest_neighbors = set() # Use a set to store unique pairs to avoid duplicates

        # Helper function to convert 2D grid coordinates (row, col) to a 1D qubit index
        def get_1d_index(r, c):
            return r * cols + c

        # Add horizontal connections (between qubits in the same row)
        for r in range(rows):
            for c in range(cols - 1):
                q1 = get_1d_index(r, c)
                q2 = get_1d_index(r, c + 1)
                # Store pairs as sorted tuples to ensure uniqueness regardless of order ((q1,q2) or (q2,q1))
                nearest_neighbors.add(tuple(sorted((q1, q2))))

        # Add vertical connections (between qubits in the same column)
        for r in range(rows - 1):
            for c in range(cols):
                q1 = get_1d_index(r, c)
                q2 = get_1d_index(r + 1, c)
                nearest_neighbors.add(tuple(sorted((q1, q2))))

        # --- 2. Generate Coupling Constants (j, μ) ---
        # Create the range of values for j and μ: [0, 5] with 0.1 spacing.
        # np.arange(0.0, 5.01, 0.1) ensures 5.0 is included due to floating-point precision.
        j_mu_values_grid = np.arange(0.0, 5.01, 0.1)

        # Generate all possible (j, μ) combinations from the grid
        all_j_mu_combinations = list(itertools.product(j_mu_values_grid, repeat=2))

        # Sample the requested number of unique combinations
        if num_hamiltonians > len(all_j_mu_combinations):
            print(f"Warning: Requested {num_hamiltonians} Hamiltonians, but only "
                f"{len(all_j_mu_combinations)} unique (j, μ) combinations exist on the grid. "
                 "Generating all available unique combinations instead.")
            selected_j_mu_combinations = all_j_mu_combinations
        else:
            # Randomly select the specified number of unique (j, μ) pairs
            rng = random.Random(self.seed) if self.seed is not None else random
            selected_j_mu_combinations = rng.sample(all_j_mu_combinations, num_hamiltonians)

        hamiltonian_list = []

        # --- 3. Construct SparsePauliOp for each selected (j, μ) combination ---
        for j, mu in selected_j_mu_combinations:
            pauli_terms = []

            # Add interaction terms: -j * ZiZj for each nearest-neighbor pair
            for q1, q2 in nearest_neighbors:
                # Create a Pauli string for ZiZj: 'I' for most qubits, 'Z' for q1 and q2
                label = ['I'] * num_qubits
                label[q1] = 'Z'
                label[q2] = 'Z'
                pauli_terms.append(("".join(label), -j)) # Coefficient is -j

            # Add transverse field terms: -μ * Xk for each individual qubit
            for k in range(num_qubits):
                # Create a Pauli string for Xk: 'I' for most qubits, 'X' for qubit k
                label = ['I'] * num_qubits
                label[k] = 'X'
                pauli_terms.append(("".join(label), -mu)) # Coefficient is -μ

            # Create a SparsePauliOp object from the list of (Pauli string, coefficient) tuples.
            # SparsePauliOp automatically handles combining terms with identical Pauli strings.
            current_hamiltonian = SparsePauliOp.from_list(pauli_terms)
            hamiltonian_list.append(current_hamiltonian)

        return hamiltonian_list
    
    
    def fermi_hubbard(self, num_sites: int, num_hamiltonians: int = 1000) -> list[SparsePauliOp]:
        """
        Generates a dataset (list) of 1D Fermi-Hubbard Hamiltonians.

        Each Hamiltonian is created by sampling (t, U) coupling constants from
        a uniform grid where t, U ∈ [0, 5] with a spacing of 0.1.

        Args:
            num_sites (int): The number of lattice sites for all Hamiltonians in the dataset.
                             Each Hamiltonian in the list will have `num_sites * 2` qubits.
            num_hamiltonians (int): The number of unique (t, U) Hamiltonians to generate.
                                 Defaults to 1000 as specified.

        Returns:
            list[SparsePauliOp]: A list of Qiskit SparsePauliOp objects, each representing
                                 a Fermi-Hubbard Hamiltonian with different (t, U) coupling constants.

        Raises:
            ValueError: If num_sites is less than 1 or num_hamiltonians is non-positive.
        """

        # Generate all possible combinations of (t, U) from the defined grid
        all_tu_combinations = list(itertools.product(self.t_u_values, repeat=2))

        # Sample the requested number of unique (t, U) pairs.
        # If num_hamiltonians is greater than the total possible combinations,
        # it will generate all available combinations.
        if num_hamiltonians > len(all_tu_combinations):
            print(f"Warning: Requested {num_hamiltonians} Hamiltonians, but only "
                  f"{len(all_tu_combinations)} unique (t, U) combinations exist on the grid. "
                  "Generating all available unique combinations instead.")
            selected_tu_combinations = all_tu_combinations
        else:
            rng = random.Random(self.seed) if self.seed is not None else random
            selected_tu_combinations = rng.sample(all_tu_combinations, num_hamiltonians)

        hubbard_hamiltonian_list = []

        # Generate a Hamiltonian for each sampled (t, U) pair
        for t_val, U_val in selected_tu_combinations:
            # Call the previously defined function to construct a single Fermi-Hubbard Hamiltonian
            hamiltonian = self._generate_single_fermi_hubbard(num_sites, t_val, U_val)
            hubbard_hamiltonian_list.append(hamiltonian)

        return hubbard_hamiltonian_list
    
    # =============================================================================
    # 2. QUANTUM CHEMISTRY
    # =============================================================================
    
    def molecular(self, molecule_name: str, distances: list) -> list[tuple[SparsePauliOp, float]]:
        """
        Generates a list of molecular Hamiltonians at various interatomic distances.

        Args:
            molecule_name (str): The name of the molecule. Supported: "H2", "LiH", "BeH2".
            distances (list): A list of bond distances in Angstroms.

        Returns:
            A list of tuples, each containing (qubit_hamiltonian, nuclear_repulsion_energy).
        """
        hamiltonians = []
        for d in distances:
            if molecule_name.upper() == "H2":
                hamiltonian, nre = self._h2_hamiltonian(d)
            elif molecule_name.upper() == "LIH":
                hamiltonian, nre = self._lih_hamiltonian(d)
            elif molecule_name.upper() == "BEH2":
                hamiltonian, nre = self._beh2_hamiltonian(d)
            else:
                raise ValueError(f"Molecule '{molecule_name}' is not supported.")
            hamiltonians.append((hamiltonian, nre))
        return hamiltonians
    
    # =============================================================================
    # 3. OPTIMIZATION PROBLEMS (QAOA)
    # =============================================================================

    def max_cut(self, num_qubits: int, num_hamiltonians: int) -> List[SparsePauliOp]:
        """
        Generates a list of random Max-Cut problem Hamiltonians.

        Each Hamiltonian corresponds to a random graph with n_qubits nodes.
        The Hamiltonian is constructed as: H = 1/2 * sum_{i,j in E} w_ij * (I - Zi Zj).

        Args:
            n_qubits (int): The number of nodes (qubits) for each random graph.
            num_hamiltonians (int): The number of random Hamiltonians to generate.

        Returns:
            List[SparsePauliOp]: A list of Max-Cut Hamiltonians.
        """
        if num_qubits < 2:
            raise ValueError("Number of qubits must be at least 2 for Max-Cut.")
        
        rng = random.Random(self.seed) if self.seed is not None else random
        hamiltonian_list = []

        # 1. Generate all possible unique edges for a graph of this size
        possible_edges = list(itertools.combinations(range(num_qubits), 2))

        for _ in range(num_hamiltonians):
            # 2. Determine a random number of edges for this specific graph
            max_possible_edges = len(possible_edges)
            min_edges = num_qubits - 1 # Ensure the graph is connected
            num_edges = rng.randint(min_edges, max_possible_edges)
            
            # 3. Randomly sample a unique subset of edges
            graph_edges = rng.sample(possible_edges, num_edges)
            
            # 4. Generate random weights for the selected edges
            weights = [rng.uniform(0.1, 5.0) for _ in range(num_edges)]
            
            terms = []
            total_weight = sum(weights)
            terms.append(('I' * num_qubits, -total_weight / 2.0))

            for (i, j), w in zip(graph_edges, weights):
                pauli_string = ['I'] * num_qubits
                pauli_string[i] = 'Z'
                pauli_string[j] = 'Z'
                terms.append((''.join(pauli_string), w / 2.0))
            
            hamiltonian_list.append(SparsePauliOp.from_list(terms))
            
        return hamiltonian_list

    def portfolio_optimization(self, n_assets: int, num_hamiltonians: int,
                               risk_penalty: float = 0.5) -> List[SparsePauliOp]:
        """
        Generates a list of random Portfolio Optimization problem Hamiltonians.

        The Hamiltonian is: H = -sum_{i} r_i * Zi + risk_penalty * sum_{i,j} sigma_ij * Zi * Zj.

        Args:
            n_assets (int): The number of assets (qubits) in the portfolio.
            num_hamiltonians (int): The number of random Hamiltonians to generate.
            risk_penalty (float, optional): A weighting factor for the risk term.

        Returns:
            List[SparsePauliOp]: A list of Portfolio Optimization Hamiltonians.
        """
        if n_assets <= 0:
            raise ValueError("Number of assets must be positive.")

        rng = np.random.default_rng(self.seed)
        hamiltonian_list = []

        for _ in range(num_hamiltonians):
            # For each sample, generate random returns and a random valid risk matrix
            returns = rng.uniform(-0.1, 0.2, size=n_assets)
            # Create a random positive semi-definite matrix for risk
            temp_matrix = rng.random((n_assets, n_assets))
            risk_matrix = np.dot(temp_matrix, temp_matrix.transpose())

            terms = []
            for i, r in enumerate(returns):
                pauli_string = ['I'] * n_assets
                pauli_string[i] = 'Z'
                terms.append((''.join(pauli_string), -r))

            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    pauli_string = ['I'] * n_assets
                    pauli_string[i] = 'Z'
                    pauli_string[j] = 'Z'
                    terms.append((''.join(pauli_string), risk_penalty * 2 * risk_matrix[i, j]))
            
            hamiltonian_list.append(SparsePauliOp.from_list(terms))
            
        return hamiltonian_list

    def traveling_salesman(self, n_cities: int, num_hamiltonians: int, 
                           penalty: float = 100.0) -> List[SparsePauliOp]:
        """
        Generates a list of random Traveling Salesman Problem (TSP) Hamiltonians.

        WARNING: The number of qubits is n_cities^2. This is only practical for n_cities <= 4.

        Args:
            n_cities (int): The number of cities in the problem.
            num_hamiltonians (int): The number of random Hamiltonians to generate.
            penalty (float, optional): A large constant to enforce constraints.

        Returns:
            List[SparsePauliOp]: A list of TSP Hamiltonians.
        """
        if n_cities < 2:
            raise ValueError("Number of cities must be at least 2.")
            
        rng = np.random.default_rng(self.seed)
        hamiltonian_list = []
        n_qubits = n_cities * n_cities

        for _ in range(num_hamiltonians):
            # For each sample, generate a random symmetric distance matrix
            positions = rng.random((n_cities, 2))
            distance_matrix = np.array([[np.linalg.norm(pos1 - pos2) for pos2 in positions] for pos1 in positions])

            terms = []
            # Part 1: Distance Objective
            for i in range(n_cities):
                for j in range(i + 1, n_cities):
                    for t in range(n_cities):
                        q1_idx = i * n_cities + t
                        q2_idx = j * n_cities + ((t + 1) % n_cities)
                        dist = distance_matrix[i, j]
                        
                        terms.append(('I' * n_qubits, dist / 4.0))
                        pauli_z1 = ['I'] * n_qubits; pauli_z1[q1_idx] = 'Z'; terms.append((''.join(pauli_z1), -dist / 4.0))
                        pauli_z2 = ['I'] * n_qubits; pauli_z2[q2_idx] = 'Z'; terms.append((''.join(pauli_z2), -dist / 4.0))
                        pauli_z1z2 = ['I'] * n_qubits; pauli_z1z2[q1_idx] = 'Z'; pauli_z1z2[q2_idx] = 'Z'; terms.append((''.join(pauli_z1z2), dist / 4.0))

            # Part 2: Constraints
            for t in range(n_cities): # Sum over cities for a fixed time
                for i in range(n_cities):
                    for j in range(i + 1, n_cities):
                        q1_idx = i * n_cities + t
                        q2_idx = j * n_cities + t
                        pauli_z1z2 = ['I'] * n_qubits; pauli_z1z2[q1_idx] = 'Z'; pauli_z1z2[q2_idx] = 'Z';
                        terms.append((''.join(pauli_z1z2), penalty / 4.0))

            for i in range(n_cities): # Sum over times for a fixed city
                for t1 in range(n_cities):
                    for t2 in range(t1 + 1, n_cities):
                        q1_idx = i * n_cities + t1
                        q2_idx = i * n_cities + t2
                        pauli_z1z2 = ['I'] * n_qubits; pauli_z1z2[q1_idx] = 'Z'; pauli_z1z2[q2_idx] = 'Z';
                        terms.append((''.join(pauli_z1z2), penalty / 4.0))

            hamiltonian_list.append(SparsePauliOp.from_list(terms).reduce())
            
        return hamiltonian_list

    # =============================================================================
    # 4. RANDOM MATRIX THEORY
    # =============================================================================
    
    def gaussian_orthogonal_ensemble(
        self,
        num_qubits: int,
        num_hamiltonians: int,
        sigma: float = 1.0
    ) -> List[SparsePauliOp]:
        """
        Generates a list of random Hamiltonians from the Gaussian Orthogonal Ensemble (GOE)
        that are guaranteed to be Hermitian.
        """
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")

        hamiltonian_list = []
        rng = np.random.default_rng(self.seed)
        paulis = ['I', 'X', 'Y', 'Z']

        # Generate all 4^n possible Pauli strings
        all_pauli_strings = ["".join(p) for p in itertools.product(paulis, repeat=num_qubits)]

        for _ in range(num_hamiltonians):
            terms = []
            for pauli_string in all_pauli_strings:
                if 'I' * num_qubits == pauli_string:
                    continue
                # Only include Pauli strings with an EVEN number of Y operators.
                if pauli_string.count('Y') % 2 == 0:
                    coeff = rng.normal(0, sigma)
                    terms.append((pauli_string, coeff))

            hamiltonian = SparsePauliOp.from_list(terms).simplify()
            hamiltonian_list.append(hamiltonian)

        return hamiltonian_list


    def sparse_random_hamiltonian(
        self,
        num_qubits: int,
        num_hamiltonians: int,
        density: float = 0.1,
        max_pauli_weight: int = 2
    ) -> List[SparsePauliOp]:
        """
        Generates a list of sparse random Hamiltonians with limited Pauli weight.

        This more closely models typical Hamiltonians found in quantum algorithms,
        which are often sparse and composed of low-weight (k-local) terms.

        Args:
            n_qubits (int): The number of qubits for each Hamiltonian.
            num_hamiltonians (int): The total number of Hamiltonians to create.
            density (float, optional): The approximate fraction of possible k-local terms
                                       to include. Must be between 0 and 1. Defaults to 0.1.
            max_pauli_weight (int, optional): The maximum number of non-identity Pauli operators
                                              in any single term (k in k-local). Defaults to 2.

        Returns:
            List[SparsePauliOp]: A list of sparse random Hamiltonians.
        """
        if not (0 < density <= 1.0):
            raise ValueError("Density must be between 0 and 1.")
        if max_pauli_weight < 1 or max_pauli_weight > num_qubits:
            raise ValueError("max_pauli_weight must be between 1 and n_qubits.")

        hamiltonian_list = []
        rng = np.random.default_rng(self.seed)
        paulis = ['X', 'Y', 'Z']

        # Pre-calculate all possible terms up to max_pauli_weight
        possible_terms = []
        for k in range(1, max_pauli_weight + 1):
            for positions in itertools.combinations(range(num_qubits), k):
                for operators in itertools.product(paulis, repeat=k):
                    pauli_string = ['I'] * num_qubits
                    for pos, op in zip(positions, operators):
                        pauli_string[pos] = op
                    possible_terms.append("".join(pauli_string))
    
        num_terms_to_select = int(density * len(possible_terms))
        if num_terms_to_select < 1:
            raise ValueError("Density is too low to select any terms. Increase density or max_pauli_weight.")

        for _ in range(num_hamiltonians):
            # For each Hamiltonian, sample a subset of the possible terms
            selected_strings = rng.choice(possible_terms, size=num_terms_to_select, replace=False)
        
            # Assign random real coefficients
            coeffs = rng.normal(0, 1, size=len(selected_strings))
          
            hamiltonian_list.append(SparsePauliOp.from_list(list(zip(selected_strings, coeffs))))

        return hamiltonian_list
    
    def power_law_hamiltonian(
        self,
        num_qubits: int,
        num_hamiltonians: int,
        alpha: float = 2.0
    ) -> List[SparsePauliOp]:
        """
        Generates a list of random Hamiltonians with power-law decaying interactions.

        This type of Hamiltonian is useful for modeling systems with long-range
        interactions, where the strength of the interaction between two sites
        decays as a function of their distance. The Hamiltonian contains all-to-all
        two-qubit Pauli terms.

        Args:
            n_qubits (int): The number of qubits for each Hamiltonian.
            num_hamiltonians (int): The total number of Hamiltonians to create.
            alpha (float, optional): The exponent of the power-law decay.
                                     Higher values lead to more local interactions.
                                     Defaults to 2.0.

        Returns:
            List[SparsePauliOp]: A list of power-law random Hamiltonians.
        """
        if num_qubits < 2:
            raise ValueError("Number of qubits must be at least 2 for pairwise interactions.")

        hamiltonian_list = []
        rng = np.random.default_rng(self.seed)
        paulis = ['X', 'Y', 'Z']

        for _ in range(num_hamiltonians):
            terms = []
            # Generate terms with power-law distributed ranges for all pairs
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    distance = float(j - i)

                    # Strength decays with distance^(-alpha), with a random sign and magnitude
                    strength = (distance**(-alpha)) * rng.normal(0, 1)

                    # Randomly choose the type of Pauli interaction (XX, XY, XZ, YX, etc.)
                    pauli_i = rng.choice(paulis)
                    pauli_j = rng.choice(paulis)

                    pauli_string = ['I'] * num_qubits
                    pauli_string[i] = pauli_i
                    pauli_string[j] = pauli_j

                    terms.append((''.join(pauli_string), strength))

            hamiltonian_list.append(SparsePauliOp.from_list(terms))

        return hamiltonian_list


    # --- Private Helper Methods ---

    
    def _generate_single_fermi_hubbard(self, num_sites: int, t: float, U: float) -> SparsePauliOp:
            
            """
            Generates a Qiskit SparsePauliOp representing the 1D Fermi-Hubbard Hamiltonian
            with open boundary conditions, mapped to qubits using the Jordan-Wigner transformation.

            The Hamiltonian terms include:
            - Hopping (kinetic energy): -t * sum_{<i,j>, sigma} (c_i_sigma+ c_j_sigma + H.c.)
            - On-site interaction (potential energy): U * sum_i n_i_up n_i_down

            Args:
                num_sites (int): The number of lattice sites. Each site contributes two spin orbitals.
                                 So, the total number of spin orbitals will be `num_sites * 2`.
                                 The resulting qubit Hamiltonian will have `num_sites * 2` qubits.
                t (float): The hopping (kinetic) energy strength.
                U (float): The on-site (potential) interaction energy strength.

            Returns:
                SparsePauliOp: The qubit Hamiltonian for the Fermi-Hubbard model.

            Raises:
                ValueError: If num_sites is less than 1.
            """
            if num_sites < 1:
                raise ValueError("Number of sites must be at least 1.")

            num_spin_orbitals = num_sites * 2
            fermi_op_terms = {}

            # 1. Hopping terms (-t * (c_i_sigma+ c_j_sigma + c_j_sigma+ c_i_sigma))
            for i in range(num_sites - 1):  # Loop over bonds (i, i+1) for open boundary conditions
                # Spin up hopping
                orb_i_up = 2 * i
                orb_j_up = 2 * (i + 1)
                fermi_op_terms[f"+_{orb_i_up} -_{orb_j_up}"] = -t
                fermi_op_terms[f"+_{orb_j_up} -_{orb_i_up}"] = -t

                # Spin down hopping
                orb_i_down = 2 * i + 1
                orb_j_down = 2 * (i + 1) + 1
                fermi_op_terms[f"+_{orb_i_down} -_{orb_j_down}"] = -t
                fermi_op_terms[f"+_{orb_j_down} -_{orb_i_down}"] = -t

            # 2. On-site interaction terms (U * n_i_up n_i_down)
            for i in range(num_sites):
                orb_i_up = 2 * i
                orb_i_down = 2 * i + 1
                fermi_op_terms[f"+_{orb_i_up} -_{orb_i_up} +_{orb_i_down} -_{orb_i_down}"] = U

            fermi_hamiltonian = FermionicOp(fermi_op_terms, num_spin_orbitals=num_spin_orbitals)
            
            qubit_hamiltonian = self.mapper.map(fermi_hamiltonian)
    
            return qubit_hamiltonian
    
    # Make sure you have these versions installed:
    # pip install qiskit-nature==0.7.2 qiskit-terra==0.42.0 pyscf

    def _create_molecular_problem(self, atom_string: str, charge: int, spin: int, basis: str = "sto3g"):
        """Private helper to run the chemistry driver."""
        driver = PySCFDriver(atom=atom_string, charge=charge, spin=spin, basis=basis)
        return driver.run()
    
    def _h2_hamiltonian(self, distance: float):
        problem = self._create_molecular_problem(f"H 0 0 0; H 0 0 {distance}", 0, 0)
        transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
        problem = transformer.transform(problem)
        fermionic_op = problem.hamiltonian.second_q_op()
        return self.mapper.map(fermionic_op), problem.nuclear_repulsion_energy
    
    def _lih_hamiltonian(self, distance: float):
        problem = self._create_molecular_problem(f"Li 0 0 0; H 0 0 {distance}", 0, 0)
        transformer = ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=4)
        problem = transformer.transform(problem)
        fermionic_op = problem.hamiltonian.second_q_op()
        return self.mapper.map(fermionic_op), problem.nuclear_repulsion_energy
    
    def _beh2_hamiltonian(self, distance: float):
        problem = self._create_molecular_problem(f"Be 0 0 0; H 0 0 {-distance}; H 0 0 {distance}", 0, 0)
        transformer = ActiveSpaceTransformer(num_electrons=6, num_spatial_orbitals=7)
        problem = transformer.transform(problem)
        fermionic_op = problem.hamiltonian.second_q_op()
        return self.mapper.map(fermionic_op), problem.nuclear_repulsion_energy