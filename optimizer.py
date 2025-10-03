
import random
import numpy as np
import multiprocessing

from deap import creator, base, tools, algorithms
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import library, Gate
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import StatevectorEstimator 


# --- Define the DEAP creator ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, num_qubits=None,
               min_depth=None, max_depth=None, gateset=None)

class Individual(list):
    """
    Represents a quantum circuit as a list of layers for the genetic algorithm.
    """
    def __init__(self, iterable: iter, num_qubits: int, min_depth: int, max_depth: int, gateset: list, p2gates: float, p1gate:float):
        super().__init__(iterable)
        self.fitness = creator.FitnessMin()
        self.num_qubits = num_qubits
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.gateset = gateset
        self.p2gates = p2gates
        self.p1gate = p1gate

    @classmethod
    def from_random_gates(cls, num_qubits: int, min_depth: int, max_depth: int, gateset: list, p2gates: float, p1gate:float):
        """
        Class method to generate a random circuit individual using a specific gateset.
        """
        gates_map = {
            name: gate for name, gate in library.get_standard_gate_name_mapping().items()
            if name in gateset
        }
        qc_list = []
        depth = random.randint(min_depth, max_depth)
        for _ in range(depth):
            layer, available_qubits = [], list(range(num_qubits))
            random.shuffle(available_qubits)
            
            while len(available_qubits) > 0:
                gate = cls._random_gate(len(available_qubits), gates_map, p2gates, p1gate)
                if gate is None: break
                
                num_gate_qubits = gate.num_qubits
                selected_qubits = tuple(available_qubits[:num_gate_qubits])
                layer.append((gate, selected_qubits))
                available_qubits = available_qubits[num_gate_qubits:]
            
            if layer: qc_list.append(layer)

        return cls(qc_list, num_qubits, min_depth, max_depth, gateset, p2gates, p1gate)

    def build_circuit(self) -> QuantumCircuit:
        """Builds and returns a Qiskit QuantumCircuit object."""
        qc = QuantumCircuit(self.num_qubits)
        for layer in self:
            for gate_obj, qubit_indices in layer:
                qc.append(gate_obj, list(qubit_indices))
        return qc
    
    @staticmethod
    def _random_gate(max_qubits: int, gates_map: dict, p2gates: float, p1gate:float) -> Gate:
        """Static helper to select a random gate from a provided map."""
        filtered_gates = [g for g in gates_map.values() if g.num_qubits <= max_qubits]
        if not filtered_gates: return None
        
        weights = []
        for g in filtered_gates:
            if g.num_qubits == 1: weights.append(p1gate) #probability to select 1-qubit gates
            elif g.num_qubits == 2: weights.append(p2gates) #probability to select 2-qubit gates
            else: weights.append(0.1)
    
        gate = random.choices(filtered_gates, weights=weights, k=1)[0].copy()
        for index in range(len(gate.params)):
            gate.params[index] = random.uniform(0, 2 * np.pi)
        return gate
    
    
# --- Global helper for multiprocessing ---

# Objective Function of the GA to change depending on the task
def evaluate_objective(individual, estimator, H: SparsePauliOp) -> tuple[float]:
    '''
    Evaluate the quantum circuit by computing the expectation value with random parameters
    and adding complexity penalty (counting 2-qubit gates as 2 gates).
    '''
    qc = individual.build_circuit()
    
    # Randomize all parameters for structure-focused evaluation
    for instruction in qc.data:
        if hasattr(instruction.operation, 'params') and instruction.operation.params:
            for i in range(len(instruction.operation.params)):
                instruction.operation.params[i] = random.uniform(0, 2 * np.pi)
    
    # Evaluate with random parameters
    job = estimator.run([(qc, H)])   # Running 1 VQE iteration
    base_score = job.result()[0].data["evs"].item()
    
    # Calculate complexity penalty - count 2-qubit gates as 2 gates
    gate_count = 0
    for layer in individual:
        for gate, _ in layer:
            if gate.num_qubits == 1:
                gate_count += 1
            elif gate.num_qubits == 2:
                gate_count += 2  # Count 2-qubit gates as 2 gates
            else:
                gate_count += gate.num_qubits  # General case for n-qubit gates
    
    # Add penalty for circuit complexity
    complexity_penalty = gate_count * 0.001  
    
    return (base_score + complexity_penalty,)


class GeneticAlgorithmOptimizer:
    """
    Finds an optimal VQE ansatz for a given Hamiltonian using a genetic algorithm.
    """
    def __init__(self, H: SparsePauliOp, ngen: int = 100, npop: int = 150,
                 min_depth: int = 4, max_depth: int = 8, cxpb: float = 0.8,
                 mutpb: float = 0.6, tourn_ratio: float = 0.05, gateset: list = ['cx', 'ry', 'rz'], p2gates: float = 0.1, p1gate:float = 0.15):
        """
        Initializes and configures the genetic algorithm.
        """
        self.H = H
        self.ngen = ngen
        self.npop = npop
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.gateset = gateset
        self.p2gates = p2gates
        self.p1gate = p1gate
        self.tourn_size = int(tourn_ratio * npop)

        # Pass necessary attributes to the creator for DEAP's internals
        creator.Individual.num_qubits = H.num_qubits
        creator.Individual.min_depth = self.min_depth
        creator.Individual.max_depth = self.max_depth
        creator.Individual.gateset = self.gateset
        
        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _setup_toolbox(self):
        """Configures the DEAP toolbox with all necessary functions."""
        self.toolbox.register('individual', Individual.from_random_gates, 
                              num_qubits=self.H.num_qubits, min_depth=self.min_depth, 
                              max_depth=self.max_depth, gateset= self.gateset, p2gates=self.p2gates, p1gate=self.p1gate)
        
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxOnePoint) # Here you can select different crossover methods
        self.toolbox.register("mutate", self._mutate) # Custom mutation method
        self.toolbox.register("select", tools.selTournament, tournsize=self.tourn_size) # Here you can select different selection methods
        
        # Register the objective function with all its dependencies
        # Change this 2 lines if needed to change the objective function
        estimator = StatevectorEstimator()
        self.toolbox.register('evaluate', evaluate_objective, H=self.H, estimator=estimator)
        
    def run(self) -> tuple[QuantumCircuit, list, tools.Logbook]:
        """
        Executes the genetic algorithm and returns the best found circuit.
        """

        with multiprocessing.Pool() as pool:
            self.toolbox.register("map", pool.map)
            
            stats = tools.Statistics(key=lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            
            best = tools.HallOfFame(1)
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

            # --- Genetic Algorithm Main Loop ---
            population = self.toolbox.population(self.npop)
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            best.update(population)

            record = stats.compile(population)
            logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
            # Start the generational loop
            for gen in range(1, self.ngen + 1):
                # Select the next generation individuals
                offspring = self.toolbox.select(population, self.npop)
            
                # Vary the pool of individuals
                offspring = algorithms.varAnd(offspring, self.toolbox, self.cxpb, self.mutpb)
            
                # --- FILTERING STEP ---
                # Discard individuals that exceed the maximum allowed depth
                valid_offspring = [ind for ind in offspring if len(ind) <= self.max_depth]
            
                # If the population size has shrunk, generate more individuals
                while len(valid_offspring) < self.npop:
                    new_ind = self.toolbox.individual()
                    # If the new individual is also valid, add it
                    if len(new_ind) <= self.max_depth:
                        valid_offspring.append(new_ind)
                # ---------------------------
            
                # Evaluate the individuals with an unvalued fitness
                invalid_ind = [ind for ind in valid_offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Update the hall of fame with the new population
                best.update(valid_offspring)

                # Replace the old population by the offspring
                population[:] = valid_offspring

                # Append the current generation statistics to the logbook
                record = stats.compile(population)
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            best_circuit = best[0].build_circuit()
            
            #removing redondant gates
            qc_best_simplified, initial_params = self._simplify_circuit(best_circuit)

        
        return qc_best_simplified, initial_params, logbook
    

    # --- Private Mutation Helper Methods ---
    
    def _mutate(self, qc: Individual, **weights: float) -> tuple[creator.Individual]:

        # These weights set the probability of each mutation type if the mutaion is selected
        weights.setdefault('insert', 3)
        weights.setdefault('delete', 2)
        weights.setdefault('flip', 2)
        weights.setdefault('layers', 1.5)
        weights.setdefault('qubits', 1.5)
        weights.setdefault('hoist', 0.5)
        weights.setdefault('split', 1.0)       
        weights.setdefault('merge', 0.5)
        #weights.setdefault('params', 0.01) #not needed anymore for structure optimization
    
        key = random.choices(list(weights.keys()), list(weights.values()))[0]
        match key:
            case 'insert':
                self._insert_mutation(qc, qc.gateset)
            case 'delete':
                self._delete_mutation(qc, qc.gateset)
            case 'flip':
                self._gate_flip(qc, qc.gateset)
            case 'layers':
                self._swap_layers(qc, qc.gateset)
            case 'qubits':
                self._swap_qubits(qc, qc.gateset)
            case 'hoist':
                self._hoist_mutation(qc, qc.gateset)
            case 'params':
                self._parameters_mutation(qc, qc.gateset)
        return qc,
        
    def _gate_flip(self,qc: Individual, gateset: list) -> tuple[Individual]:
        '''
        Replace a gate with a new random gate

        Parameters
        ----------
        qc : Individual
            Quantum circuit to mutate.

        Returns
        -------
        qc : tuple[Individual]
            Mutated quantum circuit. 7

        '''
        # Create the gates_map from the provided gateset
        gates_map = {name: gate for name, gate in library.get_standard_gate_name_mapping().items() if name in gateset}
        for i in random.sample(range(len(qc)), len(qc)):
            if len(qc[i]) > 0:
                j = random.randrange(len(qc[i]))
                del qc[i][j]
            
                qubits = list(range(qc.num_qubits))
                for gate, used_qubits in qc[i]:
                    qubits = list(filter(lambda x: not x in used_qubits, qubits))
                            
                gate = Individual._random_gate(len(qubits), gates_map, self.p2gates, self.p1gate)            
                qc[i].append((gate, random.sample(qubits, gate.num_qubits)))
                return qc,
        return qc,


    def _swap_layers(self, qc: Individual, gateset: list) -> tuple[Individual]:
        '''
        Swap two random layers

        Parameters
        ----------
        qc : Individual
            Quantum circuit to mutate.

        Returns
        -------
        qc : Individual
            Mutated quantum circuit.

        '''
        if len(qc) >= 2:
            i, j = random.sample(range(len(qc)), 2)
            qc[i], qc[j] = qc[j], qc[i]
        return qc,


    def _swap_qubits(self, qc: Individual, gateset: list) -> tuple[Individual]:
        '''
        Swap two qubits lines

        Parameters
        ----------
        qc : Individual
            Quantum circuit to mutate.

        Returns
        -------
        qc : Individual
            Mutated quantum circuit.

        '''
        if qc.num_qubits >= 2:
            i, j = random.sample(range(qc.num_qubits), 2)
            for k in range(len(qc)):
                for l in range(len(qc[k])):
                    t = []
                    for m in range(len(qc[k][l][1])):
                        if qc[k][l][1][m] == i:
                            t.append(j)
                        elif qc[k][l][1][m] == j:
                            t.append(i)
                        else:
                            t.append(qc[k][l][1][m])
                    t = tuple(t)
                    qc[k][l] = (qc[k][l][0], t)
        return qc, 


    def _parameters_mutation(self, qc: Individual, gateset: list) -> tuple[Individual]:
        '''
        Mutate a random parameter of a random gate

        Parameters
        ----------
        qc : Individual
           Quantum circuit to mutate.

        Returns
        -------
        qc : Individual
            Mutated quantum circuit.

        '''
        std=0.05  # Standard deviation for Gaussian mutation
        for i in random.sample(range(len(qc)), len(qc)):
            for j in random.sample(range(len(qc[i])), len(qc[i])):
                # Check if the gate has parameters before trying to access them
                if len(qc[i][j][0].params) > 0:
                    k = random.choice(range(len(qc[i][j][0].params)))
                    try:
                        fitness_val = abs(qc.fitness.values[0])
                        std = min(max(fitness_val * 0.1, 0.01), 0.1)  # Between 0.01 and 0.1
                    except (IndexError, AttributeError):
                        std = 0.05  
                
                    old_param = qc[i][j][0].params[k]
                    new_param = random.gauss(old_param, std)
                    # Keep parameters in [0, 2π) range
                    qc[i][j][0].params[k] = new_param % (2 * np.pi)
                    return qc,
        # If no gate with parameters was found, return the original circuit
        return qc,


    def _insert_mutation(self, qc: Individual, gateset: list) -> tuple[Individual]:
        '''
        Tries to add a random gate in a random layer, if it can't and it's depth 
        is less than max_depth create a new layer with the new gate

        Parameters
        ----------
        qc : Individual
            Quantum circuit to mutate.

        Returns
        -------
        qc : Individual
            Mutated quantum circuit.

        '''
        # Create the gates_map from the provided gateset
        gates_map = {name: gate for name, gate in library.get_standard_gate_name_mapping().items() if name in gateset}

        layers = random.sample(range(len(qc)), len(qc))
        for i in layers:
            qubits = list(range(qc.num_qubits))
            for gate, used_qubits in qc[i]:
                qubits = list(filter(lambda x: not x in used_qubits, qubits))
        
            if len(qubits) > 0:
                gate = Individual._random_gate(len(qubits), gates_map, self.p2gates, self.p1gate)
                qc[i].append((gate, random.sample(qubits, gate.num_qubits)))
                return qc,
    
        # Tries to create a new layer
        if len(qc) < qc.max_depth:
            gate = Individual._random_gate(qc.num_qubits, gates_map, self.p2gates, self.p1gate)
            qubits = random.sample(range(qc.num_qubits), gate.num_qubits)
            qc.append([(gate, qubits)])
        return qc,


    def _delete_mutation(self, qc: Individual, gateset: list) -> tuple[Individual]:
        '''
        Tries to delete a single gate without leaving an empty layer

        Parameters
        ----------
        qc : Individual
            Quantum circuit to mutate.

        Returns
        -------
        qc : Individual
            Mutated quantum circuit.

        '''
        layers = random.sample(range(len(qc)), len(qc))
        for i in layers:
            if len(qc[i]) > 1:
                del qc[i][random.randrange(len(qc[i]))]
                return qc,
    
        # Tries to delete a layer
        if len(qc) > qc.min_depth:
            i = random.randrange(len(qc))
            del qc[i]
        return qc,


    def _hoist_mutation(self, qc: Individual, gateset: list) -> tuple[Individual]:
        '''
        Tries to delete a many layers to reduce bloating

        Parameters
        ----------
        qc : Individual
            Quantum circuit to mutate.

        Returns
        -------
        qc : Individual
            Mutated quantum circuit.

        '''
        if len(qc) <= qc.min_depth:
            return qc,
    
        i = random.randrange(qc.min_depth-1, len(qc)-1)
        j = random.randrange(i+1, len(qc))
        qc[i:] = qc[j:]
    
        return qc,

    def _split_layer_mutation(self, qc: Individual, gateset: list) -> tuple[Individual]:
        '''Split a layer into two layers for more fine-grained structure'''
        if len(qc) >= qc.max_depth or len(qc) == 0:
            return qc,
            
        layer_idx = random.randrange(len(qc))
        if len(qc[layer_idx]) <= 1:  # Can't split single gate layers meaningfully
            return qc,
            
        # Split the layer randomly
        split_point = random.randrange(1, len(qc[layer_idx]))
        first_half = qc[layer_idx][:split_point]
        second_half = qc[layer_idx][split_point:]
        
        # Replace original layer with first half and insert second half after
        qc[layer_idx] = first_half
        qc.insert(layer_idx + 1, second_half)
        
        return qc,

    def _merge_layers_mutation(self, qc: Individual, gateset: list) -> tuple[Individual]:
        '''Merge two adjacent layers if they don't conflict'''
        if len(qc) <= qc.min_depth or len(qc) < 2:
            return qc,
            
        # Pick two adjacent layers
        first_layer_idx = random.randrange(len(qc) - 1)
        second_layer_idx = first_layer_idx + 1
        
        # Check if layers can be merged (no qubit conflicts)
        used_qubits_first = set()
        for _, qubits in qc[first_layer_idx]:
            used_qubits_first.update(qubits)
            
        used_qubits_second = set()
        for _, qubits in qc[second_layer_idx]:
            used_qubits_second.update(qubits)
            
        # If no overlap, merge is safe
        if not used_qubits_first.intersection(used_qubits_second):
            merged_layer = qc[first_layer_idx] + qc[second_layer_idx]
            qc[first_layer_idx] = merged_layer
            del qc[second_layer_idx]
        
        return qc,


    def _simplify_circuit(self, qc: QuantumCircuit) -> tuple[QuantumCircuit, list[float]]:
        """
        Simplifies a quantum circuit by combining adjacent single-qubit rotation gates
        (Rx, Ry, Rz) on the same qubit.
        It returns a new circuit with Qiskit Parameter objects for the combined angles
        and a list of their initial numerical values.

        This method is applied at the end only to the best circuit found by the GA to 
        reduce algorithm complexity, and it's very important if you chose a basic gate 
        set (e.g. ['cx', 'ry', 'rz']) to not have repetions.
        """
        simplified_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)

        gate_constructors = {
            'rx': RXGate,
            'ry': RYGate,
            'rz': RZGate
        }

        last_rotations = {} # {qubit_index: {'type': 'rx', 'angle': value_float}}
        initial_parameter_values = [] # To store the numerical values for the new Parameter objects

        for instruction in qc.data:
            op = instruction.operation
            qubits = instruction.qubits
            q_indices = [qc.qubits.index(q) for q in qubits] # Corrected Qubit to index mapping

            # Extract numerical value of parameters from original gate (assuming GA outputs numerical params)
            current_op_params_numerical = []
            if hasattr(op, 'params') and op.params:
                for p in op.params:
                    current_op_params_numerical.append(float(p))
        
            # Check if it's a single-qubit parameterized rotation gate (Rx, Ry, Rz)
            if op.num_qubits == 1 and op.name in gate_constructors and len(current_op_params_numerical) == 1:
                q_idx = q_indices[0] # Only one qubit for single-qubit gates
                current_angle = current_op_params_numerical[0] # Use the numerical value

                if q_idx in last_rotations and last_rotations[q_idx]['type'] == op.name:
                    # Combine with the last rotation of the same type on this qubit
                    last_rotations[q_idx]['angle'] += current_angle
                else:
                    # This path means a different type of rotation or no prior rotation on this qubit.
                    # Flush any pending rotation for this specific qubit before starting a new one.
                    if q_idx in last_rotations:
                        rot_type = last_rotations[q_idx]['type']
                        rot_angle_val = last_rotations[q_idx]['angle'] # Get accumulated numerical angle
                    
                        # Fold angle to [0, 2pi) if it's a rotation, to keep it within typical VQE bounds
                        rot_angle_val = rot_angle_val % (2 * np.pi) 

                        if abs(rot_angle_val) > 1e-9: # Only add if not effectively 0
                            # Create a NEW Qiskit Parameter for this combined angle
                            new_param = Parameter(f'{rot_type}_{q_idx}_{len(initial_parameter_values)}')
                            simplified_qc.append(gate_constructors[rot_type](new_param), [q_idx])
                            initial_parameter_values.append(rot_angle_val) # Store its numerical initial value
                        del last_rotations[q_idx]

                    # Start tracking new rotation
                    last_rotations[q_idx] = {'type': op.name, 'angle': current_angle}
            else:
                # If it's a non-rotation gate (like CX, H) or a multi-qubit gate,
                # or a single-qubit gate that's not Rx/Ry/Rz.
                # Flush all pending rotations on the *involved qubits* before adding this gate.
                for q_idx_in_current_op in q_indices: # Iterate only over qubits current instruction acts upon
                    if q_idx_in_current_op in last_rotations: # If this qubit has a pending rotation
                        rot_type = last_rotations[q_idx_in_current_op]['type']
                        rot_angle_val = last_rotations[q_idx_in_current_op]['angle']
                        rot_angle_val = rot_angle_val % (2 * np.pi) # Fold angle

                        if abs(rot_angle_val) > 1e-9:
                            # Create a NEW Qiskit Parameter for this combined angle
                            new_param = Parameter(f'{rot_type}_{q_idx_in_current_op}_{len(initial_parameter_values)}')
                            simplified_qc.append(gate_constructors[rot_type](new_param), [q_idx_in_current_op])
                            initial_parameter_values.append(rot_angle_val) # Store its numerical initial value
                        del last_rotations[q_idx_in_current_op]
            
                # Now append the current non-rotation/multi-qubit instruction.
                # If the original 'op' had Qiskit Parameter objects, they are carried over here.
                # If it had numerical parameters, they are also carried over (and won't be optimized by VQE).
                simplified_qc.append(op, qubits, instruction.clbits)

        # Flush any remaining pending rotations at the end of the circuit
        for q_idx in last_rotations:
            rot_type = last_rotations[q_idx]['type']
            rot_angle_val = last_rotations[q_idx]['angle']
            rot_angle_val = rot_angle_val % (2 * np.pi) # Fold angle

            if abs(rot_angle_val) > 1e-9:
                # Create a NEW Qiskit Parameter for this combined angle
                new_param = Parameter(f'{rot_type}_{q_idx}_{len(initial_parameter_values)}')
                simplified_qc.append(gate_constructors[rot_type](new_param), [q_idx])
                initial_parameter_values.append(rot_angle_val) # Store its numerical initial value

        return simplified_qc, initial_parameter_values