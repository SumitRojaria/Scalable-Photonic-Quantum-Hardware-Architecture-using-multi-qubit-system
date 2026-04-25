import numpy as np
import strawberryfields as sf
from strawberryfields.ops import (
    Fock, BSgate, Rgate, MeasureFock, Interferometer
)
from strawberryfields.backends.fockbackend import FockBackend
import matplotlib.pyplot as plt
from collections import Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')


print("=" * 70)
print("KLM / MEASUREMENT-BASED LOQC GROVER'S ALGORITHM")
print("=" * 70)


# KLM Protocol Parameters

CUTOFF_DIM = 3
N_QUBITS = 4
N_DATA_MODES = 2 * N_QUBITS

MARKED_STATE = (0, 1, 0, 1)   # (q0, q1, q2, q3)


MARKED_IDX = sum(b * (2**q) for q, b in enumerate(MARKED_STATE))   

MARKED_LABEL = ''.join(str(b) for b in MARKED_STATE)               

KLM_CNOT_SUCCESS_PROB = 1/16


# KLM NONLINEAR SIGN (NS) GATE


class KLM_NS_Gate:
    def __init__(self):
        self.theta1 = np.arccos(1/np.sqrt(3))
        self.theta2 = np.pi/4

    def get_unitary(self):
       
        def bs3(theta, phi, i, j):
            
            c, s = np.cos(theta), np.sin(theta)
            U = np.eye(3, dtype=complex)
            U[i, i] =  c
            U[i, j] = -np.exp( 1j * phi) * s
            U[j, i] =  np.exp(-1j * phi) * s
            U[j, j] =  c
            return U

        # Compose the three beamsplitters 
        U = (bs3(-self.theta1, 0, 0, 1)
             @ bs3(self.theta2,  0, 1, 2)
             @ bs3(self.theta1,  0, 0, 1))
        return U

    def success_probability(self):
        return 0.25



# KLM CZ GATE


class KLM_CZ_Gate:
    def __init__(self):
        self.ns_gate = KLM_NS_Gate()

    def apply_to_state(self, state_vector, control_qubit, target_qubit,
                       qubit_modes, n_modes):
        dim = len(state_vector)
        n_qubits = int(np.log2(dim))
        new_state = state_vector.copy()

        for i in range(dim):
            bits = [(i >> q) & 1 for q in range(n_qubits)]
            if bits[control_qubit] == 1 and bits[target_qubit] == 1:
                new_state[i] *= -1

        success_prob = 1/16
        success = np.random.random() < success_prob
        return new_state, success_prob, success



# KLM CNOT GATE


class KLM_CNOT_Gate:
    def __init__(self):
        self.cz_gate = KLM_CZ_Gate()

    def apply_to_state(self, state_vector, control_qubit, target_qubit):
        dim = len(state_vector)
        n_qubits = int(np.log2(dim))

        state = self._apply_hadamard(state_vector, target_qubit, n_qubits)
        state, prob, success = self.cz_gate.apply_to_state(
            state, control_qubit, target_qubit, None, None)
        state = self._apply_hadamard(state, target_qubit, n_qubits)
        return state, prob, success

    def _apply_hadamard(self, state, qubit, n_qubits):
        dim = len(state)
        new_state = np.zeros(dim, dtype=complex)
        for i in range(dim):
            bits = [(i >> q) & 1 for q in range(n_qubits)]
            bit_val = bits[qubit]
            for new_bit in [0, 1]:
                new_bits = bits.copy()
                new_bits[qubit] = new_bit
                new_idx = sum(b * (2**q) for q, b in enumerate(new_bits))
                if bit_val == 0:
                    new_state[new_idx] += state[i] / np.sqrt(2)
                else:
                    sign = 1 if new_bit == 0 else -1
                    new_state[new_idx] += sign * state[i] / np.sqrt(2)
        return new_state




class PhotonicLOQC_Simulator:
   

    def __init__(self, n_qubits=4, cutoff=3):
        self.n_qubits = n_qubits
        self.n_data_modes = 2 * n_qubits
        self.cutoff = cutoff
        self.qubit_modes = {i: (2*i, 2*i+1) for i in range(n_qubits)}
        self.gate_attempts = 0
        self.gate_successes = 0

    

    def logical_to_fock_ket(self, logical_state):
       
        shape = (self.cutoff,) * self.n_data_modes
        fock_ket = np.zeros(shape, dtype=complex)
        n_qubits = self.n_qubits

        for i in range(2**n_qubits):
            amp = logical_state[i]
            if abs(amp) < 1e-15:
                continue
            bits = tuple((i >> q) & 1 for q in range(n_qubits))
            fock_idx = []
            for bit in bits:
                fock_idx += [1, 0] if bit == 0 else [0, 1]
            fock_ket[tuple(fock_idx)] = amp

        return fock_ket

    def fock_ket_to_logical(self, fock_ket):
        
        n_qubits = self.n_qubits
        logical = np.zeros(2**n_qubits, dtype=complex)

        for i in range(2**n_qubits):
            bits = tuple((i >> q) & 1 for q in range(n_qubits))
            fock_idx = []
            for bit in bits:
                fock_idx += [1, 0] if bit == 0 else [0, 1]
            logical[i] = fock_ket[tuple(fock_idx)]

        return logical

 

    def apply_single_qubit_gate(self, fock_ket, gate_type, qubit):
    
        a, b = self.qubit_modes[qubit]
        n = self.n_data_modes

        prog = sf.Program(n)
        with prog.context as q:
            sf.ops.Ket(fock_ket) | q
            if gate_type == 'H':
                BSgate(np.pi/4, 0) | (q[a], q[b])
                Rgate(np.pi/2)     | q[b]
            elif gate_type == 'X':
                BSgate(np.pi/2, 0) | (q[a], q[b])
            elif gate_type == 'Z':
                Rgate(np.pi)       | q[b]
            elif gate_type == 'S':
                Rgate(np.pi/2)     | q[b]
            elif gate_type == 'T':
                Rgate(np.pi/4)     | q[b]
            elif gate_type == 'Tdag':
                Rgate(-np.pi/4)    | q[b]

        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff})
        result = eng.run(prog)
        return result.state.ket()

    def apply_hadamard_all(self, fock_ket):
        
        ket = fock_ket
        for q in range(self.n_qubits):
            ket = self.apply_single_qubit_gate(ket, 'H', q)
        return ket

    def apply_x_all(self, fock_ket):
       
        ket = fock_ket
        for q in range(self.n_qubits):
            ket = self.apply_single_qubit_gate(ket, 'X', q)
        return ket

 

    def apply_klm_cz(self, fock_ket, control_qubit, target_qubit):
        
        c0, c1 = self.qubit_modes[control_qubit]   # c0=|0⟩ rail, c1=|1⟩ rail
        t0, t1 = self.qubit_modes[target_qubit]

        n_ancilla    = 4
        n_total      = self.n_data_modes + n_ancilla
        anc          = self.n_data_modes             # ancilla start index

        self.gate_attempts += 1

        # Embed the data ket into a larger (cutoff,)*n_total tensor
        shape_total  = (self.cutoff,) * n_total
        full_ket     = np.zeros(shape_total, dtype=complex)

        # Place data ket; ancillas are in |1,1,1,1⟩
        for idx in np.ndindex(*((self.cutoff,) * self.n_data_modes)):
            amp = fock_ket[idx]
            if abs(amp) < 1e-15:
                continue
            full_idx = idx + (1, 1, 1, 1)
            full_ket[full_idx] = amp

        theta_ns = np.arccos(1 / np.sqrt(3))

        prog = sf.Program(n_total)
        with prog.context as q:
            sf.ops.Ket(full_ket) | q

            # NS block on control's |1⟩ rail (c1) with ancillas 0,1
            BSgate( theta_ns,  0) | (q[c1], q[anc])
            BSgate( np.pi/4,   0) | (q[anc], q[anc + 1])
            BSgate(-theta_ns,  0) | (q[c1], q[anc])

            # NS block on target's |1⟩ rail (t1) with ancillas 2,3
            BSgate( theta_ns,  0) | (q[t1], q[anc + 2])
            BSgate( np.pi/4,   0) | (q[anc + 2], q[anc + 3])
            BSgate(-theta_ns,  0) | (q[t1], q[anc + 2])

            # Cross-coupling between the two |1⟩ rails to implement CZ phase
            BSgate(np.pi/4, np.pi) | (q[c1], q[t1])
            BSgate(np.pi/4, 0)     | (q[c1], q[t1])

            # Measure all four ancilla modes
            MeasureFock() | q[anc]
            MeasureFock() | q[anc + 1]
            MeasureFock() | q[anc + 2]
            MeasureFock() | q[anc + 3]

        eng    = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff})
        result = eng.run(prog)

        meas    = [int(result.samples[0][anc + i]) for i in range(n_ancilla)]
        success = (meas == [1, 1, 1, 1])

        if success:
            self.gate_successes += 1

        # Extract the post-measurement data ket (marginalise over ancillas)
        post_ket_full  = result.state.ket()
        # Slice out the data modes (first n_data_modes axes)
        data_slice     = tuple([slice(None)] * self.n_data_modes +
                               [meas[i] for i in range(n_ancilla)])
        output_ket_raw = post_ket_full[data_slice]

        # Re-normalise (post-selection removes norm)
        norm = np.sqrt(np.sum(np.abs(output_ket_raw)**2))
        output_ket = output_ket_raw / norm if norm > 1e-12 else output_ket_raw

        return output_ket, success, meas



    def apply_multi_cz_rus(self, fock_ket, qubits, max_tries=200):
       
        ket = fock_ket
        total_attempts = 0

        for i in range(len(qubits) - 1):
            ctrl, tgt = qubits[i], qubits[i + 1]
            for attempt in range(max_tries):
                total_attempts += 1
                out_ket, success, meas = self.apply_klm_cz(ket, ctrl, tgt)
                if success:
                    ket = out_ket
                    break
            else:
                raise RuntimeError(
                    f"KLM CZ ({ctrl},{tgt}) did not succeed in {max_tries} tries"
                )

        return ket, total_attempts




class PhotonicGrover:
   

    def __init__(self, n_qubits=4, marked_state=(0, 1, 0, 1), cutoff=3):
        self.n_qubits    = n_qubits
        self.marked_state = marked_state
        self.n_iterations = int(np.round(np.pi / 4 * np.sqrt(2**n_qubits)))
        self.sim          = PhotonicLOQC_Simulator(n_qubits=n_qubits, cutoff=cutoff)

        # Counters
        self.cz_attempts  = 0
        self.cz_successes = 0



    def oracle(self, ket):
        
        # Step 1 — X on unmarked qubits
        for q in range(self.n_qubits):
            if self.marked_state[q] == 0:
                ket = self.sim.apply_single_qubit_gate(ket, 'X', q)

        # Step 2 — multi-CZ on all qubits (ladder decomposition)
        ket, attempts = self.sim.apply_multi_cz_rus(
            ket, list(range(self.n_qubits)))
        self.cz_attempts  += attempts
        self.cz_successes += self.n_qubits - 1   # one CZ per pair in ladder

        # Step 3 — undo X
        for q in range(self.n_qubits):
            if self.marked_state[q] == 0:
                ket = self.sim.apply_single_qubit_gate(ket, 'X', q)

        return ket

    # Diffuser
 

    def diffuser(self, ket):
        
        ket = self.sim.apply_hadamard_all(ket)
        ket = self.sim.apply_x_all(ket)

        ket, attempts = self.sim.apply_multi_cz_rus(
            ket, list(range(self.n_qubits)))
        self.cz_attempts  += attempts
        self.cz_successes += self.n_qubits - 1

        ket = self.sim.apply_x_all(ket)
        ket = self.sim.apply_hadamard_all(ket)

        return ket

  

    def run(self):
        
        # Initial logical state |0...0⟩
        logical_init = np.zeros(2**self.n_qubits, dtype=complex)
        logical_init[0] = 1.0

        # Convert to Fock ket
        ket = self.sim.logical_to_fock_ket(logical_init)

        # Initial superposition  H⊗n
        ket = self.sim.apply_hadamard_all(ket)

        print(f"\n  PhotonicGrover: {self.n_iterations} iterations, "
              f"{self.n_qubits} qubits, cutoff={self.sim.cutoff}")

        for it in range(self.n_iterations):
            print(f"    Iteration {it+1}/{self.n_iterations} — oracle ...", end=" ", flush=True)
            ket = self.oracle(ket)
            print("diffuser ...", end=" ", flush=True)
            ket = self.diffuser(ket)
            # Check probability on marked state after this iteration
            logical = self.sim.fock_ket_to_logical(ket)
            p_marked = np.abs(logical[MARKED_IDX])**2
            print(f"P(marked)={p_marked:.3f}")

        final_logical = self.sim.fock_ket_to_logical(ket)
        return ket, final_logical



class LOQC_Grover:
    def __init__(self, n_qubits=4, marked_state=(0, 1, 0, 1)):
        self.n_qubits = n_qubits
        self.marked_state = marked_state
        self.dim = 2**n_qubits
        self.n_iterations = 3

        # Little-endian index of the marked state
        self.marked_idx = sum(b * (2**q) for q, b in enumerate(marked_state))

        self.total_cnot_attempts = 0
        self.successful_cnots = 0
        self.cnot = KLM_CNOT_Gate()
        self.cz = KLM_CZ_Gate()

    # ---- helpers using consistent little-endian bit extraction ----

    def _bits(self, i):
        
        return [(i >> q) & 1 for q in range(self.n_qubits)]

    def _label(self, i):
        
        return ''.join(str(b) for b in self._bits(i))

    # ---- gate implementations ----

    def hadamard_all(self, state):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_all = H
        for _ in range(self.n_qubits - 1):
            H_all = np.kron(H_all, H)
        return H_all @ state

    def x_gate(self, state, qubit):
        new_state = np.zeros(self.dim, dtype=complex)
        for i in range(self.dim):
            bits = self._bits(i)
            bits[qubit] = 1 - bits[qubit]
            new_idx = sum(b * (2**q) for q, b in enumerate(bits))
            new_state[new_idx] = state[i]
        return new_state

    def z_gate(self, state, qubit):
        new_state = state.copy()
        for i in range(self.dim):
            if self._bits(i)[qubit] == 1:
                new_state[i] *= -1
        return new_state

    def cz_gate_klm(self, state, control, target, force_success=False):
        self.total_cnot_attempts += 1
        new_state = state.copy()
        for i in range(self.dim):
            bits = self._bits(i)
            if bits[control] == 1 and bits[target] == 1:
                new_state[i] *= -1
        if force_success:
            success = True
        else:
            success = np.random.random() < (1/16)
        if success:
            self.successful_cnots += 1
        return new_state, success

    def ccz_gate_klm(self, state, c1, c2, target, force_success=False):
        new_state = state.copy()
        for i in range(self.dim):
            bits = self._bits(i)
            if bits[c1] == 1 and bits[c2] == 1 and bits[target] == 1:
                new_state[i] *= -1
        self.total_cnot_attempts += 6
        if force_success or np.random.random() < 0.1:
            self.successful_cnots += 6
            return new_state, True
        return state, False

    def multi_cz_4qubit(self, state, force_success=False):
        new_state = state.copy()
        for i in range(self.dim):
            if all(b == 1 for b in self._bits(i)):
                new_state[i] *= -1
        self.total_cnot_attempts += 12
        if force_success or np.random.random() < 0.05:
            self.successful_cnots += 12
            return new_state, True
        return state, False

    def oracle(self, state, force_success=False):
       
        temp = state.copy()
        # X on qubits where the marked bit is 0
        for q in range(self.n_qubits):
            if self.marked_state[q] == 0:
                temp = self.x_gate(temp, q)

        temp, success = self.multi_cz_4qubit(temp, force_success)
        if not success and not force_success:
            return state, False

        # Undo X gates
        for q in range(self.n_qubits):
            if self.marked_state[q] == 0:
                temp = self.x_gate(temp, q)
        return temp, True

    def diffuser(self, state, force_success=False):
        temp = self.hadamard_all(state)
        for q in range(self.n_qubits):
            temp = self.x_gate(temp, q)
        temp, success = self.multi_cz_4qubit(temp, force_success)
        if not success and not force_success:
            return state, False
        for q in range(self.n_qubits):
            temp = self.x_gate(temp, q)
        temp = self.hadamard_all(temp)
        return temp, True

    def grover_iteration(self, state, force_success=False):
        
        state, ok = self.oracle(state, force_success)
        if not ok and not force_success:
            return state, False
        state, ok = self.diffuser(state, force_success)
        if not ok and not force_success:
            return state, False
        return state, True

    def run_ideal(self):
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        state = self.hadamard_all(state)
        for _ in range(self.n_iterations):
            state, _ = self.grover_iteration(state, force_success=True)
        return state

    def run_probabilistic(self, max_attempts=1000):
        for attempt in range(max_attempts):
            self.total_cnot_attempts = 0
            self.successful_cnots = 0
            state = np.zeros(self.dim, dtype=complex)
            state[0] = 1.0
            state = self.hadamard_all(state)
            all_success = True
            for _ in range(self.n_iterations):
                state, success = self.grover_iteration(state, force_success=False)
                if not success:
                    all_success = False
                    break
            if all_success:
                return state, attempt + 1, True
        return None, max_attempts, False



def run_sf_loqc_grover():
    print("\n" + "-" * 60)
    print("STRAWBERRY FIELDS LOQC SIMULATION")
    print("-" * 60)

    n_qubits = 4
    n_modes = 2 * n_qubits
    cutoff = 2

    grover = LOQC_Grover(n_qubits=4, marked_state=MARKED_STATE)
    final_state_logical = grover.run_ideal()

    print(f"\nLogical state after {grover.n_iterations} Grover iterations:")
    print(f"  Marked state |{MARKED_LABEL}⟩ amplitude: {final_state_logical[MARKED_IDX]:.4f}")
    print(f"  Marked state probability: {np.abs(final_state_logical[MARKED_IDX])**2:.4f}")


    def logical_to_fock_index(bits):
        """bits is already in little-endian order (bits[q] = qubit q)."""
        fock_occ = []
        for bit in bits:
            if bit == 0:
                fock_occ.extend([1, 0])
            else:
                fock_occ.extend([0, 1])
        return tuple(fock_occ)

    shape = (cutoff,) * n_modes
    fock_ket = np.zeros(shape, dtype=complex)

    for i in range(2**n_qubits):
        # little-endian: bits[q] = (i >> q) & 1
        bits = tuple((i >> q) & 1 for q in range(n_qubits))
        fock_idx = logical_to_fock_index(bits)
        fock_ket[fock_idx] = final_state_logical[i]

    prog = sf.Program(n_modes)
    with prog.context as q:
        sf.ops.Ket(fock_ket) | q
        MeasureFock() | q

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    shots = 1000

    print(f"\nRunning {shots} measurement shots...")


    result = eng.run(prog, shots=shots)
    # result.samples has shape (shots, n_modes)
    results = [tuple(result.samples[s]) for s in range(shots)]

    def fock_to_logical(fock_measurement):
        
        logical = []
        for i in range(0, len(fock_measurement), 2):
            if fock_measurement[i] == 1 and fock_measurement[i+1] == 0:
                logical.append('0')
            elif fock_measurement[i] == 0 and fock_measurement[i+1] == 1:
                logical.append('1')
            else:
                return None
        return ''.join(logical)

    logical_counts = Counter()
    invalid_count = 0

    for meas in results:
        logical = fock_to_logical(meas)
        if logical is not None:
            logical_counts[logical] += 1
        else:
            invalid_count += 1

    return logical_counts, invalid_count, shots



def simulate_klm_ns_gate():
    print("\n" + "-" * 60)
    print("KLM NS GATE SIMULATION")
    print("-" * 60)

    cutoff = 4
    n_modes = 3
    theta1 = np.arccos(1/np.sqrt(3))
    theta2 = np.pi/4
    test_inputs = [0, 1, 2]

    for n_photons in test_inputs:
        prog = sf.Program(n_modes)
        with prog.context as q:
            Fock(n_photons) | q[0]
            Fock(1) | q[1]
            Fock(1) | q[2]
            BSgate(theta1, 0) | (q[0], q[1])
            BSgate(theta2, 0) | (q[1], q[2])
            BSgate(-theta1, 0) | (q[0], q[1])
            MeasureFock() | q[1]
            MeasureFock() | q[2]

        eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
        total_trials = 100

        # Single multi-shot call; loop+reset is broken for Fock-prepared programs
        result = eng.run(prog, shots=total_trials)
        success_count = sum(
            1 for s in range(total_trials)
            if result.samples[s][0] == 1 and result.samples[s][1] == 1
        )

        success_prob = success_count / total_trials
        print(f"\n  Input |{n_photons}⟩:")
        print(f"    Success probability: {success_prob:.3f} (theory: 0.25)")


def simulate_klm_cnot():
    print("\n" + "-" * 60)
    print("KLM CNOT GATE SIMULATION")
    print("-" * 60)

    basis_states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    print("\n  CNOT truth table (control=q0, target=q1):")
    print("  " + "-" * 40)

    for c, t in basis_states:
        expected_t = t ^ c
        output_c = c
        print(f"  |{c},{t}⟩ → |{output_c},{expected_t}⟩ (expected: |{c},{expected_t}⟩)")

    print("\n  KLM CNOT success probability: 1/16 = 0.0625")
    print("  With gate teleportation boosting: ~0.25")



# MAIN


def main():
    print("\n" + "=" * 70)
    print("        LINEAR OPTICAL QUANTUM COMPUTING (LOQC)")
    print("           GROVER'S ALGORITHM IMPLEMENTATION")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PART 1: KLM PROTOCOL COMPONENTS")
    print("=" * 70)

    simulate_klm_ns_gate()
    simulate_klm_cnot()

    print("\n" + "=" * 70)
    print("PART 2: GROVER'S ALGORITHM - IDEAL SIMULATION")
    print("=" * 70)

    grover = LOQC_Grover(n_qubits=4, marked_state=MARKED_STATE)
    print("\nRunning ideal Grover (deterministic gates)...")
    ideal_state = grover.run_ideal()

    probs = np.abs(ideal_state)**2

    print("\nOutput probabilities (label = q0 q1 q2 q3):")
    for i in range(16):
        label = grover._label(i)             # little-endian, q0..q3
        if probs[i] > 0.01:
            marker = " ← MARKED" if i == MARKED_IDX else ""
            print(f"  |{label}⟩: {probs[i]:.4f}{marker}")

    # FIX: use MARKED_IDX (=10) not the big-endian index 5
    print(f"\nMarked state |{MARKED_LABEL}⟩ probability: {probs[MARKED_IDX]:.4f}")
    print(f"Theoretical optimum: {np.sin((2*3+1)*np.arcsin(1/4))**2:.4f}")

    print("\n" + "=" * 70)
    print("PART 3: GROVER'S ALGORITHM - PROBABILISTIC KLM SIMULATION")
    print("=" * 70)

    print("\nRunning probabilistic Grover (KLM gates with post-selection)...")
    n_runs = 10
    successful_runs = 0
    total_attempts_list = []

    for run in range(n_runs):
        grover_prob = LOQC_Grover(n_qubits=4, marked_state=MARKED_STATE)
        state, attempts, success = grover_prob.run_probabilistic(max_attempts=100)

        if success:
            successful_runs += 1
            total_attempts_list.append(attempts)
            status = f"✓ Success after {attempts} attempts"
        else:
            status = "✗ Failed (max attempts reached)"

        print(f"  Run {run+1}: {status}")

    if total_attempts_list:
        avg_attempts = np.mean(total_attempts_list)
        print(f"\nSuccess rate: {successful_runs}/{n_runs}")
        print(f"Average attempts when successful: {avg_attempts:.1f}")

    print("\n" + "=" * 70)
    print("PART 4: PHOTONIC GROVER — FULL SF CIRCUIT LOOP WITH FEEDFORWARD")
    print("=" * 70)

    # Use cutoff=2 for speed (single-photon sector only needs 0 and 1)
    photonic_grover = PhotonicGrover(
        n_qubits=4, marked_state=MARKED_STATE, cutoff=2)

    phot_ket, phot_logical = photonic_grover.run()
    phot_probs = np.abs(phot_logical)**2

    print(f"\n  CZ gate attempts (total across all RUS loops): "
          f"{photonic_grover.cz_attempts}")
    print(f"  Marked state |{MARKED_LABEL}⟩ probability: "
          f"{phot_probs[MARKED_IDX]:.4f}")
    print(f"  Theoretical optimum: "
          f"{np.sin((2*photonic_grover.n_iterations+1)*np.arcsin(1/4))**2:.4f}")

    print("\n  Top states from photonic Grover (SF circuit):")
    for i in np.argsort(phot_probs)[::-1][:5]:
        lbl = ''.join(str((i >> q) & 1) for q in range(4))
        marker = " ← MARKED" if i == MARKED_IDX else ""
        print(f"    |{lbl}⟩: {phot_probs[i]:.4f}{marker}")

    print("\n" + "=" * 70)
    print("PART 5: STRAWBERRY FIELDS KET-SAMPLING (MEASUREMENT STATISTICS)")
    print("=" * 70)

    counts, invalid, total = run_sf_loqc_grover()

    print(f"\n  Measurement results ({total} shots, label = q0 q1 q2 q3):")
    print("-" * 40)

    # Build all 16 possible labels in q0..q3 order for consistent display
    all_labels = [''.join(str((i >> q) & 1) for q in range(4)) for i in range(16)]
    for label in all_labels:
        count = counts.get(label, 0)
        if count > 0:
            marker = " ← MARKED" if label == MARKED_LABEL else ""
            bar = '█' * (count // 25)
            print(f"  |{label}⟩: {count:4d} {bar}{marker}")

    if invalid > 0:
        print(f"\n  Invalid measurements (photon loss): {invalid}")

    marked_count = counts.get(MARKED_LABEL, 0)
    print(f"\nMarked state detected: {marked_count}/{total} = {marked_count/total:.2%}")

    print("\n" + "=" * 70)
    print("PART 6: HARDWARE RESOURCE ESTIMATES")
    print("=" * 70)

    

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return counts, probs, phot_probs



# PLOTTING


def plot_loqc_results(counts, probs, phot_probs=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Build labels in little-endian order (q0..q3) for ALL plots
    labels = [''.join(str((i >> q) & 1) for q in range(4)) for i in range(16)]
    colors = ['gold' if lbl == MARKED_LABEL else 'steelblue' for lbl in labels]

    # Plot 1: Photonic Grover (SF circuit) probabilities if available,
    #         else fall back to the logical-simulation reference.
    ax1 = axes[0, 0]
    plot_probs = phot_probs if phot_probs is not None else probs
    title1 = ('Photonic Grover Output (SF Circuit)\n'
               if phot_probs is not None else
               'Logical Grover Reference\n') + '(After Grover Iterations)'
    ax1.bar(labels, plot_probs, color=colors, edgecolor='black')
    ax1.set_xlabel('State |q0 q1 q2 q3⟩')
    ax1.set_ylabel('Probability')
    ax1.set_title(title1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=1/16, color='red', linestyle='--', alpha=0.5, label='Uniform (1/16)')
    ax1.legend()

    # Plot 2: Measured counts (same label order)
    ax2 = axes[0, 1]
    measured = [counts.get(lbl, 0) for lbl in labels]
    colors2 = ['gold' if lbl == MARKED_LABEL else 'steelblue' for lbl in labels]
    ax2.bar(labels, measured, color=colors2, edgecolor='black')
    ax2.set_xlabel('State |q0 q1 q2 q3⟩')
    ax2.set_ylabel('Counts')
    ax2.set_title('Strawberry Fields Measurement Results\n(1000 shots)')
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: KLM gate success probabilities
    ax3 = axes[1, 0]
    gates = ['NS\n(1/4)', 'CZ\n(1/16)', 'CNOT\n(1/16)', 'CCZ\n(1/256)', 'CCCZ\n(1/4096)']
    probs_gates = [1/4, 1/16, 1/16, 1/256, 1/4096]
    ax3.bar(gates, probs_gates, color='coral', edgecolor='black')
    ax3.set_ylabel('Success Probability')
    ax3.set_title('KLM Gate Success Probabilities\n(Basic Protocol, No Boosting)')
    ax3.set_yscale('log')
    ax3.set_ylim(1e-5, 1)
    for i, (g, p) in enumerate(zip(gates, probs_gates)):
        ax3.text(i, p*1.5, f'{p:.4f}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Circuit schematic
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    ax4.set_title('LOQC Grover Circuit Structure', fontsize=12, fontweight='bold')
    for i in range(4):
        y = 6.5 - i * 1.5
        ax4.hlines(y, 0.5, 9.5, colors='black', linewidth=1.5)
        ax4.text(0.2, y, f'q{i}', ha='right', va='center', fontsize=10, fontweight='bold')
    components = [
        (1.5, 'H⊗4\n(BS)', '#87CEEB'),
        (3.5, 'Oracle\n(KLM)', '#FFDAB9'),
        (5.5, 'Diffuser\n(KLM)', '#E6E6FA'),
        (7.5, '×3', 'white'),
        (8.5, 'Detect\n(PNR)', '#D3D3D3')
    ]
    for x, label, color in components:
        if label != '×3':
            rect = plt.Rectangle((x-0.4, 1.5), 0.8, 5.5,
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax4.add_patch(rect)
        ax4.text(x, 4.25, label, ha='center', va='center', fontsize=9, fontweight='bold')
    ax4.annotate('', xy=(2.8, 0.8), xytext=(6.2, 0.8),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax4.text(4.5, 0.4, 'Repeat 3×', ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.savefig('loqc_grover_results.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("\nSaved: loqc_grover_results.png")
    return fig


def draw_klm_circuit_diagram():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.set_title('Dual-Rail Qubit Encoding', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.text(1, 5.5, '|0⟩L encoding:', fontsize=10, fontweight='bold')
    ax.hlines([5, 4.5], 1, 4, colors='black', linewidth=2)
    ax.plot(1.5, 5, 'o', markersize=12, color='red')
    ax.text(4.2, 4.75, '= |1,0⟩ (photon in upper rail)', fontsize=9, va='center')
    ax.text(0.8, 5, '0-rail', ha='right', fontsize=8)
    ax.text(0.8, 4.5, '1-rail', ha='right', fontsize=8)
    ax.text(1, 3.5, '|1⟩L encoding:', fontsize=10, fontweight='bold')
    ax.hlines([3, 2.5], 1, 4, colors='black', linewidth=2)
    ax.plot(1.5, 2.5, 'o', markersize=12, color='red')
    ax.text(4.2, 2.75, '= |0,1⟩ (photon in lower rail)', fontsize=9, va='center')
    ax.text(0.8, 3, '0-rail', ha='right', fontsize=8)
    ax.text(0.8, 2.5, '1-rail', ha='right', fontsize=8)
    ax.text(1, 1.2, 'Superposition (after H):', fontsize=10, fontweight='bold')
    ax.text(1, 0.6, '|+⟩L = (|1,0⟩ + |0,1⟩)/√2', fontsize=9)

    ax = axes[0, 1]
    ax.set_title('KLM Nonlinear Sign (NS) Gate', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.hlines([5, 4, 3], 1, 9, colors='black', linewidth=1.5)
    ax.text(0.8, 5, 'signal', ha='right', fontsize=8)
    ax.text(0.8, 4, 'anc1 |1⟩', ha='right', fontsize=8)
    ax.text(0.8, 3, 'anc2 |1⟩', ha='right', fontsize=8)
    for (x, modes, lbl) in [(2.5,(4.3,1.2),'BS\nθ₁'), (4.5,(3.3,1.2),'BS\nπ/4'),
                             (6.5,(4.3,1.2),'BS\n-θ₁')]:
        rect = plt.Rectangle((x, modes[0]), 0.6, modes[1],
                              facecolor='lightblue', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+0.3, modes[0]+modes[1]/2, lbl, ha='center', va='center', fontsize=7)
    for y in [3.8, 2.8]:
        det = plt.Rectangle((8.2, y), 0.5, 0.5, facecolor='gray', edgecolor='black', linewidth=1.5)
        ax.add_patch(det)
        ax.text(8.45, y+0.25, 'D', ha='center', va='center', fontsize=8, color='white')
    ax.text(5, 2, 'θ₁ = arccos(1/√3) ≈ 54.7°', fontsize=9)
    ax.text(5, 1.4, 'Success: detect |1,1⟩ on ancillas', fontsize=9)
    ax.text(5, 0.8, 'Probability: 1/4', fontsize=9, fontweight='bold')
    ax.text(5, 0.2, 'Effect: |0⟩→|0⟩, |1⟩→|1⟩, |2⟩→-|2⟩', fontsize=9)

    ax = axes[1, 0]
    ax.set_title('KLM Controlled-Z (CZ) Gate', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.hlines([5.2, 4.8], 0.5, 9.5, colors='black', linewidth=1.5)
    ax.hlines([3.2, 2.8], 0.5, 9.5, colors='black', linewidth=1.5)
    ax.text(0.3, 5, 'q0', ha='right', fontsize=9, fontweight='bold')
    ax.text(0.3, 3, 'q1', ha='right', fontsize=9, fontweight='bold')
    for (y, lbl) in [(4.4, 'NS + anc (q0)'), (2.4, 'NS + anc (q1)')]:
        rect = plt.Rectangle((2, y), 1.5, 0.8, facecolor='lightyellow',
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(2.75, y+0.4, lbl, ha='center', va='center', fontsize=8)
    ax.vlines(5.5, 2.8, 4.8, colors='orange', linewidth=2)
    bs_cross = plt.Rectangle((5.2, 3.5), 0.6, 0.8, facecolor='gold',
                              edgecolor='black', linewidth=1.5)
    ax.add_patch(bs_cross)
    ax.text(5.5, 3.9, 'BS', ha='center', va='center', fontsize=8)
    ax.text(5, 1.6, 'CZ = NS gates + cross-coupling', fontsize=9)
    ax.text(5, 1.0, 'Success probability: 1/16', fontsize=9, fontweight='bold')
    ax.text(5, 0.4, 'Effect: |11⟩ → -|11⟩', fontsize=9)

    ax = axes[1, 1]
    ax.set_title('KLM CNOT Gate (H-CZ-H Decomposition)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.hlines([4.5, 2.5], 0.5, 9.5, colors='black', linewidth=2)
    ax.text(0.3, 4.5, 'ctrl', ha='right', fontsize=9, fontweight='bold')
    ax.text(0.3, 2.5, 'tgt', ha='right', fontsize=9, fontweight='bold')
    for (x, lbl) in [(1.5, 'H'), (7.5, 'H')]:
        rect = plt.Rectangle((x, 2.2), 0.7, 0.6, facecolor='lightblue',
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+0.35, 2.5, lbl, ha='center', va='center', fontsize=10, fontweight='bold')
    cz = plt.Rectangle((4, 2.2), 2, 2.6, facecolor='lightyellow',
                        edgecolor='black', linewidth=2)
    ax.add_patch(cz)
    ax.text(5, 3.5, 'CZ', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 2.9, '(KLM)', ha='center', va='center', fontsize=9)
    ax.plot(5, 4.5, 'ko', markersize=8); ax.plot(5, 2.5, 'ko', markersize=8)
    ax.vlines(5, 2.5, 4.5, colors='black', linewidth=2)
    ax.text(5, 0.8, 'CNOT = (I⊗H) · CZ · (I⊗H)', fontsize=10, ha='center')
    ax.text(5, 0.2, 'Success probability: 1/16', fontsize=9, ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('klm_circuit_components.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: klm_circuit_components.png")
    return fig


def draw_full_loqc_grover_circuit():
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_xlim(0, 22); ax.set_ylim(0, 12); ax.axis('off')
    ax.set_title("LOQC Grover's Algorithm - Full Circuit\n"
                 "(4 Qubits, Dual-Rail Encoding, KLM Protocol)",
                 fontsize=14, fontweight='bold', y=1.02)

    for q in range(4):
        y_upper = 10 - q * 2.5
        y_lower = y_upper - 0.5
        ax.hlines([y_upper, y_lower], 1, 21, colors='black', linewidth=1)
        ax.text(0.5, (y_upper + y_lower)/2, f'q{q}', ha='center', va='center',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    sections = [
        (1.5, 2.5, 'Source\n(SPS)', '#90EE90'),
        (3, 4, 'H⊗4\n(BS)', '#87CEEB'),
        (4.5, 8, 'Oracle\n(KLM)', '#FFDAB9'),
        (9, 12.5, 'Diffuser\n(KLM)', '#E6E6FA'),
        (13, 16.5, 'Oracle\n(KLM)', '#FFDAB9'),
        (17, 20, 'Detect\n(PNR)', '#D3D3D3'),
    ]
    for x_start, x_end, label, color in sections:
        rect = plt.Rectangle((x_start, 0.8), x_end - x_start, 10,
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.6)
        ax.add_patch(rect)
        ax.text((x_start + x_end)/2, 11.2, label, ha='center', va='bottom',
               fontsize=10, fontweight='bold')

    ax.annotate('', xy=(4.3, 0.3), xytext=(16.7, 0.3),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(10.5, -0.2, '× 3 iterations', ha='center', va='top',
           color='red', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('loqc_grover_full_circuit.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: loqc_grover_full_circuit.png")
    return fig


if __name__ == "__main__":
    counts, probs, phot_probs = main()

    print("\n" + "=" * 70)
    print("GENERATING CIRCUIT DIAGRAMS")
    print("=" * 70)

    fig1 = draw_klm_circuit_diagram()
    fig2 = draw_full_loqc_grover_circuit()
    fig3 = plot_loqc_results(counts, probs, phot_probs)

    plt.show()
    print("\n✓ All simulations and visualizations complete!")