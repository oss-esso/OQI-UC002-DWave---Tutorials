import cirq
import math
from itertools import product
from typing import List, Tuple, Dict

# f(x) = 1.5*x0 + (-0.7)*x1 + 0.5*x0*x1 + 2.0*x2  (we will expand to any monomials)
example_coeffs = {
    (): 0.0,             # constant (optional)
    (0,): 1.5,
    (1,): -0.7,
    (0,1): 0.5,
    (2,): 2.0,
    # Add more monomials up to n-1...
}


def scale_coeffs(coeffs:Dict[Tuple[int, ...], float], p: int) -> Dict[Tuple[int, ...], int]:
    scaled={k: int(round(v * (2 ** p))) for k,v in coeffs.items()}
    return scaled


def partition_indices(n: int, block_size: int) -> List[List[int]]:

    blocks = []
    for i in range(0, n, block_size):
        blocks.append(list(range(i, min(i + block_size, n))))
    return blocks


def build_block_lookup(scaled_coeffs:Dict[Tuple[int, ...], int], block_indices:List[int]) -> Dict[str, int]:
    k = len(block_indices)
    table = {}
    for bits in product('01', repeat=k):
        bitstr = ''.join(bits)

        assignment = {block_indices[i]: int(bitstr[i]) for i in range(k)}
        val = 0

        for mon, c  in scaled_coeffs.items():
            term = c
            for idx in mon:
                term *= assignment.get(idx, 0)
            val += term

        table[bitstr] = val  # Store as string key, not int
    return table


def int_to_bits(x:int, bits:int) -> List[int]:
    return [(x >> i) & 1 for i in range(bits)]


def controlled_write_value(block_qubits: List[cirq.Qid],
                           value_qubits: List[cirq.Qid],
                           pattern: str,
                           value: int) -> cirq.Circuit:
    
    ops = []

    for qb, bit in zip(block_qubits, pattern):
        if bit == '0':
            ops.append(cirq.X(qb))  # Fixed typo: appen -> append

    val_bits = int_to_bits(value, len(value_qubits))
    control_qubits = list(block_qubits)
    for i, vb in enumerate(value_qubits):
        if val_bits[i] == 1:
            ops.append(cirq.X(vb).controlled_by(*control_qubits))

    for qb, bit in zip(block_qubits, pattern):
        if bit=='0':
            ops.append(cirq.X(qb))

    return cirq.Circuit(ops)


def cuccaro_adder(a_bits: List[cirq.Qid], b_bits: List[cirq.Qid], carry_qubit: cirq.Qid) -> cirq.Circuit:
    """
    Return circuit performing |a>|b>|0_c> -> |a>|a+b>|g_c>,
    where carry_qubit will hold final carry (optional).
    Assumes a_bits and b_bits are same length; LSB at index 0.
    This is the generate-propagate-kill (Cuccaro) pattern.
    """
    n = len(a_bits)
    ops = []
    # Majority (MAJ) chain
    # For i=0..n-1:
    # MAJ: (a_i, b_i, c) => updates c and b to propagate carry
    # implemented as: CNOT(b_i, a_i); CNOT(c, b_i); Toffoli(a_i, c -> b_i); etc
    # We'll follow standard Cuccaro sequence (explicit construction):
    # NOTE: cirq has CCX as cirq.TOFFOLI; use it via cirq.TOFFOLI(a,b,c)
    for i in range(n):
        ai = a_bits[i]
        bi = b_bits[i]
        ops.append(cirq.CNOT(ai, bi))
        ops.append(cirq.CNOT(carry_qubit if i == 0 else b_bits[i-1], ai))
        ops.append(cirq.TOFFOLI(ai, bi, carry_qubit if i == 0 else b_bits[i-1]))
    # Now do the inverse (Unmajority) to compute sum bits into b
    for i in reversed(range(n)):
        ai = a_bits[i]
        bi = b_bits[i]
        ops.append(cirq.TOFFOLI(ai, bi, carry_qubit if i == 0 else b_bits[i-1]))
        ops.append(cirq.CNOT(carry_qubit if i == 0 else b_bits[i-1], ai))
        ops.append(cirq.CNOT(ai, bi))
    # The above is a conceptual sketch; in practice the Cuccaro uses specific order and ancilla for carries
    # For small examples this naive chain will be acceptable for demonstration, but you may replace with optimized Cuccaro code.
    return cirq.Circuit(ops)


def block_load_and_add(circuit: cirq.Circuit,
                       block_qubits: List[cirq.Qid],
                       value_qubits: List[cirq.Qid],
                       acc_qubits: List[cirq.Qid],
                       carry_qubit: cirq.Qid,
                       lookup_table: Dict[str,int]):
    
    for pat, val in lookup_table.items():
        cw = controlled_write_value(block_qubits, value_qubits, pat, val)
        circuit.append(cw)

        circuit.append(cuccaro_adder(value_qubits, acc_qubits, carry_qubit))

        # Use cirq.inverse() to get the inverse of the circuit
        circuit.append(cirq.inverse(cw))



def add_constant_to_register(circuit: cirq.Circuit, const_int: int, target_qubits: List[cirq.Qid], carry_qubit: cirq.Qid):
    """Add classical constant const_int into target register (LSB first) using controlled Xs and adder structure.
    For demonstration we implement by flipping a temporary register as if it holds const; production code should implement adder with constant addition optimized."""
    # Quick-and-dirty: load const into an ancilla const register, then cuccaro_add(const_reg, target_reg), then clear const_reg.
    # For simplicity we assume const fits into len(target_qubits) bits.
    n = len(target_qubits)
    const_bits = int_to_bits(const_int, n)
    const_qubits = [cirq.NamedQubit(f'const_{i}') for i in range(n)]
    # Note: these const_qubits must be present in the full device allocation; here we simply return ops to be appended by caller.
    # For clarity we will instead implement adding by flipping target qubits controlled by classical bits (NOT reversible).
    for i, bit in enumerate(const_bits):
        if bit == 1:
            circuit.append(cirq.X(target_qubits[i]))

    # Now we can add the constant using the cuccaro adder
    circuit.append(cuccaro_adder(const_qubits, target_qubits, carry_qubit))

    # Finally, we need to uncompute the constant register
    for i, bit in enumerate(const_bits):
        if bit == 1:
            circuit.append(cirq.X(target_qubits[i]))

        


def build_demo_oracle_circuit(n, block_size, scaled_coeffs, p, threshold_int, debug=False):
    """
    Build a Cirq circuit that:
    - has n variable qubits (we'll not initialize them to superposition here; that's done by Grover wrapper)
    - for demonstration: computes accumulator via block dictionaries, then measures accumulator and
      returns measurement result (so classical controller can decide marking).
    This is NOT a fully coherent oracle (we measure acc); it's a practical scaffold you can replace with reversible comparator.
    """
    # Setup blocks
    blocks = partition_indices(n, block_size)
    # bit widths
    # choose bits for per-block value registers: enough to represent max block contribution
    block_tables = []
    max_block_val = 0
    for b in blocks:
        table = build_block_lookup(scaled_coeffs, b)
        block_tables.append(table)
        local_max = max(abs(v) for v in table.values())
        if local_max > max_block_val:
            max_block_val = local_max
    # B bits for accumulator (unsigned)
    B = max(1, math.ceil(math.log2(max_block_val * len(blocks) + 1)))
    # prepare qubits
    var_qubits = [cirq.NamedQubit(f'x{i}') for i in range(n)]
    acc_qubits = [cirq.NamedQubit(f'acc{i}') for i in range(B)]
    # per-block value registers
    block_value_regs = []
    for bi in range(len(blocks)):
        block_value_regs.append([cirq.NamedQubit(f'v{bi}_{i}') for i in range(B)])  # B bits per block for simplicity
    carry_qubit = cirq.NamedQubit('carry')
    # Build circuit
    circuit = cirq.Circuit()
    # (1) The caller should put var_qubits into superposition; here we leave it to caller.
    # (2) For each block: controlled write its block value (in B bits) and add into acc
    for bi, b in enumerate(blocks):
        block_qs = [var_qubits[i - blocks[0][0]] for i in b]  # careful: simpler to index directly
        # We need correct mapping from b indices to var_qubits; do direct mapping:
        block_qs = [var_qubits[i] for i in b]
        value_qs = block_value_regs[bi]
        table = block_tables[bi]
        # load & add
        block_load_and_add(circuit, block_qs, value_qs, acc_qubits, carry_qubit, table)
    # (3) Measure accumulator (demo)
    circuit.append(cirq.measure(*acc_qubits, key='acc'))
    # Return circuit and qubit lists so caller can do Grover wrapper
    meta = {
        'var_qubits': var_qubits,
        'acc_qubits': acc_qubits,
        'block_value_regs': block_value_regs,
        'carry_qubit': carry_qubit,
        'blocks': blocks,
        'B': B
    }
    return circuit, meta




def diffuser_on_qubits(qubits: List[cirq.Qid]) -> cirq.Circuit:
    """Standard diffuser (inversion about mean) on the variable qubits."""
    ops = []
    ops += [cirq.H(q) for q in qubits]
    ops += [cirq.X(q) for q in qubits]
    # multi-controlled Z implemented by H on last, multi-controlled X, H, etc.
    ops.append(cirq.H(qubits[-1]))
    ops.append(cirq.TOFFOLI(*qubits[:-1], qubits[-1]).controlled_by())  # placeholder, cirq doesn't allow controlled_by on gate list
    # For demonstration we'll use a simple implementation: apply Z on all-zeros via projection (not optimized)
    ops.append(cirq.H(qubits[-1]))
    ops += [cirq.X(q) for q in qubits]
    ops += [cirq.H(q) for q in qubits]
    return cirq.Circuit(ops)

def run_demo_gas(n=6, block_size=3, p=4, shots=200):
    # prepare coeffs & scale
    scaled = scale_coeffs(example_coeffs, p)
    # initial threshold: classical sample (all zeros)
    initial_T = 2**(p) * 100  # large
    threshold = initial_T
    # build oracle demo circuit
    oracle_circ, meta = build_demo_oracle_circuit(n, block_size, scaled, p, threshold)
    # Now build a driver circuit: prepare var qubits in uniform superposition, call oracle_circ as subcircuit
    var_qs = meta['var_qubits']
    sim = cirq.Simulator()
    # build a circuit that sets var qubits to uniform superposition, runs oracle, and measures vars when accumulator <= threshold
    qc = cirq.Circuit()
    qc.append([cirq.H(q) for q in var_qs])
    # append oracle demo (which currently measures the accumulator and returns acc value)
    qc += oracle_circ
    # simulate
    res = sim.run(qc, repetitions=shots)
    acc_bits = res.measurements['acc']  # array of shape (shots, B)
    # compute integer values
    acc_values = [sum((bit<<i) for i, bit in enumerate(row)) for row in acc_bits]
    # Now classically pick those shots with acc <= threshold and return measured variable assignment for them
    # (we would also get var measurement if we measured var qubits earlier)
    # For demo we simply print distribution of accumulator values
    from collections import Counter
    counts = Counter(acc_values)
    print("Accumulator distribution (value: counts):")
    for val, c in counts.items():
        print(val, c)
    return counts



if __name__ == "__main__":
    run_demo_gas()

