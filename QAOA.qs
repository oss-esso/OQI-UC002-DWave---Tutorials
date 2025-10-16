// QAOA.qs - Quantum Approximate Optimization Algorithm Implementation
// 
// This Q# program implements QAOA for solving MaxCut and other optimization problems.
// It demonstrates gate-based quantum computing as an alternative to quantum annealing.

namespace QAOAOptimization {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Arrays;

    /// # Summary
    /// Applies the cost Hamiltonian for MaxCut problem
    /// 
    /// # Description
    /// For each edge (i,j), applies: exp(-i * gamma * Z_i Z_j)
    /// This encodes the optimization objective
    ///
    /// # Input
    /// ## gamma
    /// QAOA parameter for cost Hamiltonian layer
    /// ## edges
    /// Array of edges, each edge is a tuple (vertex_i, vertex_j)
    /// ## qubits
    /// Qubit register
    operation ApplyCostHamiltonian(
        gamma : Double, 
        edges : (Int, Int)[], 
        qubits : Qubit[]
    ) : Unit {
        for edge in edges {
            let (i, j) = edge;
            // Implements exp(-i * gamma * Z_i Z_j) using:
            // CNOT - Rz - CNOT sequence
            CNOT(qubits[i], qubits[j]);
            Rz(2.0 * gamma, qubits[j]);
            CNOT(qubits[i], qubits[j]);
        }
    }

    /// # Summary
    /// Applies the mixer Hamiltonian
    /// 
    /// # Description
    /// For each qubit, applies: exp(-i * beta * X_i)
    /// This enables exploration of the solution space
    ///
    /// # Input
    /// ## beta
    /// QAOA parameter for mixer Hamiltonian layer
    /// ## qubits
    /// Qubit register
    operation ApplyMixerHamiltonian(
        beta : Double, 
        qubits : Qubit[]
    ) : Unit {
        for qubit in qubits {
            // X rotation implements exp(-i * beta * X)
            Rx(2.0 * beta, qubit);
        }
    }

    /// # Summary
    /// Complete QAOA circuit for p layers
    /// 
    /// # Description
    /// Builds and executes the full QAOA circuit:
    /// 1. Initialize in equal superposition
    /// 2. Apply p layers of (Cost + Mixer) Hamiltonians
    /// 3. Measure in computational basis
    ///
    /// # Input
    /// ## gammas
    /// Array of gamma parameters (length p)
    /// ## betas
    /// Array of beta parameters (length p)
    /// ## edges
    /// Problem graph edges
    /// ## nQubits
    /// Number of qubits (vertices)
    ///
    /// # Output
    /// Measurement results (bitstring)
    operation QAOACircuit(
        gammas : Double[], 
        betas : Double[], 
        edges : (Int, Int)[], 
        nQubits : Int
    ) : Result[] {
        // Allocate qubits
        use qubits = Qubit[nQubits];
        
        // Step 1: Initialize in equal superposition |+>^n
        ApplyToEach(H, qubits);
        
        // Step 2: Apply p QAOA layers
        let p = Length(gammas);
        for layer in 0..p-1 {
            // Cost Hamiltonian U_C(gamma)
            ApplyCostHamiltonian(gammas[layer], edges, qubits);
            
            // Mixer Hamiltonian U_M(beta)
            ApplyMixerHamiltonian(betas[layer], qubits);
        }
        
        // Step 3: Measure all qubits
        return ForEach(M, qubits);
    }

    /// # Summary
    /// QAOA circuit with manual qubit reset (for repeated sampling)
    ///
    /// # Description
    /// Same as QAOACircuit but with explicit measurement and reset
    /// Useful for running multiple shots in a single quantum execution
    ///
    /// # Input
    /// ## gammas
    /// Array of gamma parameters
    /// ## betas
    /// Array of beta parameters
    /// ## edges
    /// Problem graph edges
    /// ## nQubits
    /// Number of qubits
    ///
    /// # Output
    /// Array of measurement results (as integers)
    operation QAOACircuitWithReset(
        gammas : Double[], 
        betas : Double[], 
        edges : (Int, Int)[], 
        nQubits : Int
    ) : Int[] {
        use qubits = Qubit[nQubits];
        
        // Initialize and apply QAOA
        ApplyToEach(H, qubits);
        let p = Length(gammas);
        for layer in 0..p-1 {
            ApplyCostHamiltonian(gammas[layer], edges, qubits);
            ApplyMixerHamiltonian(betas[layer], qubits);
        }
        
        // Measure and convert to integers
        let results = ForEach(M, qubits);
        mutable bitstring = [];
        for result in results {
            set bitstring += [result == One ? 1 | 0];
        }
        
        // Reset qubits for next iteration
        ResetAll(qubits);
        
        return bitstring;
    }

    /// # Summary
    /// Example: QAOA for triangle graph (3 vertices, 3 edges)
    ///
    /// # Description
    /// Solves MaxCut on a triangle graph with given parameters
    /// This is the example from Tutorial 7
    ///
    /// # Input
    /// ## gamma
    /// Cost Hamiltonian parameter
    /// ## beta
    /// Mixer Hamiltonian parameter
    ///
    /// # Output
    /// Measurement result (bitstring for partition)
    operation QAOATriangleExample(gamma : Double, beta : Double) : Result[] {
        // Triangle graph: edges (0,1), (1,2), (0,2)
        let edges = [(0, 1), (1, 2), (0, 2)];
        let nQubits = 3;
        
        // Single layer QAOA (p=1)
        let gammas = [gamma];
        let betas = [beta];
        
        return QAOACircuit(gammas, betas, edges, nQubits);
    }

    /// # Summary
    /// Resource estimation for QAOA circuit
    ///
    /// # Description
    /// Estimates the quantum resources needed:
    /// - Number of qubits
    /// - Gate count
    /// - Circuit depth
    ///
    /// # Input
    /// ## p
    /// Number of QAOA layers
    /// ## nQubits
    /// Number of vertices (qubits)
    /// ## nEdges
    /// Number of edges in graph
    ///
    /// # Output
    /// Tuple of (total gates, circuit depth)
    function EstimateQAOAResources(p : Int, nQubits : Int, nEdges : Int) : (Int, Int) {
        // Initial Hadamards
        let initialGates = nQubits;
        
        // Per layer:
        // - Cost: 3 gates per edge (CNOT, Rz, CNOT)
        // - Mixer: 1 gate per qubit (Rx)
        let gatesPerLayer = (3 * nEdges) + nQubits;
        
        // Measurement (not counted as gates)
        let totalGates = initialGates + (p * gatesPerLayer);
        
        // Circuit depth (assuming parallel execution where possible)
        // - Initial Hadamards: depth 1
        // - Cost layer depth: depends on graph structure (worst case: nEdges)
        // - Mixer layer depth: 1 (all parallel)
        let depthPerLayer = nEdges + 1;
        let circuitDepth = 1 + (p * depthPerLayer);
        
        return (totalGates, circuitDepth);
    }

    /// # Summary
    /// Entry point for testing QAOA implementation
    ///
    /// # Description
    /// Demonstrates QAOA on triangle graph with sample parameters
    @EntryPoint()
    operation TestQAOA() : Unit {
        Message("QAOA Implementation Test");
        Message("========================");
        Message("");
        
        Message("Problem: MaxCut on Triangle Graph");
        Message("  Vertices: 3 (A=0, B=1, C=2)");
        Message("  Edges: (0,1), (1,2), (0,2)");
        Message("");
        
        // Test with sample parameters
        let gamma = 0.5;
        let beta = 1.0;
        
        Message($"QAOA Parameters: gamma={gamma}, beta={beta}");
        Message("Running QAOA circuit (p=1)...");
        Message("");
        
        // Run multiple times to see distribution
        mutable results = [];
        for trial in 1..10 {
            let result = QAOATriangleExample(gamma, beta);
            Message($"  Trial {trial}: {ResultArrayAsString(result)}");
            set results += [result];
        }
        
        Message("");
        Message("Resource Estimation:");
        let (gates, depth) = EstimateQAOAResources(1, 3, 3);
        Message($"  Total gates: {gates}");
        Message($"  Circuit depth: {depth}");
        Message($"  Qubits: 3");
        
        Message("");
        Message("Note: For optimization, parameters (gamma, beta) should be");
        Message("      tuned using classical optimizer to minimize expected cost.");
    }

    /// # Summary
    /// Helper function to convert Result array to string
    function ResultArrayAsString(results : Result[]) : String {
        mutable str = "";
        for result in results {
            set str += result == One ? "1" | "0";
        }
        return str;
    }
}
