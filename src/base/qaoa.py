import time
import os
import numpy as np
import cudaq
from cudaq import spin
from typing import List
import json
import argparse
import csv

parser = argparse.ArgumentParser("bipartite_qaoa")
parser.add_argument("num_qubits", help="Number of vertices in the bipartite graph.", type=int)
parser.add_argument("num_layers", help="Number of layers to be used in QAOA.", type=int)
parser.add_argument("target", help="Target device to be used - \n1. \"qpp-cpu\" for CPU \n2. \"nvidia\" for GPU", type=str)
parser.add_argument("optimizer", help="Optimizer to be used - \n1. \"COBYLA\" \n2. \"NelderMead\"", type=str)
args = parser.parse_args()

fileprefix = "out/maxIter100"
try:
    os.makedirs(fileprefix)
except FileExistsError:
    # directory already exists
    pass

#print("number of qubits in bipartite graph")

startTime = time.time()


num_qubits = args.num_qubits
layer_count = args.num_layers
target = args.target
opt = args.optimizer
# First we define the graph nodes (i.e., vertices) and edges as lists of integers so that they can be broadcast into
# a cudaq.kernel.
nodes: List[int] = []
for i in range(0, num_qubits):
    nodes.append(i)

edges = []
for i in range(0, int(num_qubits/2)):
    for j in range(int(num_qubits/2), num_qubits):
        edges.append([i, j])

#print(nodes)
#print(edges)
edges_src: List[int] = [edges[i][0] for i in range(len(edges))]
edges_tgt: List[int] = [edges[i][1] for i in range(len(edges))]

# Problem parameters
# The number of qubits we'll need is the same as the number of vertices in our graph
qubit_count: int = len(nodes)

data = {
    "alert": "started",
}


with open(f"{fileprefix}/status/numQubits_{num_qubits}_numLayers_{layer_count}_target_{target}_opt_{opt}.json", "w") as f:
    json.dump(data, f, indent=4)


# Each layer of the QAOA kernel contains 2 parameters
parameter_count: int = 2 * layer_count


@cudaq.kernel
def qaoaProblem(qubit_0: cudaq.qubit, qubit_1: cudaq.qubit, alpha: float):
    """Build the QAOA gate sequence between two qubits that represent an edge of the graph
    Parameters
    ----------
    qubit_0: cudaq.qubit
        Qubit representing the first vertex of an edge
    qubit_1: cudaq.qubit
        Qubit representing the second vertex of an edge
    thetas: List[float]
        Free variable

    Returns
    -------
    cudaq.Kernel
        Subcircuit of the problem kernel for Max-Cut of the graph with a given edge
    """
    x.ctrl(qubit_0, qubit_1)
    rz(2.0 * alpha, qubit_1)
    x.ctrl(qubit_0, qubit_1)


# We now define the kernel_qaoa function which will be the QAOA circuit for our graph
# Since the QAOA circuit for max cut depends on the structure of the graph,
# we'll feed in global concrete variable values into the kernel_qaoa function for the qubit_count, layer_count, edges_src, edges_tgt.
# The types for these variables are restricted to Quake Values (e.g. qubit, int, List[int], ...)
# The thetas plaeholder will be our free parameters
@cudaq.kernel
def kernel_qaoa(qubit_count: int, layer_count: int, edges_src: List[int],
                edges_tgt: List[int], thetas: List[float]):
    """Build the QAOA circuit for max cut of the graph with given edges and nodes
    Parameters
    ----------
    qubit_count: int
        Number of qubits in the circuit, which is the same as the number of nodes in our graph
    layer_count : int
        Number of layers in the QAOA kernel
    edges_src: List[int]
        List of the first (source) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes
    edges_tgt: List[int]
        List of the second (target) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes
    thetas: List[float]
        Free variables to be optimized

    Returns
    -------
    cudaq.Kernel
        QAOA circuit for Max-Cut for max cut of the graph with given edges and nodes
    """
    # Let's allocate the qubits
    qreg = cudaq.qvector(qubit_count)
    # And then place the qubits in superposition
    h(qreg)

    # Each layer has two components: the problem kernel and the mixer
    for i in range(layer_count):
        # Add the problem kernel to each layer
        for edge in range(len(edges_src)):
            qubitu = edges_src[edge]
            qubitv = edges_tgt[edge]
            qaoaProblem(qreg[qubitu], qreg[qubitv], thetas[i])
        # Add the mixer kernel to each layer
        for j in range(qubit_count):
            rx(2.0 * thetas[i + layer_count], qreg[j])

# The problem Hamiltonian
# Define a function to generate the Hamiltonian for a max cut problem using the graph
# with the given edges


def hamiltonian_max_cut(edges_src, edges_tgt):
    """Hamiltonian for finding the max cut for the graph with given edges and nodes

    Parameters
    ----------
    edges_src: List[int]
        List of the first (source) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes
    edges_tgt: List[int]
        List of the second (target) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes

    Returns
    -------
    cudaq.SpinOperator
        Hamiltonian for finding the max cut of the graph with given edges
    """

    hamiltonian = 0

    for edge in range(len(edges_src)):

        qubitu = edges_src[edge]
        qubitv = edges_tgt[edge]
        # Add a term to the Hamiltonian for the edge (u,v)
        hamiltonian += 0.5 * (spin.z(qubitu) * spin.z(qubitv) -
                              spin.i(qubitu) * spin.i(qubitv))

    return hamiltonian



cudaq.set_target('nvidia')
#cudaq.set_target('qpp-cpu')

# Generate the Hamiltonian for our graph
hamiltonian = hamiltonian_max_cut(edges_src, edges_tgt)
#print(hamiltonian)

# Define the objective, return `<state(params) | H | state(params)>`
# Note that in the `observe` call we list the kernel, the hamiltonian, and then the concrete global variable values of our kernel
# followed by the parameters to be optimized.


def objective(parameters):
    return cudaq.observe(kernel_qaoa, hamiltonian, qubit_count, layer_count,
                         edges_src, edges_tgt, parameters).expectation()


optimizationStartTime = time.time()
# Optimize!
# Specify the optimizer and its initial parameters.
cudaq.set_random_seed(13)
optimizer = cudaq.optimizers.NelderMead()
if(opt == "COBYLA"):
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 100
    optimal_expectation, optimal_parameters = optimizer.optimize(
        dimensions=parameter_count, function=objective)
else:
    optimizer = cudaq.optimizers.NelderMead()
    optimal_expectation, optimal_parameters = optimizer.optimize(
        dimensions=parameter_count, function=objective)
np.random.seed(13)
optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8,
                                                 parameter_count)
#print("Initial parameters = ", optimizer.initial_parameters)
optimizationEndTime = time.time()

# Alternatively we can use the vqe call (just comment out the above code and uncomment the code below)
# optimal_expectation, optimal_parameters = cudaq.vqe(
#    kernel=kernel_qaoa,
#    spin_operator=hamiltonian,
#    argument_mapper=lambda parameter_vector: (qubit_count, layer_count, edges_src, edges_tgt, parameter_vector),
#    optimizer=optimizer,
#    parameter_count=parameter_count)

#print('optimal_expectation =', optimal_expectation)
#print('Therefore, the max cut value is at least ', -1 * optimal_expectation)
#print('optimal_parameters =', optimal_parameters)

# Sample the circuit using the optimized parameters
# Since our kernel has more than one argument, we need to list the values for each of these variables in order in the `sample` call.
samplingStartTime = time.time()
counts = cudaq.sample(kernel_qaoa, qubit_count, layer_count, edges_src,
                      edges_tgt, optimal_parameters, shots_count=1000)
samplingEndTime = time.time()
#print(counts)

# Identify the most likely outcome from the sample
maxStartTime = time.time()
max_cut = max(counts, key=lambda x: counts[x])
maxEndTime = time.time()
endTime = time.time()
#print('The max cut is given by the partition: ',
#      max(counts, key=lambda x: counts[x]))

#print(cudaq.__version__)



#print("Total Time: ")
#print(end_time - start_time)

optTime = optimizationEndTime - optimizationStartTime
sampTime = samplingEndTime - samplingStartTime
maxCutTime = maxEndTime - maxStartTime
totalTime = endTime - startTime

data = {
    "nodes": nodes,
    "edges": edges,
    "initial_parameters": optimizer.initial_parameters,
    "optimal_expectation": optimal_expectation,
    "max_cut_value_atleast": -1 * optimal_expectation,
    "optimal_parameters": optimal_parameters,
    "maxcut_partition": max(counts, key=lambda x: counts[x]),
    "cudaq_version": cudaq.__version__,
    "total_time": totalTime,
    "optimization_time": optTime,
    "sampling_time": sampTime
}


with open(f"{fileprefix}/numQubits_{num_qubits}_numLayers_{layer_count}_target_{target}_opt_{opt}.json", "w") as f:
    json.dump(data, f, indent=4)

with open(f"{fileprefix}/timeRecords.csv", 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    if csvfile.tell() == 0:
        writer.writerow(['JobID', 'NumberNodes', 'NumberLayers', 'TimeOptimize', 'TimeSample', 'MaxCutTime', 'TotalTime', 'MaxCutPartition', 'Target', 'Optimizer'])
    writer.writerow([os.environ.get("SLURM_JOB_ID"), num_qubits, layer_count, optTime, sampTime, maxCutTime, totalTime, max(counts, key=lambda x: counts[x]), target, opt])
