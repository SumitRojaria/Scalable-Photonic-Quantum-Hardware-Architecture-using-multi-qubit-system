from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

n = 4
qc = QuantumCircuit(n)


for q in range(n):
    qc.h(q)


qc.x([0, 2])
qc.h(3)
qc.mcx([0, 1, 2], 3)
qc.h(3)
qc.x([0, 2])


for q in range(n):
    qc.h(q)
    qc.x(q)

# H.X.H = Z (multi- controlled Z gate)
qc.h(3)
qc.mcx([0, 1, 2], 3)
qc.h(3)

for q in range(n):
    qc.x(q)
    qc.h(q)


qc.measure_all()

# accounting for the hardware constraints ( 0 ~ 1 ~ 2 ~ 3)
hardware_coupling_map = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
native_basis_gates = ['id', 'rz', 'sx', 'x', 'cx']

transpiled_qc = transpile(
    qc,
    basis_gates=native_basis_gates,
    coupling_map=hardware_coupling_map,
    optimization_level=3
)

transpiled_qc.draw('mpl', fold= 20)
simulator = AerSimulator()
result = simulator.run(transpiled_qc, shots=1024).result()
counts = result.get_counts()


print(counts)
print("Depth:", transpiled_qc.depth())
print("Gate counts:", transpiled_qc.count_ops())


plot_histogram(counts)
plt.show()