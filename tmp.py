from qton import Qcircuit
from qton import operators
# qc = Qcircuit(2)
# print(qc.measure(0))
# print(qc.state)
# qc.x(0)
# print(qc.measure(0))
# print(qc.state)
a, b = 1
print(a, b)
# model = circuit.detector_error_model(decompose_errors=True)
# print("model= ", model)
# matching = pymatching.Matching.from_detector_error_model(model)
# sampler = circuit.compile_detector_sampler()
# syndrome, actual_observables = sampler.sample(shots=1, separate_observables=True)
# num_errors = 0
# for i in range(syndrome.shape[0]):
#     predicted_observables = matching.decode(syndrome[i, :])
#     num_errors += not np.array_equal(actual_observables[i, :], predicted_observables)
#
# print(num_errors)  # prints 8
