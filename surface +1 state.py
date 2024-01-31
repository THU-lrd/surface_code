import surfacecode_qton as sc
import qton


# 定义surface
sc.qdic = [[-1, 1, 2], [-2, 0, 3], [-1, 0, 1, 3, 4], [-2, 1, 2, 4, 5],
           [-2, 3, 4, 6, 7], [-1, 4, 5, 7, 8], [-2, 5, 8], [-1, 6, 7]]
sc.num_measure = 8
sc.num_data = 9
init_state = []
circuit0 = qton.Qcircuit(sc.num_data + sc.num_measure)
circuit1 = sc.runcirc(sc.qdic, circuit0)
record = sc.get_record(circuit1)
print(circuit1.state)
# circuit = sc.init_quiescent_state(circuit)
for i in range(0, len(record[-1])):
    if record[-1][i] == -1:
        print(i)
        if sc.qdic[i][0] == -1:
            print(-1)
            circuit1.h(i + sc.num_data)
            # print("before init:", circuit.measure(i + sc.num_data))
            for j in range(1, len(sc.qdic[i])):
                # circuit.x(sc.qdic[i][j])
                circuit1.cx(i + sc.num_data, sc.qdic[i][j])
            circuit1.h(i + sc.num_data)
            circuit1.measure(i + sc.num_data)
            # print("after init:", circuit.measure(i + sc.num_data))
            sc.reset(circuit1, i + sc.num_data)
        else:
            print(-2)
            # print("before init:", circuit.measure(i + sc.num_data))
            for j in range(1, len(sc.qdic[i])):
                # circuit.z(sc.qdic[i][j])
                circuit1.cx(sc.qdic[i][j], i + sc.num_data)
            circuit1.measure(i + sc.num_data)
            sc.reset(circuit1, i + sc.num_data)
            # print("after init:", circuit.measure(i + sc.num_data))
circuit2 = sc.runcirc(sc.qdic, circuit1)
sc.record = sc.get_record(circuit2)
print(circuit2.state)
circuit3 = sc.runcirc(sc.qdic, circuit2)
sc.record = sc.get_record(circuit3)
print(circuit3.state)
circuit4 = sc.runcirc(sc.qdic, circuit3)
sc.record = sc.get_record(circuit4)
print(circuit4.state)
print(sc.record)

