import surfacecode_qton as sc
# import stim
import numpy as np
import qton
import copy
import matplotlib.pyplot as plt
import pymatching
# 无噪声
sc.qdic = [[-1, 1, 2], [-2, 0, 3], [-1, 0, 1, 3, 4], [-2, 1, 2, 4, 5],
           [-2, 3, 4, 6, 7], [-1, 4, 5, 7, 8], [-2, 5, 8], [-1, 6, 7]]
sc.num_measure = 8
sc.num_data = 9
init_state = []
for i in range(0, 10):
    circuit = qton.Qcircuit(sc.num_data + sc.num_measure)
    # print(circuit.state)
    circuit = sc.runcirc(sc.qdic, circuit)
    circuit = sc.init_quiescent_state(circuit)
    sc.record = sc.get_record(circuit)
    # print(circuit.state)
    # x_values = []
    # y_values = []
    # init = copy.deepcopy(circuit.state)
    # init_state.append(circuit.state)
    if i > 1:
        print(sc.record[-1] == sc.record[-3])
    else:
        pass
print(sc.record)
# for i in range(0, 10):
#     circuit = sc.runcirc(sc.qdic, circuit)
#     record = sc.get_record(circuit)
#     new = copy.deepcopy(circuit.state)
#     print(record[-1] == record[-2])
#     y_dot = abs(np.vdot(init, new))**2
#     x_values.append(i)
#     y_values.append(y_dot)
#     print(y_dot)
#     print(np.dot(new, new))
#
# # 绘制折线图
# plt.plot(x_values, y_values, label='fidelity')
#
# # 添加标题和标签
# plt.title('time-evolution fidelity')
# plt.xlabel('cycle number')
# plt.ylabel('fidelity')
#
# # 添加图例
# plt.legend()
#
# # 显示图形
# plt.show()





