import surfacecode_qton as sc
# import stim
import numpy as np
import math
import qton
from qton import operators
import copy
import matplotlib.pyplot as plt
import pymatching
import networkx as nx

# g = nx.Graph()
# g.add_edge()

# 定义surface
sc.qdic = [[-1, 1, 2], [-2, 0, 3], [-1, 0, 1, 3, 4], [-2, 1, 2, 4, 5],
           [-2, 3, 4, 6, 7], [-1, 4, 5, 7, 8], [-2, 5, 8], [-1, 6, 7]]
sc.num_measure = 8
sc.num_data = 9


x_values = []
y_total_values = []
y_aver_values = []
y_std_values = []
repeat_num = 50
cycle_num = 10

dict_x_stabilizer = {'0000': [], '1000': [2], '0100': [0], '0010': [5], '0001': [6], '1100': [1],
                     '1010': [2, 5], '1001': [2, 6], '0110': [4], '0101': [0, 6], '0011': [7], '1110': [1, 5],
                     '1011': [2, 7], '1101': [1, 6], '0111': [0, 7], '1111': [1, 7]}
dict_z_stabilizer = {'0000': [], '1000': [0], '0100': [1], '0010': [6], '0001': [8], '1100': [0, 1],
                     '1010': [3], '1001': [0, 8], '0110': [4], '0101': [5], '0011': [6, 8], '1110': [0, 4],
                     '1011': [3, 8], '1101': [0, 5], '0111': [6, 5], '1111': [3, 5]}
fidelitys = np.zeros([repeat_num, cycle_num])
for r in range(repeat_num):  # 执行"repeat_num"次操作a（执行“cycle_num”次 surface cycle, 记该过程为操作a）
    init_state = []
    circuit = qton.Qcircuit(sc.num_data + sc.num_measure)
    circuit = sc.runcirc(sc.qdic, circuit)
    record = sc.get_record(circuit)
    init = copy.deepcopy(circuit.state)
    y_values = []
    for i in range(cycle_num):  # 执行“cycle_num”次 surface cycle, 记该过程为操作a
        for j in range(9):
            tmp = np.random.choice([0, 1, 2, 3], p=[0.99, 0.003, 0.003, 0.004])
            if tmp == 1:
                circuit.x(j)
            elif tmp == 2:
                circuit.y(j)
            elif tmp == 3:
                circuit.z(j)
        circuit = sc.runcirc(sc.qdic, circuit)
        sc.record = sc.get_record(circuit)
        # 查表法
        str_x = ''
        str_z = ''
        for w in range(sc.num_measure):
            if sc.qdic[w][0] == -1:
                if sc.record[-1][w] != sc.record[-2][w]:
                    str_x += '1'
                else:
                    str_x += '0'
                    # = ''.join([str_x, '0'])
            else:
                if sc.record[-1][w] != sc.record[-2][w]:
                    str_z = ''.join([str_z, '1'])
                else:
                    str_z = ''.join([str_z, '0'])
        for s in dict_x_stabilizer[str_x]:
            circuit.z(s)
        for t in dict_z_stabilizer[str_z]:
            circuit.x(t)


        # MWPM方法
        # check_matrix = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        #                          [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        #                          [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        #                          [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        # m = pymatching.Matching(check_matrix, repetitions=1)
        # prob_error = m.decode(sc.record[-1])
        # for k in range(9):
        #     if prob_error[k] == 1 and prob_error[k + 9] == 0:
        #         circuit.x(k)
        #     if prob_error[k] == 1 and prob_error[k + 9] == 1:
        #         circuit.y(k)
        #     if prob_error[k] == 0 and prob_error[k + 9] == 1:
        #         circuit.z(k)
        #     else:
        #         pass

        new = copy.deepcopy(circuit.state)
        y_dot = abs(np.dot(init, new)) ** 2
        fidelitys[r, i] = y_dot

f_avg = fidelitys.mean(axis=0)
error = fidelitys.std(axis=0)/np.sqrt(repeat_num)
plt.errorbar(x=range(cycle_num), y=f_avg, yerr=error, fmt='-o')
#         if r == 0:
#             x_values.append(i)
#         y_values.append(y_dot)
#     y_total_values.append(y_values)
# for j in range(1, cycle_num + 1):
#     tmp_y = []
#     for k in range(repeat_num):
#         tmp_y.append(y_total_values[k][j - 1])
#         print(np.mean(tmp_y))
#     y_aver_values.append(np.mean(tmp_y))
#     y_std_values.append(np.std(tmp_y)/np.sqrt(repeat_num))

# print(sc.record)
# 绘制折线图
# plt.plot(x_values, y_aver_values, label='aver-fidelity')
plt.errorbar(x_values, y_aver_values, yerr=y_std_values, fmt='o', elinewidth=0.3, capsize=3, label='error-bar')
# 添加标题和标签
plt.title('time-evolution fidelity')
plt.xlabel('cycle number')
plt.ylabel('fidelity')

# 添加图例
plt.legend()

# 显示图形
plt.show()