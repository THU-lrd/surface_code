import surfacecode_qton as sc
import numpy as np
import qton
import copy
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# 定义surface
sc.qdic = [[-1, 1, 2], [-2, 0, 3], [-1, 0, 1, 3, 4], [-2, 1, 2, 4, 5],
           [-2, 3, 4, 6, 7], [-1, 4, 5, 7, 8], [-2, 5, 8], [-1, 6, 7]]
sc.num_measure = 8
sc.num_data = 9



y_total_values = []
y_aver_values = []
y_std_values = []
repeat_num = 50
cycle_num = 20
x_values = [k for k in range(cycle_num)]
dict_x_stabilizer = {'0000': [], '1000': [2], '0100': [0], '0010': [5], '0001': [6], '1100': [1],
                     '1010': [2, 5], '1001': [2, 6], '0110': [4], '0101': [0, 6], '0011': [7], '1110': [1, 5],
                     '1011': [2, 7], '1101': [1, 6], '0111': [0, 7], '1111': [1, 7]}
dict_z_stabilizer = {'0000': [], '1000': [0], '0100': [1], '0010': [6], '0001': [8], '1100': [0, 1],
                     '1010': [3], '1001': [0, 8], '0110': [4], '0101': [5], '0011': [6, 8], '1110': [0, 4],
                     '1011': [3, 8], '1101': [0, 5], '0111': [6, 5], '1111': [3, 5]}
fidelitys1 = np.zeros([repeat_num, cycle_num])
fidelitys2 = np.zeros([repeat_num, cycle_num])
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
        new1 = copy.deepcopy(circuit.state)
        y1_dot = abs(np.dot(init, new1)) ** 2
        fidelitys1[r, i] = y1_dot
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

        new2 = copy.deepcopy(circuit.state)
        y2_dot = abs(np.dot(init, new2)) ** 2
        fidelitys2[r, i] = y2_dot

f_avg2 = fidelitys2.mean(axis=0)
error2 = fidelitys2.std(axis=0)/np.sqrt(repeat_num)
# plt.errorbar(x=range(cycle_num), y=f_avg2, yerr=error2, fmt='-o')
f_avg1 = fidelitys1.mean(axis=0)
error1 = fidelitys1.std(axis=0)/np.sqrt(repeat_num)
# plt.errorbar(x=range(cycle_num), y=f_avg1, yerr=error1)

# 绘制折线图
# plt.plot(x_values, y_aver_values, label='aver-fidelity')
# plt.errorbar(x_values, y_aver_values, yerr=y_std_values, fmt='o', elinewidth=0.3, capsize=3, label='error-bar')
# # 添加标题和标签
# plt.title('time-evolution fidelity')
# plt.xlabel('cycle number')
# plt.ylabel('fidelity')

# 绘制指数曲线拟合图像
fig, ax = plt.subplots()
ax.plot(x_values, f_avg2, 'b--')
ax.plot(x_values, f_avg1, 'r--')
def target_func(x, a0, a1, a2):
    return a0 * np.exp(-x / a1) + a2

a0 = max(f_avg2) - min(f_avg2)
a1 = x_values[round(len(x_values) / 2)]
a2 = min(f_avg2)
p0 = [a0, a1, a2]
print('p0 = ', p0)
b0 = max(f_avg1) - min(f_avg1)
b1 = x_values[round(len(x_values) / 2)]
b2 = min(f_avg1)
p1 = [b0, b1, b2]
print('p1 = ', p1)
para2, cov = optimize.curve_fit(target_func, x_values, f_avg2, p0=p0)
print('para2 = ', para2)
para1, cov = optimize.curve_fit(target_func, x_values, f_avg1, p0=p1)
print('para1 = ', para1)
y_fit2 = [target_func(a, *para2) for a in x_values]
ax.plot(x_values, y_fit2, 'g')
y_fit1 = [target_func(a, *para1) for a in x_values]
ax.plot(x_values, y_fit1, 'g')

# 添加图例
plt.legend()

# 显示图形
plt.show()