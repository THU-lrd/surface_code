# import qton.operators.gates as qga
# from qton import *
import numpy as np
# import qton.simulators._basic_qcircuit_ as qc
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import stim


def MX(circ: QuantumCircuit, qdic: np.array, index: int):
    circ.h(index)
    for i in range(1, len(qdic[index - 18])):
        circ.cx(index, qdic[index - 18][i])
    circ.h(index)
    return circ


def MZ(circ: QuantumCircuit, qdic, index: int):
    for i in range(1, len(qdic[index - 18])):
        circ.cz(index, qdic[index - 18][i])
    return circ


circ = QuantumCircuit(35, 35)


def runcirc(circ: QuantumCircuit, qdic: np.array):
    for i in range(0, len(qdic)):
        if qdic[i][0] == 0:
            circ = MX(circ, qdic, i + 18)
        else:
            circ = MZ(circ, qdic, i + 18)
    return circ


# [0, *, *, *]表示X-measure qubit作用的data qubit, [1, *, *, *]表示Z-measure qubit作用的data qubit
qdic = [[0, 0, 1, 3], [0, 1, 2, 4], [1, 0, 3, 5], [1, 1, 3, 4, 6], [1, 2, 4, 7],
        [0, 3, 5, 6, 8], [0, 4, 6, 7, 9], [1, 5, 8, 10], [1, 6, 8, 9, 11],
        [1, 7, 9, 12], [0, 8, 10, 11, 13], [0, 9, 11, 12, 14], [1, 10, 13, 15],
        [1, 11, 13, 14, 16], [1, 12, 14, 17], [0, 13, 15, 16], [0, 14, 16, 17]]


# print(circ.draw())


# Z-cut hole
class Logical_Operators:
    """Creat logical operators.

    Examples: Z_L, X_L, H_L, CX_L, S_L, T_L

    """

    def cut(self, circuit: QuantumCircuit, qdic: list, turn_off_1: int, turn_off_2: int,
            scale: int, square_X=True):

        global cut
        label_X = np.array([])
        label_Z = np.array([])
        # X-cut(X-cut上的measure qubit的index)
        if square_X:
            for i in range(0, scale):
                label_X = np.append(label_X, [turn_off_2 - 3 * (scale - 1) + i, turn_off_2 - 2 * (scale - 1) + i * 5,
                                              turn_off_2 + 3 * (scale - 1) - i, turn_off_2 + 2 * (scale - 1) - 5 * i])
            # Z-cut(Z-cut上的measure qubit的index)
            i = turn_off_1 + 5
            while i <= turn_off_2 - 5:
                label_Z = np.append(label_Z, i)
                i += 5

            # 更新字典
            cut = np.append(label_X, turn_off_1)
            for i in cut:
                qdic[int(i - 18)] = [qdic[int(i - 18)][0]]

            circuit = runcirc(circuit, qdic)
        # Z-cut
        if not square_X:
            # 更新 logical_Z
            for i in range(0, scale):
                label_Z = np.append(label_X, [turn_off_2 - 3 * (scale - 1) + i, turn_off_2 - 2 * (scale - 1) + i * 5,
                                              turn_off_2 + 3 * (scale - 1) - i, turn_off_2 + 2 * (scale - 1) - 5 * i])
            # 更新 logical_X
            i = turn_off_1 + 5
            while i <= turn_off_2 - 5:
                label_X = np.append(label_X, i)
                i += 5

            # 更新字典
            cut = np.append(label_Z, turn_off_1)
            for i in cut:
                index = int(i - 18)
                qdic[int(i - 18)] = [qdic[int(i - 18)][0]]

            circuit = runcirc(circuit, qdic)

        return circuit, qdic

    def xcut_init(self, circuit: QuantumCircuit, qdic: list, turn_off1: int, turn_off2: int, difficult=True):

        if difficult:

            # 更新字典
            i = turn_off1
            while i < turn_off2:
                qdic[i - 18] = [0]
                qdic[i - 18 + 2] = qdic[i - 18 + 2].remove(i - 15)
                qdic[i - 18 + 3] = qdic[i - 18 + 3].remove(i - 15)
                i += 5
            qdic[turn_off2 - 18] = [0]
            # 更新电路
            circuit = runcirc(circuit, qdic)

            # 初始化孤立数据比特
            j = turn_off1 - 15
            while j <= turn_off2 - 20:
                circuit = circuit.measure(j, j)
                j += 5

            # 更新字典
            k = turn_off1 + 5
            while k <= turn_off2 - 5:
                qdic[k - 18] = [0, k - 18, k - 20, k - 17, k - 15]
                qdic[k - 18 - 2] = qdic[k - 18 - 2].append(k - 20)
                qdic[k - 18 - 3] = qdic[k - 18 - 3].append(k - 20)
                k += 5
            qdic[turn_off2 - 18 - 2] = qdic[turn_off2 - 18 - 2].append(turn_off2 - 20)
            qdic[turn_off2 - 18 - 3] = qdic[turn_off2 - 18 - 3].append(turn_off2 - 20)
            # 更新电路
            return circuit, qdic

        else:
            circuit, qdic = self.cut(circuit, qdic, turn_off1, turn_off2, 1, True)
            return circuit, qdic

    def xmeasure(self, circuit: QuantumCircuit, qdic: list, turn_off1: int, turn_off2: int, difficult=True):
        if difficult:
            # 更新字典
            k = turn_off1 + 5
            while k <= turn_off2 - 5:
                qdic[k - 18] = [0]
                qdic[k - 18 - 2] = qdic[k - 18 - 2].remove(k - 20)
                qdic[k - 18 - 3] = qdic[k - 18 - 3].remove(k - 20)
                k += 5
            qdic[turn_off2 - 18 - 2] = qdic[turn_off2 - 18 - 2].remove(turn_off2 - 20)
            qdic[turn_off2 - 18 - 3] = qdic[turn_off2 - 18 - 3].remove(turn_off2 - 20)
            # 更新电路

            # 初始化孤立数据比特
            j = turn_off1 - 15
            while j <= turn_off2 - 20:
                circuit = circuit.measure(j, j)
                j += 5

            # 更新字典
            i = turn_off1
            while i < turn_off2:
                qdic[i - 18] = [0, i - 18, i - 20, i - 17, i - 15]
                qdic[i - 18 + 2] = qdic[i - 18 + 2].append(i - 15)
                qdic[i - 18 + 3] = qdic[i - 18 + 3].append(i - 15)
                i += 5
            # 更新电路

            circuit = runcirc(circuit, qdic)

            return circuit, qdic

        else:
            qdic[turn_off1 - 18] = [0, turn_off1 - 18, turn_off1 - 20, turn_off1 - 17, turn_off1 - 15]
            qdic[turn_off2 - 18] = [0, turn_off2 - 18, turn_off2 - 20, turn_off2 - 17, turn_off2 - 15]
            circuit = runcirc(circuit, qdic)
            return circuit, qdic

    def qmove(self, scale: int, circ: QuantumCircuit, qdic: list, turn_off_1: int, turn_off_2: int, turn_to_2: int, Z_cut=True):

        circuit, qdic = self.cut(circ, qdic, turn_off_1, turn_off_2,
                                           scale, not Z_cut)

        m = min(turn_to_2, turn_off_2)
        M = max(turn_off_2, turn_to_2)

        # 更新字典
        """
        
        """
        if m == turn_off_2:
            i = m + 5
            while i <= M:
                qdic[i - 18] = [qdic[i - 18][0]]
                i += 5
        else:
            i = M - 5
            while i >= m:
                qdic[i - 18] = [qdic[i - 18][0]]
                i -= 5

        # 更新电路
        circuit = runcirc(circuit, qdic)

        return circuit, qdic


lo = Logical_Operators()

circ, qdic = lo.cut(circ, qdic, 16, 26, 1, False)


print(circ.draw('mpl'))
circ.draw('mpl')
plt.show()
