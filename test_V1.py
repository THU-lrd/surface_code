import numpy as np
import copy
import stim
from stim import Circuit

s = stim.TableauSimulator()


def MX(circ: Circuit, qdic: list, index: int):
    circ.append("H", [index])
    for i in range(1, len(qdic[index - 18])):
        circ.append("CNOT", [index, qdic[index - 18][i]])
    circ.append("H", [index])
    circ.append("M", [index])
    return circ


def MZ(circ: Circuit, qdic: list, index: int):
    for i in range(1, len(qdic[index - 18])):
        circ.append("CZ", [index, qdic[index - 18][i]])
    circ.append("M", [index])
    return circ


def runcirc(circ: Circuit, qdic: np.array):
    for i in range(0, len(qdic)):
        if qdic[i][0] == 0:
            circ = MX(circ, qdic, i + 18)
        else:
            circ = MZ(circ, qdic, i + 18)
    return circ


def measure_result(circuit: Circuit):
    s.do(circuit)
    record_tmp = np.array(list(map(int, s.current_measurement_record())))
    for k in range(0, len(record_tmp)):
        record_tmp[k] = 2 * (record_tmp[k] - 1 / 2)
    record_tmp = list(record_tmp)
    return record_tmp


def get_record():
    global circuit
    global record
    global measure_total
    tmp_result1 = copy.deepcopy(measure_result(circuit))
    measure_total.append(tmp_result1)
    current_total = tmp_result1[2 * len(measure_total[-2]):]
    record.append(current_total)
    return record


circuit = stim.Circuit()
# [0, *, *, *]表示X-measure qubit作用的data qubit, [1, *, *, *]表示Z-measure qubit作用的data qubit
qdic = [[0, 0, 1, 3], [0, 1, 2, 4], [1, 0, 3, 5], [1, 1, 3, 4, 6], [1, 2, 4, 7],
        [0, 3, 5, 6, 8], [0, 4, 6, 7, 9], [1, 5, 8, 10], [1, 6, 8, 9, 11],
        [1, 7, 9, 12], [0, 8, 10, 11, 13], [0, 9, 11, 12, 14], [1, 10, 13, 15],
        [1, 11, 13, 14, 16], [1, 12, 14, 17], [0, 13, 15, 16], [0, 14, 16, 17]]

circuit = runcirc(circuit, qdic)

initial = [measure_result(circuit)]
measure_total = initial
print('measure_total_initial= ', measure_total)
record = copy.deepcopy(initial)
print('record_initial= ', record)


# Z-cut hole
class Logical_Operators:
    """Creat logical operators.

    Examples: Z_L, X_L, H_L, CX_L, S_L, T_L

    """

    # record = []
    # circuit = globals()
    # qdic = []
    #
    # def __init__(self, record, circuit, qdic):
    #     self.record = record
    #     self.circuit = circuit
    #     self.qdic = qdic

    def cut(self, turn_off_1: int, turn_off_2: int,
            scale: int, square_X=True):

        global circuit
        global qdic
        global record
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
            record = get_record()
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
                qdic[int(i - 18)] = [qdic[int(i - 18)][0]]

            circuit = runcirc(circuit, qdic)
            print('cut1=', record)
            record = get_record()
            print('cut2=', record)

        return circuit, qdic, record

    def apply_cut(self, turn_off_1: int, turn_off_2: int,
                  scale: int, square_X=True):
        if square_X:
            def apply_xl():
                X_L = np.array([])
                for k in range(0, scale):
                    X_L = np.append(X_L,
                                    [turn_off_2 - 3 * (scale - 1) - 17 + k, turn_off_2 - 2 * (scale - 1) - 15 + k * 5,
                                     turn_off_2 + 3 * (scale - 1) - 18 - k, turn_off_2 + 2 * (scale - 1) - 20 - 5 * k])
                for k in X_L:
                    circuit.append("X", k)
                return circuit

            def apply_zl():
                Z_L = np.array([])
                j = turn_off_1 - 15
                while j <= turn_off_2 - 20:
                    Z_L = np.append(Z_L, j)
                    j += 5

                for k in Z_L:
                    circuit.append("Z", k)
                return circuit

        else:
            def apply_zl():
                Z_L = np.array([])
                for k in range(0, scale):
                    Z_L = np.append(Z_L,
                                    [turn_off_2 - 3 * (scale - 1) - 17 + k, turn_off_2 - 2 * (scale - 1) - 15 + k * 5,
                                     turn_off_2 + 3 * (scale - 1) - 18 - k, turn_off_2 + 2 * (scale - 1) - 20 - 5 * k])
                for k in Z_L:
                    circuit.append("Z", k)
                return circuit

            def apply_xl():
                X_L = np.array([])
                j = turn_off_1 - 15
                while j <= turn_off_2 - 20:
                    X_L = np.append(X_L, j)
                    j += 5

                for j in X_L:
                    circuit.append("X", j)
                return circuit

        return apply_xl, apply_zl

    def xcut_init(self, turn_off1: int, turn_off2: int, difficult=True):
        global circuit
        global qdic
        global record

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
                circuit = circuit.append("M", j)
                j += 5
            record = get_record()

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
            circuit = runcirc(circuit, qdic)
            record = get_record()
            return circuit, qdic, record

        else:
            circuit, qdic, record = self.cut(turn_off1, turn_off2, 1, True)
            return circuit, qdic, record

    def xmeasure(self, turn_off1: int, turn_off2: int, difficult=True):
        global circuit
        global qdic
        global record
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
            circuit = runcirc(circuit, qdic)

            # 初始化孤立数据比特
            j = turn_off1 - 15
            while j <= turn_off2 - 20:
                circuit = circuit.append("M", j)
                j += 5
            record = get_record()

            # 更新字典
            i = turn_off1
            while i < turn_off2:
                qdic[i - 18] = [0, i - 18, i - 20, i - 17, i - 15]
                qdic[i - 18 + 2] = qdic[i - 18 + 2].append(i - 15)
                qdic[i - 18 + 3] = qdic[i - 18 + 3].append(i - 15)
                i += 5
            # 更新电路

            circuit = runcirc(circuit, qdic)
            record = get_record()

            return circuit, qdic, record

        else:
            qdic[turn_off1 - 18] = [0, turn_off1 - 18, turn_off1 - 20, turn_off1 - 17, turn_off1 - 15]
            qdic[turn_off2 - 18] = [0, turn_off2 - 18, turn_off2 - 20, turn_off2 - 17, turn_off2 - 15]
            circuit = runcirc(circuit, qdic)
            record = get_record()
            return circuit, qdic, record

    def qmove(self, turn_off_1: int, turn_off_2: int, turn_to_2: int,
              X_cut=True):

        global circuit
        global qdic
        global record

        record = get_record()

        X_L, Z_L = self.apply_cut(turn_off_1, turn_off_2, 1, X_cut)
        m = min(turn_to_2, turn_off_2)
        M = max(turn_off_2, turn_to_2)

        # 更新字典
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
                circuit = runcirc(circuit, qdic)
        # 更新电路
        if not X_cut:

            if m == turn_off_2:  # 从上向下移动
                i = m + 5
                while i <= M:
                    circuit.append("MX", i - 20)
                    i += 5
            else:  # 从下向上移动
                i = M - 5
                while i >= m:
                    circuit.append("MX", i - 15)
                    i -= 5
        else:
            if m == turn_off_2:  # 从上向下移动
                i = m + 5
                while i <= M:
                    circuit.append("M", i - 20)
                    i += 5
            else:  # 从下向上移动
                i = M - 5
                while i >= m:
                    circuit.append("M", i - 15)
                    i -= 5

        record = get_record()
        record[-1].append('move1')

        # 更新字典
        if m == turn_off_2:
            i = m
            while i <= M - 5:
                qdic[i - 18] = [1, i - 18, i - 20, i - 17, i - 15]
                i += 5
        else:
            i = M
            while i >= m + 5:
                qdic[i - 18] = [1, i - 18, i - 20, i - 17, i - 15]
                i -= 5

        # 更新电路
        circuit = runcirc(circuit, qdic)
        record = get_record()
        record[-1].append('move2')

        # Byproduct operators
        phase2 = 1
        n = (M - m) / 5
        for i in record[-2][len(record[-2]) - 1: len(record[-2]) - 1 - n:-1]:  # 从后往前取n个
            phase2 *= i
        phase1 = 1
        i = m + 5
        while i <= M:
            phase1 *= record[-2][i - 18]
            phase1 *= record[-1][i - 5 - 18]
            i += 5

        if not X_cut:
            if phase2 == 1:
                p_X = 0
            else:
                p_X = 1
                record[-1].append('z_err')
            if phase1 == 1:
                p_Z = 0
            else:
                p_Z = 1
                record[-1].append('x_err')
        else:
            if phase2 == 1:
                p_Z = 0
            else:
                p_Z = 1
                record[-1].append('x_err')
            if phase1 == 1:
                p_X = 0
            else:
                p_X = 1
                record[-1].append('z_err')

        return circuit, qdic

    def qbraid_subx(self, scale: int, z1: int, z2: int, x2: int,
                   x1: int, X_L1=True, X_L2=True):  # 输入时默认index大小为z2>z1, x2>x1

        global circuit
        global qdic
        global record
        # 更新字典
        # 挖去一部分
        i = z2 + 5
        while i <= x2 + 2 * 3:
            qdic[i - 18] = [qdic[i - 18][0]]
            i += 5
        i = x2 + 2 * 3 + 1
        while i <= x2 + 3 * 3:
            qdic[i - 18] = [qdic[i - 18][0]]
            i += 1
        i = x2 + 3 * 3 - 5
        while i >= x2 - 2 * 3:
            qdic[i - 18] = [qdic[i - 18][0]]
            i -= 5

        circuit = runcirc(circuit, qdic)
        # 获取孤立比特X基测量结果
        i = z2 + 5
        while i <= x2 + 2 * 3:
            if X_L1:
                circuit.append("MX", i - 20)
            else:
                pass
            i += 5
        i = x2 + 2 * 3 + 1
        while i <= x2 + 3 * 3:
            if X_L1:
                circuit.append("MX", i - 18)
            else:
                pass
            i += 1
        i = x2 + 3 * 3 - 5
        while i >= x2 - 2 * 3:
            if X_L1:
                circuit.append("MX", i - 15)
            else:
                pass
            i -= 5

        record = get_record()

        # 填充
        i = z2
        while i <= x2 + 2 * 3:
            qdic[i - 18] = [1, i - 18, i - 20, i - 17, i - 15]
            i += 5
        i = x2 + 2 * 3 + 1
        while i <= x2 + 3 * 3:
            qdic[i - 18] = [1, i - 18, i - 20, i - 17, i - 15]
            i += 1
        i = x2 + 3 * 3 - 5
        while i >= x2 - 2 * 3:
            qdic[i - 18] = [1, i - 18, i - 20, i - 17, i - 15]
            i -= 5
        circuit = runcirc(circuit, qdic)
        # 更新 X_L 算子
        apply_XL, _ = self.apply_cut(z1, z2, 1, False)
        def new_XL():
            j = z2 + 5
            while j <= x2 + 2 * 3:
                if X_L1:
                    circuit.append("X", j - 20)
                else:
                    circuit.append("I", j - 20)
                j += 5
            j = x2 + 2 * 3 + 1
            while j <= x2 + 3 * 3:
                if X_L1:
                    circuit.append("X", j - 18)
                else:
                    circuit.append("I", j - 18)
                j += 1
            j = x2 + 3 * 3 - 5
            while j >= x2 - 2 * 3:
                if X_L1:
                    circuit.append("X", j - 15)
                else:
                    circuit.append("I", j - 15)
                j -= 5
            return circuit

        # 挖去剩下的
        i = x2 - 2 * 3 - 1
        while i >= z2:
            qdic[i - 18] = [qdic[i - 18][0]]
            i -= 1

        circuit = runcirc(circuit, qdic)

        i = x2 - 2 * 3 - 1
        while i >= z2:
            if X_L1:
                circuit.append("MX", i - 17)
            else:
                pass
            i -= 1

        record = get_record()

        # 填充
        i = x2 - 2 * 3
        while i > z2:
            qdic[i - 18] = [1, i - 18, i - 20, i - 17, i - 15]
            i -= 1

        i = x2 - 2 * 3 - 1
        while i >= z2:
            if X_L1:
                circuit.append("X", i - 17)
            else:
                circuit.append("I", i - 17)
            i -= 1

        # 更新电路
        if X_L1:
            circuit = runcirc(circuit, qdic)
            phase1 = 1
            phase2 = 1
            # byproduct operator
            l1 = record[-2][len(record[-2]) - 1: len(record[-2]) - 1 - 9:-1]  # 从后往前取9个(第一次打洞产生的孤立点)
            for i in l1:
                phase1 *= i
            l2 = record[-1][len(record[-2]) - 1: len(record[-2]) - 1 - 3:-1]  # 从后往前取3个(第二次打洞产生的孤立点)
            for i in l2:
                phase1 *= i
            for i in record[-1][x2 - 6: x2 - 6 + 1]:
                phase2 *= i
            for i in record[-1][x2 - 6 + 2: x2 - 6 + 2 + 1 * 5]:
                phase2 *= i
            for i in record[-1][x2 - 6 + 2 + 2 * 5: x2 - 6 + 2 + 2 * 5 - 1]:
                phase2 *= i
            for i in record[-1][x2 - 6 + 2 + 2 * 5 - 2: x2 - 6 + 2 + 2 * 5 - 2 - 1 * 5]:
                phase2 *= i

            if phase1 == 1:
                p_x1 = 0
            else:
                p_x1 = 1
                record[-1].append('1cut_z_err')
            if phase2 == 1:
                p_x2 = 0
            else:
                p_x2 = 1
                record[-1].append('2cut_z_err')
        else:
            pass
        return apply_XL, new_XL

    def qbraid(self, scale: int, z1: int, z2: int, x1: int, x2: int):  # 输入时默认z2>z1, x2>x1.
        global circuit
        apply_XL, new_XL = self.qbraid_subx(1, z1, z2, x1, x2)
        circuit = apply_XL()
        circuit = new_XL()
        return circuit

    def qbraid_subz(self, scale: int, z1: int, z2: int, x2: int,
                    x1: int, Z_L1=True, Z_L2=True):  # 输入时默认index大小为z2>z1, x2>x1
        global circuit
        global qdic
        global record
        # 扩充ZL2
        for i in range(x2 + 5 + 3, x2 + 5 + 3 + 1):
            for j in qdic[i - 18][1:]:
                if Z_L2:
                    circuit.append("Z", j)
                else:
                    circuit.append("I", j)

        i = x2 + 5 + 3 + 1 - 5
        while i <= x2 + 5 + 3 + 1 - 3*5:
            for j in qdic[i - 18][1:]:
                if Z_L2:
                    circuit.append("Z", j)
                else:
                    circuit.append("I", j)
            i -= 5

        i = x2 + 5 + 3 + 1 - 3*5 - 1
        while i >= z2:
            for j in qdic[i - 18][1:]:
                if Z_L2:
                    circuit.append("Z", j)
                else:
                    circuit.append("I", j)
            i -= 1

        # 还原ZL2
        record = get_record()
        phase2 = 1
        i = z2 + 1
        while i <= x2 + 5 + 3 + 1 - 3 * 5:
            phase2 *= record[-1][i - 18]
            i += 1

        i = x2 + 5 + 3 + 1 - 2*5
        while i <= x2 + 5 + 3 + 1:
            phase2 *= record[-1][i - 18]
            i -= 5

        phase2 *= record[-1][x2 + 5 + 3 - 18]

        if phase2 == 1:
            record[-1].append('cut2_x_err')
        else:
            pass












lo = Logical_Operators()

circ, qdic, a = lo.cut(16, 26, 1, False)
print(circ.diagram())
