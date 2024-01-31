import numpy as np
import copy
import stim
from matplotlib import pyplot as plt
from stim import Circuit

s = stim.TableauSimulator()

num_measure = 0
num_data = 1
width = 1
length = 1
gap = 1


def init(num_m: int, num_d: int, wide: int, len: int):
    global num_measure
    global num_data
    global width
    global length
    global gap
    num_measure = num_m
    num_data = num_d
    width = wide
    length = len
    gap = width - 1 + width
    return


def MX(circ: Circuit, qdict: list, index: int):
    circ.append("H", [index])
    for i in range(1, len(qdict[index - num_data])):
        circ.append("CNOT", [index, qdict[index - num_data][i]])
    circ.append("H", [index])
    circ.append("M", [index])
    return circ


def MZ(circ: Circuit, qdict: list, index: int):
    for i in range(1, len(qdict[index - num_data])):
        circ.append("CNOT", [qdict[index - num_data][i], index])
    circ.append("M", [index])
    return circ


def runcirc(qdict: np.array):
    circ = stim.Circuit()
    for i in range(0, len(qdict)):
        if qdic[i][0] == -1:
            circ = MX(circ, qdict, i + num_data)
        else:
            circ = MZ(circ, qdict, i + num_data)
    return circ


def measure_result(circ: Circuit):
    s.do(circ)
    # print('调用s.do')
    # print(len(s.current_measurement_record()))
    for i in range(0, num_measure):
        s.reset(i + num_data)
    tmp = copy.deepcopy(s.current_measurement_record())
    record_tmp = np.array(list(map(int, tmp)))
    for k in range(0, len(record_tmp)):
        record_tmp[k] = 2 * (record_tmp[k] - 1 / 2)
    record_tmp = list(record_tmp)
    return record_tmp


def get_record(circ: Circuit):
    global record
    global measure_total
    tmp_result1 = measure_result(circ)
    tmp1 = copy.deepcopy(tmp_result1)
    measure_total.append(tmp1)

    if len(measure_total) == 1:
        current_total = tmp1[0:  len(measure_total[-1])]
    else:
        current_total = tmp1[len(measure_total[-2]):]
    record.append(current_total)
    return record


def turn_off(remove_list: list):
    global qdic
    for i in remove_list:
        for j in qdic:
            if i in j:
                j.remove(i)
            else:
                pass
    return qdic


def turn_on(append_list: list):
    for i in append_list:
        for j in range(0, len(qdic_const)):
            if i in qdic_const[j] and i not in qdic[j]:
                qdic[j].append(i)
            else:
                pass
    return qdic


# qdic元素中[-1, *, *, *]表示X-measure qubit作用的data qubit, [-2, *, *, *]表示Z-measure qubit作用的data qubit

qdic = []
qdic_const = []


def get_qdic(xbound=True):
    global qdic
    global qdic_const
    if xbound:
        for i in range(0, width - 1):
            tmp = [-1, i, i + 1, i + width]
            qdic.append(tmp)
        for i in range(width - 1, num_measure - (width - 1)):
            if (i - (width - 1)) % gap == 0:
                tmp = [-2, i - (width - 1), i + 1, i + width]
                qdic.append(tmp)
            elif (i - 2 * (width - 1)) % gap == 0:
                tmp = [-2, i - (width - 1), i, i + width]
                qdic.append(tmp)
            else:
                if i % gap >= width - 1:
                    tmp = [-2, i - (width - 1), i, i + 1, i + width]
                    qdic.append(tmp)
                else:
                    tmp = [-1, i - (width - 1), i, i + 1, i + width]
                    qdic.append(tmp)
        for i in range(num_measure - (width - 1), num_measure):
            tmp = [-1, i - (width - 1), i, i + 1]
            qdic.append(tmp)
    else:
        for i in range(0, width - 1):
            tmp = [-2, i, i + 1, i + width]
            qdic.append(tmp)
        for i in range(width - 1, num_measure - (width - 1)):
            if (i - (width - 1)) % gap == 0:
                tmp = [-1, i - (width - 1), i + 1, i + width]
                qdic.append(tmp)
            elif (i - 2 * (width - 1)) % gap == 0:
                tmp = [-1, i - (width - 1), i, i + width]
                qdic.append(tmp)
            else:
                if i % gap >= width - 1:
                    tmp = [-1, i - (width - 1), i, i + 1, i + width]
                    qdic.append(tmp)
                else:
                    tmp = [-2, i - (width - 1), i, i + 1, i + width]
                    qdic.append(tmp)
        for i in range(num_measure - (width - 1), num_measure):
            tmp = [-2, i - (width - 1), i, i + 1]
            qdic.append(tmp)
    qdic_const = copy.deepcopy(qdic)
    return qdic


record = []
measure_total = []

"""byproduct_err非空时，其元素为长度为 3 的列表[a, b, c]，a代表发生x or z 错误的比特位置，
b = 0 代表错误类型为x， b = 1 代表错误类型为z， c代表发生错误的时间（c代表的是该错误发生在第c次测量和第c + 1次测量之间）"""
byproduct_err = []


class logical_qubit:
    """
        当qubit没有cut时，输入cut1 = cut2 = scale = 0, XL = ZL = [].
    """
    def __init__(self, cut1: int, cut2: int, cut_scale: int, ZL_position: list, XL_position: list):
        self.cut1 = cut1
        self.cut2 = cut2
        self.scale = cut_scale
        self.ZL = ZL_position
        self.XL = XL_position

    # 以下每个逻辑门操作开始前都要get record，以便记录最新的error syndrome
    @classmethod
    def cut(self, turn_off_1: int, turn_off_2: int, scale: int, x_cut=True):
        global qdic
        global record
        label = np.array([])  # 记录的是包含于cut里面的除了最外层的measure qubit的index
        label0 = np.array([])  # 记录的是包含于cut里面的最外层的measure qubit的index
        isolate = []  # 记录的是包含于cut里面的最外层的data qubit的index(cut中的孤立数据比特的index，存储在isolate中)

        for k in range(0, scale - 1):
            for j in range(0, k):
                label = np.append(label, [turn_off_1 - width * k + j, turn_off_1 - (width - 1) * k + gap * j,
                                          turn_off_1 + width * k - j, turn_off_1 + (width - 1) * k - gap * j,
                                          turn_off_2 - width * k + j, turn_off_2 - (width - 1) * k + gap * j,
                                          turn_off_2 + width * k - j, turn_off_2 + (width - 1) * k - gap * j])
        label = [int(item) for item in label]

        k = (scale - 1)
        j = 0
        while j < k:
            label0 = np.append(label0, [turn_off_1 - width * k + j, turn_off_1 - (width - 1) * k + gap * j,
                                        turn_off_1 + width * k - j, turn_off_1 + (width - 1) * k - gap * j,
                                        turn_off_2 - width * k + j, turn_off_2 - (width - 1) * k + gap * j,
                                        turn_off_2 + width * k - j, turn_off_2 + (width - 1) * k - gap * j])
            j += 1
        label0 = [int(item) for item in label0]
        label0.append(turn_off_1)
        label0.append(turn_off_2)

        for j in list(label0):
            qdic[j - num_data] = [qdic[j - num_data][0]]
        for j in list(label):
            isolate = list(set(qdic[j - num_data][1:]) | set(isolate))

        # 更新字典
        qdic = turn_off(isolate)
        circuit = runcirc(qdic)
        record = get_record(circuit)
        circuit_tmp = circuit.copy()

        to_mea = np.array([])

        j = 0
        while j < scale - 1:
            to_mea = np.append(to_mea, [turn_off_1 - width * scale - num_data + 1 + j,
                                        turn_off_1 - (width - 1) * scale - num_data + width + gap * j,
                                        turn_off_1 + width * scale - num_data - j,
                                        turn_off_1 + (width - 1) * scale - num_data - (width - 1) - gap * j,
                                        turn_off_2 - width * scale - num_data + 1 + j,
                                        turn_off_2 - (width - 1) * scale - num_data + width + gap * j,
                                        turn_off_2 + width * scale - num_data - j,
                                        turn_off_2 + (width - 1) * scale - num_data - (width - 1) - gap * j])
            j += 1
            to_mea = [int(item) for item in to_mea]

        for k in to_mea:
            if x_cut:
                circuit_tmp.append("M", k)
            else:
                circuit_tmp.append("MX", k)
        record = get_record(circuit_tmp)
        j = 1
        while j < scale:
            record[-1][turn_off_1 - width * scale + j - num_data] *= record[-1][num_measure + 8 * (j - 1) - 1]
            record[-1][turn_off_1 - (width - 1) * scale + gap * j - num_data] *= record[-1][
                num_measure + 8 * (j - 1) + 1 - 1]
            record[-1][turn_off_1 + width * scale - j - num_data] *= record[-1][num_measure + 8 * (j - 1) + 2 - 1]
            record[-1][turn_off_1 + (width - 1) * scale - gap * j - num_data] *= record[-1][
                num_measure + 8 * (j - 1) + 3 - 1]
            record[-1][turn_off_2 - width * scale + j - num_data] *= record[-1][num_measure + 8 * (j - 1) + 4 - 1]
            record[-1][turn_off_2 - (width - 1) * scale + gap * j - num_data] *= record[-1][
                num_measure + 8 * (j - 1) + 5 - 1]
            record[-1][turn_off_2 + width * scale - j - num_data] *= record[-1][num_measure + 8 * (j - 1) + 6 - 1]
            record[-1][turn_off_2 + (width - 1) * scale - gap * j - num_data] *= record[-1][
                num_measure + 8 * (j - 1) + 7 - 1]
            j += 1
        circuit_tmp.clear()
        record[-1] = record[-1][0: num_measure]
        self.scale = scale
        self.cut1 = turn_off_1
        self.cut2 = turn_off_2
        return circuit, qdic, record

    def apply_cut(self, scale: int, turn_off_1: int, turn_off_2: int, square_X=True):
        """
         this function can only be used after the surface code has cut holes, and the parameter: turn_off_1 <
         turn_off_2 is the index of the measure qubits in the center of the two holes
        """
        global qdic
        global record
        if square_X:
            def apply_xl(circuit: Circuit):
                X_L = np.array([])
                for k in range(0, scale):
                    X_L = np.append(X_L,
                                    [turn_off_2 - num_data + 1 - width * scale + k,
                                     turn_off_2 - num_data + width - (width - 1) * scale + k * gap,
                                     turn_off_2 - num_data + width * scale - k,
                                     turn_off_2 - num_data - (width - 1) + (width - 1) * scale - gap * k])
                X_L = [int(item) for item in X_L]
                self.XL = X_L
                return circuit

            def apply_zl(circuit: Circuit):
                Z_L = np.array([])
                j = turn_off_1 + (width - 1) * scale - num_data + 1
                while j <= turn_off_2 - width * scale - num_data + 1:
                    Z_L = np.append(Z_L, j)
                    j += gap
                Z_L = [int(item) for item in Z_L]
                self.ZL = Z_L
                return circuit

        else:
            def apply_zl(circuit: Circuit):
                Z_L = np.array([])
                for k in range(0, scale):
                    Z_L = np.append(Z_L,
                                    [turn_off_2 - width * scale - num_data + 1 + k,
                                     turn_off_2 - (width - 1) * scale - num_data + width + k * gap,
                                     turn_off_2 + width * scale - num_data - k,
                                     turn_off_2 + (width - 1) * scale - num_data - (width - 1) - gap * k])
                Z_L = [int(item) for item in Z_L]
                self.ZL = Z_L
                return circuit

            def apply_xl(circuit: Circuit):
                X_L = np.array([])
                j = turn_off_1 + (width - 1) * scale - num_data + 1
                while j <= turn_off_2 - width * scale - num_data + 1:
                    X_L = np.append(X_L, j)
                    j += gap
                X_L = [int(item) for item in X_L]
                self.XL = X_L
                return circuit

        return apply_xl, apply_zl

    def cut_init(self, scale: int, turn_off1: int, turn_off2: int, difficult=True, xcut=True):
        global qdic
        global record

        if difficult:
            # 更新字典
            i = turn_off1
            while i < turn_off2:
                qdic[i - num_data] = [qdic[i - num_data][0]]
                qdic[i - num_data + (width - 1)].remove(i - num_data + width)
                qdic[i - num_data + width].remove(i - num_data + width)
                i += gap
            qdic[turn_off2 - num_data] = [qdic[turn_off2 - num_data][0]]
            # 更新电路
            circuit = runcirc(qdic)
            record = get_record(circuit)

            # error track
            circuit_tmp = circuit.copy()
            j = turn_off1 - num_data + width
            while j <= turn_off2 - num_data - (width - 1):
                if xcut:
                    circuit_tmp.append("M", j)
                else:
                    circuit_tmp.append("MX", j)
                j += gap
            record = get_record(circuit_tmp)
            j = turn_off1 - num_data + width
            k = 1
            while j <= turn_off2 - num_data - (width - 1):
                record[-1][j] *= record[-1][num_measure + k - 1]
                record[-1][j - 1] *= record[-1][num_measure + k - 1]
                k += 1
                j += gap
            record[-1] = record[-1][0: num_measure]
            circuit_tmp.clear()
            # 初始化孤立数据比特
            j = turn_off1 - num_data + width
            while j <= turn_off2 - num_data - (width - 1):
                if xcut:
                    s.reset(j)
                else:
                    s.reset_x(j)
                j += gap
            # 更新字典
            k = turn_off1 + gap
            while k <= turn_off2 - gap:
                qdic[k - num_data] = [qdic[k - num_data][0], k - num_data, k - num_data - (width - 1),
                                      k - num_data + 1, k - num_data + width]
                qdic[k - num_data - (width - 1)].append(k - num_data - (width - 1))
                qdic[k - num_data - width].append(k - num_data - (width - 1))
                k += gap
            qdic[turn_off2 - num_data - (width - 1)].append(turn_off2 - num_data - (width - 1))
            qdic[turn_off2 - num_data - width].append(turn_off2 - num_data - (width - 1))
            # 更新电路
            circuit = runcirc(qdic)
            record = get_record(circuit)
            return circuit, qdic, record

        else:
            circuit1 = runcirc(qdic)
            _ = get_record(circuit1)
            stab_phase = 1
            if scale > 1:
                for k in range(0, scale - 1):
                    stab_phase *= record[-1][turn_off2 - width * (scale - 1) + k - num_data]
                    stab_phase *= record[-1][turn_off2 - (width - 1) * (scale - 1) + k * gap - num_data]
                    stab_phase *= record[-1][turn_off2 + width * (scale - 1) - k - num_data]
                    stab_phase *= record[-1][turn_off2 + (width - 1) * (scale - 1) - gap * k - num_data]
            else:
                stab_phase *= record[-1][turn_off2 - num_data]

            circuit2, qdic2, _ = self.cut(turn_off1, turn_off2, scale, xcut)
            l = np.array([])  # 用于记录需要被初始化的cut边界上的data qubit的下标
            for k in range(0, scale):
                l = np.append(l,
                              [turn_off2 - width * scale - num_data + 1 + k,
                               turn_off2 - (width - 1) * scale - num_data + width + k * gap,
                               turn_off2 + width * scale - num_data - k,
                               turn_off2 + (width - 1) * scale - num_data - (width - 1) - gap * k])
            L = [int(item) for item in l]
            if stab_phase == 1:
                for k in L:
                    if xcut:
                        s.reset_x(k)
                    else:
                        s.reset(k)
            else:
                for k in L:
                    if xcut:
                        s.reset_x(k)
                        s.z(k)
                    else:
                        s.reset(k)
                        s.x(k)
            return circuit2, qdic2, record

    def cut_measure(self, scale: int, turn_off1: int, turn_off2: int, difficult=True, xcut=True):
        global qdic
        global record
        if difficult:
            # 更新字典
            k = turn_off1 + gap
            while k <= turn_off2 - gap:
                qdic[k - num_data] = [qdic[k - num_data][0]]
                qdic[k - num_data - (width - 1)].remove(k - num_data - (width - 1))
                qdic[k - num_data - width].remove(k - num_data - (width - 1))
                k += gap
            qdic[turn_off2 - num_data - (width - 1)].remove(turn_off2 - num_data - (width - 1))
            qdic[turn_off2 - num_data - width].remove(turn_off2 - num_data - (width - 1))
            # 更新电路
            circuit = runcirc(qdic)
            record = get_record(circuit)
            circuit_tmp = circuit.copy()
            # error track
            j = turn_off1 - num_data + width
            while j <= turn_off2 - num_data - (width - 1):
                if xcut:
                    circuit_tmp.append("M", j)
                else:
                    circuit_tmp.append("MX", j)
                j += gap
            record = get_record(circuit_tmp)
            j = turn_off1 - num_data + width
            k = 1
            while j <= turn_off2 - num_data - (width - 1):
                record[-1][j] *= record[-1][num_measure + k - 1]
                record[-1][j - 1] *= record[-1][num_measure + k - 1]
                k += 1
                j += gap
            record[-1] = record[-1][0: num_measure]
            circuit_tmp.clear()
            # 初始化孤立数据比特
            j = turn_off1 - num_data + width
            while j <= turn_off2 - num_data - (width - 1):
                if xcut:
                    s.reset(j)
                else:
                    s.reset_x(j)
                j += gap

            # 更新字典
            i = turn_off1
            while i <= turn_off2:
                qdic[i - num_data] = [qdic[i - num_data][0], i - num_data, i - num_data - (width - 1),
                                      i - num_data + 1, i - num_data + width]
                qdic[i - num_data + (width - 1)].append(i - num_data + width)
                qdic[i - num_data + width].append(i - num_data + width)
                i += gap
            # 更新电路
            circuit = runcirc(qdic)
            record = get_record(circuit)

            return circuit, qdic, record

        else:
            qdic[turn_off1 - num_data] = [qdic[turn_off1 - num_data][0], turn_off1 - num_data,
                                          turn_off1 - num_data - (width - 1),
                                          turn_off1 - num_data + 1,
                                          turn_off1 - num_data + width]
            qdic[turn_off2 - num_data] = [qdic[turn_off2 - num_data][0],
                                          turn_off2 - num_data, turn_off2 - num_data - (width - 1),
                                          turn_off2 - num_data + 1,
                                          turn_off2 - num_data + width]
            circuit = runcirc(qdic)
            record = get_record(circuit)
            return circuit, qdic, record

    def qmove(self, scale: int, turn_off_1: int, turn_off_2: int, turn_to_2: int, line_op_pos: int,
              X_cut=True, toward_Left=False, toward_Right=False, toward_Up=False, toward_Down=True):
        """
        :param line_op_pos: the index of the data qubit on the intersection of the cut boundary and the line operator.
                            we choose it be the outermost element.
        :return: circuit, qdic, record
        """
        global qdic
        global record
        global byproduct_err
        # check parameter (empty or multi-value)
        a = not toward_Left and not toward_Right and not toward_Up and toward_Down
        b = not toward_Left and not toward_Right and toward_Up and not toward_Down
        c = not toward_Left and toward_Right and not toward_Up and not toward_Down
        d = toward_Left and not toward_Right and not toward_Up and not toward_Down
        if toward_Left or toward_Right or toward_Up or toward_Down:
            if a or b or c or d:
                pass
            else:
                print('the move direction cannot be multi-value ')
        else:
            print('the move direction cannot be empty ')

        m = min(turn_to_2, turn_off_2)
        M = max(turn_off_2, turn_to_2)
        phaseX = 1
        phaseZ = 1

        if toward_Left or toward_Right:
            multiplier = 1
            move_distance = M - m
        else:
            multiplier = gap
            move_distance = (M - m) / gap

        err_corr_distance = min(move_distance - 2 * (scale - 1), 4 * scale)
        if err_corr_distance < move_distance:
            n = int(move_distance / err_corr_distance)
        else:
            n = 0
        if m == turn_off_2:
            p = 1
        else:  # M == turn_off_2:
            p = -1
        for l in range(0, n + 1):
            label_off = []  # 记录需要被挖去的与cut同类型的measure qubit的index
            isolate = []  # 记录孤立数据比特的index
            f2 = int((width - 1) * (2 * multiplier - (gap + 1)) / (gap - 1))
            for k in range(0, scale):
                i = (turn_off_2 + p * l * err_corr_distance - p * (scale - 1) * width +
                     p * multiplier * scale + p * (- multiplier + gap + 1) * k)
                while (i <= p * min(
                        p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                        p * turn_to_2) + p * (scale - 1) * f2 + p * (- multiplier + gap + 1) * k):
                    """
                    if multiplier = gap, (- multiplier + gap + 1) = 1;                    
                    if multiplier = 1, (- multiplier + gap + 1) = gap;
                    
                    if multiplier = gap, f2 = width - 1;                    
                    if multiplier = 1, f2 = 1 - width.
                    """
                    label_off.append(i)
                    i += p * multiplier
            circuit = runcirc(qdic)
            record = get_record(circuit)
            for j in label_off:
                isolate = list(set(qdic[j - num_data][1:]) | set(isolate))
                qdic[j - num_data] = [qdic[j - num_data][0]]
                if X_cut:
                    phaseX *= record[-1][j - num_data]
                else:
                    phaseZ *= record[-1][j - num_data]
            qdic = turn_off(isolate)

            circuit = runcirc(qdic)
            record = get_record(circuit)
            circuit_tmp = circuit.copy()
            to_measure = []  # 记录需要被测量的孤立数据比特的下标
            for r in [0, scale - 1]:
                i = (turn_off_2 + p * l * err_corr_distance - p * (scale - 1) * width + p * multiplier * scale + p * (
                        - multiplier + gap + 1) * r)
                f5 = int((1 - width) * (multiplier - 1) / (gap - 1))
                while (i <= p * min(
                        p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                        p * turn_to_2) + p * (scale - 1) * f2 + p * (- multiplier + gap + 1) * r):
                    to_measure.append(i - num_data + p * (f5 + int((p - 1) / 2)))
                    """
                       if multiplier = gap, f5 = 1 - width;                    
                       if multiplier = 1, f5 = 0;
                             
                       if multiplier = gap, f2 = width - 1;                    
                       if multiplier = 1, f2 = 1 - width.
                    """
                    i += p * multiplier
            f4 = int(((1 - width) * multiplier + width * gap - 1) / (gap - 1))
            for r in range(0, scale - 1):
                to_measure.append(p * min(
                    p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                    p * turn_to_2) + p * (scale - 1) * f2 + p * (- multiplier + gap + 1) * r
                                  - num_data + p * (f4 + int((p - 1) / 2)))
                """
                   if multiplier = gap, f2 = width - 1;                    
                   if multiplier = 1, f2 = 1 - width.
                   
                   if multiplier = gap, f4 = 1;                    
                   if multiplier = 1, f4 = width。
                """
            for i in to_measure:
                if X_cut:
                    circuit_tmp.append("M", i)
                else:
                    circuit_tmp.append("MX", i)
            record = get_record(circuit_tmp)
            r = 0
            k = 0
            i = (turn_off_2 + p * l * err_corr_distance - p * (scale - 1) * width + p * multiplier * scale + p * (
                    - multiplier + gap + 1) * r)
            f1 = int(((width - 1) * multiplier + gap - width) / (gap - 1))
            while (i <= p * min(
                    p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                    p * turn_to_2) + p * (scale - 1) * f2 + p * (- multiplier + gap + 1) * r):
                if line_op_pos == (turn_off_1 + p * (scale - 1) * f2 - num_data + p * (f1 + int((p - 1) / 2))):
                    """
                       if multiplier = gap, f1 = width;                    
                       if multiplier = 1, f1 = 1;
                       
                       if multiplier = gap, f2 = width - 1;                    
                       if multiplier = 1, f2 = 1 - width.
                    """
                    while (i <= p * min(
                            p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                            p * turn_to_2) + p * (scale - 1) * f2 + p * (- multiplier + gap + 1) * r -
                           p * int(l / n) * (multiplier * (scale - 1))):
                        if X_cut:
                            phaseZ *= record[-1][num_measure + k]
                        else:
                            phaseX *= record[-1][num_measure + k]
                else:
                    pass
                record[-1][i - num_data - p * width] *= record[-1][num_measure + k]
                i += p * multiplier
                k += 1
            r = scale - 1
            i = (turn_off_2 + p * l * err_corr_distance - p * (scale - 1) * width + p * multiplier * scale + p * (
                    - multiplier + gap + 1) * r)
            while (i <= p * min(
                    p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                    p * turn_to_2) + p * (scale - 1) * f2 + p * (- multiplier + gap + 1) * r):
                if line_op_pos == (turn_off_1 + p * (width - 1) * (scale - 1) - num_data + p * (f1 + int((p - 1) / 2))):
                    """
                       if multiplier = gap, f1 = width;                    
                       if multiplier = 1, f1 = 1;
                       
                       if multiplier = gap, f2 = width - 1;                    
                       if multiplier = 1, f2 = 1 - width;
                    """
                    while (i <= p * min(p * (
                            turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                                        p * turn_to_2) + p * (scale - 1) * f2 + p * (- multiplier + gap + 1) * r -
                           p * int(l / n) * (multiplier * (scale - 1))):
                        if X_cut:
                            phaseZ *= record[-1][num_measure + k]
                        else:
                            phaseX *= record[-1][num_measure + k]
                else:
                    pass
                record[-1][i - num_data - p * f2] *= record[-1][num_measure + k]
                i += p * multiplier
                k += 1
            for r in range(0, scale - 1):
                record[-1][p * min(
                    p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                    p * turn_to_2) + p * (scale - 1) * f2 + p * (- multiplier + gap + 1) * r + p * width] \
                    *= record[-1][num_measure + k]
                k += 1
                """
                   if multiplier = gap, f2 = width - 1;                    
                   if multiplier = 1, f2 = 1 - width;
    
                   if multiplier = gap, int(((1 - width) * multiplier + width * gap - 1)/(gap - 1)) = 1;                    
                   if multiplier = 1, int(((1 - width) * multiplier + width * gap - 1)/(gap - 1)) = width.
                """
            record[-1] = record[-1][0: num_measure]
            circuit_tmp.clear()

            # 更新字典
            label_on = []  # 记录需要turn on的与cut同类型的measure qubit的index
            isolate_on = []  # 记录孤立数据比特的index
            for k in range(0, scale):
                i = (turn_off_2 + p * l * err_corr_distance - p * (scale - 1) * width + p * (
                        - multiplier + gap + 1) * k)
                while (i <= p * min(
                        p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                        p * turn_to_2) + p * (scale - 1) * f2 - p * multiplier * scale + p * (
                               - multiplier + gap + 1) * k):
                    """
                    if multiplier = gap, (- multiplier + gap + 1) = 1;                    
                    if multiplier = 1, (- multiplier + gap + 1) = gap;
    
                    if multiplier = gap, f2 = width - 1;                    
                    if multiplier = 1, f2 = 1 - width.
                    """
                    label_on.append(i)
                    i += p * multiplier
            for j in label_off:
                isolate_on = list(set(qdic[j - num_data][1:]) | set(isolate_on))
                qdic[j - num_data] = [qdic[j - num_data][0], j - num_data, j - num_data - (width - 1),
                                      j - num_data + 1, j - num_data + width]
            qdic = turn_on(isolate_on)
            circuit = runcirc(qdic)
            record = get_record(circuit)
            circuit_tmp = circuit.copy()
            for r in range(0, scale - 1):
                to_measure.append(p * min(
                    p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                    p * turn_to_2) - p * (scale - 1) * width + p * (- multiplier + gap + 1) * r
                                  - num_data + p * (f4 + int((p - 1) / 2)))
                """
                   if multiplier = gap, f2 = width - 1;                    
                   if multiplier = 1, f2 = 1 - width.
    
                   if multiplier = gap, f4 = 1;                    
                   if multiplier = 1, f4 = width。
                """
            for i in to_measure:
                if X_cut:
                    circuit_tmp.append("M", i)
                else:
                    circuit_tmp.append("MX", i)
            record = get_record(circuit_tmp)
            for r in range(0, scale - 1):
                record[-1][p * min(
                    p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                    p * turn_to_2) - p * (scale - 1) * width + p * (- multiplier + gap + 1) * r - p * f2] \
                    *= record[-1][num_measure + k]
                k += 1
                """
                   if multiplier = gap, f2 = width - 1;                    
                   if multiplier = 1, f2 = 1 - width.
                """
            record[-1] = record[-1][0: num_measure]
            circuit_tmp.clear()

            for j in label_on:
                if X_cut:
                    phaseX *= record[-1][j - num_data]
                else:
                    phaseZ *= record[-1][j - num_data]

            # byproduct operator
            f1 = int(((width - 1) * multiplier + gap - width) / (gap - 1))
            if X_cut:
                if phaseZ == 1:
                    pass
                else:
                    if line_op_pos == (turn_off_1 + p * (scale - 1) * f2 - num_data + p * (f1 + int((p - 1) / 2))):
                        """
                           if multiplier = gap, f1 = width;                    
                           if multiplier = 1, f1 = 1;
    
                           if multiplier = gap, f2 = width - 1;                    
                           if multiplier = 1, f2 = 1 - width;
                        """
                        byproduct_err.append(
                            [p * min(p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance
                                          - p * multiplier * scale), p * turn_to_2) - p * multiplier * scale
                             + p * (scale - 1) * f2 - num_data + p * (f1 + int((p - 1) / 2)), 0, len(record) - 1])
                    else:
                        """
                           line_op_pos == (turn_off_1 + p * (width - 1) * (scale - 1) - num_data + p * (f1 + int((p - 1)/2)))
                           if multiplier = gap, f2 = width - 1;                    
                           if multiplier = 1, f2 = 1 - width;
                        
                           if multiplier = gap, f1 = width;                    
                           if multiplier = 1, f1 = 1.
                        """
                        byproduct_err.append(
                            [p * min(p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance
                                          - p * multiplier * scale), p * turn_to_2) - p * multiplier * scale
                             + p * (width - 1) * (scale - 1) - num_data + p * (f1 + int((p - 1) / 2)), 0,
                             len(record) - 1])

                if phaseX == 1:
                    pass
                else:
                    byproduct_err.append(
                        [p * min(
                            p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                            p * turn_to_2) - p * multiplier * scale
                         + p * (scale - 1) * f2 - num_data + p * (f1 + int((p - 1) / 2)), 1, len(record) - 1])
            if not X_cut:
                if phaseX == 1:
                    pass
                else:
                    if line_op_pos == (turn_off_1 + p * (scale - 1) * f2 - num_data + p * (f1 + int((p - 1) / 2))):
                        byproduct_err.append(
                            [p * min(p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance
                                          - p * multiplier * scale), p * turn_to_2) - p * multiplier * scale
                             + p * (scale - 1) * f2 - num_data + p * (f1 + int((p - 1) / 2)), 1, len(record) - 1])
                    elif line_op_pos == (
                            turn_off_1 + p * (width - 1) * (scale - 1) - num_data + p * (f1 + int((p - 1) / 2))):
                        byproduct_err.append(
                            [p * min(p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance
                                          - p * multiplier * scale), p * turn_to_2) - p * multiplier * scale
                             + p * (width - 1) * (scale - 1) - num_data + p * (f1 + int((p - 1) / 2)), 1,
                             len(record) - 1])

                if phaseZ == 1:
                    pass
                else:
                    byproduct_err.append(
                        [p * min(
                            p * (turn_off_2 + p * (l + 1) * multiplier * err_corr_distance - p * multiplier * scale),
                            p * turn_to_2) - p * multiplier * scale
                         + p * (scale - 1) * f2 - num_data + p * (f1 + int((p - 1) / 2)), 1, len(record) - 1])
        circuit = runcirc(qdic)
        self.scale = scale
        self.cut1 = turn_off_1
        self.cut2 = turn_to_2
        return circuit, qdic

    def qbraid_subx(self, scale: int, z1: int, z2: int, x2: int,
                    x1: int, X_L1=True, X_L2=True, Z_L1=False, Z_L2=False):  # 输入时假设需要被编织的两个cut中心索引为z1, x2
        """因为此时的图已经是有两个洞的图，洞中的孤立数据比特已在相应基下测量并纠正过，
        因此在计算braid过程中X_L or Z_L 的移动产生的Z or X错误时，
        原始洞中的孤立比特测量结果不用计入在内，因为它在cut函数里已被测量过，
        并经过纠错后已成为相应可观测量的+1特征态 """
        global qdic
        global record
        global byproduct_err
        phasez1 = 1
        phasez2 = 1
        distance = int((max(z1, x2) - min(z1, x2)) / width)
        if Z_L1:
            for j in range(0, scale):
                i = x2 - width * distance - width * (scale - 1) - gap * scale + j
                while i <= x2 + (width - 1) * distance + (width - 1) * (scale - 1) + j:
                    phasez1 *= record[-1][i - num_data]
                    i += gap
                i = x2 + (width - 1) * distance + (width - 1) * (scale - 1) + scale - j * gap
                while i <= x2 + width * distance + width * (scale - 1) - j * gap:
                    phasez1 *= record[-1][i - num_data]
                    i += 1
                i = x2 + width * distance + width * (scale - 1) - gap * scale - j
                while i >= x2 - (width - 1) * distance - (width - 1) * (scale - 1) - j:
                    phasez1 *= record[-1][i - num_data]
                    i -= gap
            else:
                pass

        # 更新字典
        # 挖去一部分
        isolate = []  # 记录的是包含于cut里面的data qubit的下标（to be applied turn_off function）
        isolate0 = []  # 记录的是包含于cut里面的与cut同类型的（即与最外圈的stabilizer同类型的）measure qubit的index
        for j in range(0, scale):
            i = x2 - width * distance - width * (scale - 1) - gap * scale + j
            while i <= x2 + (width - 1) * distance + (width - 1) * (scale - 1) + j:
                isolate0.append(i)
                i += gap
            i = x2 + (width - 1) * distance + (width - 1) * (scale - 1) + scale - j * gap
            while i <= x2 + width * distance + width * (scale - 1) - j * gap:
                isolate0.append(i)
                i += 1
            i = x2 + width * distance + width * (scale - 1) - gap * scale - j
            while i >= x2 - (width - 1) * distance - (width - 1) * (scale - 1) - j:
                isolate0.append(i)
                i -= gap

        for j in list(isolate0):
            isolate = list(set(qdic[j - num_data][1:]) | set(isolate))
        for r in isolate0:
            qdic[r - num_data] = [qdic[r - num_data][0]]
        qdic = turn_off(isolate)

        circuit = runcirc(qdic)
        circuit_tmp = circuit.copy()
        # 获取孤立比特X基测量结果

        measure_isolate = []

        for j in [1 - scale,
                  scale - 1]:  # let m = int((j + scale + 1)/2), if j = (1 - scale), m = 1, if j = scale - 1, m = scale.
            i = x2 - width * distance - width * j - gap * int((j + scale + 1) / 2)
            while i <= x2 + (width - 1) * distance + (width - 1) * j:
                measure_isolate.append(i - num_data - (width - 1))
                i += gap
            i = x2 + (width - 1) * distance + (width - 1) * j + int((j + scale + 1) / 2)
            while i <= x2 + width * distance + width * j:
                measure_isolate.append(i - num_data)
                i += 1
            i = x2 + width * distance + width * j - gap * int((j + scale + 1) / 2)
            while i >= x2 - (width - 1) * distance - int(j * (width - 1 / 2) / (1 - scale) - 1 / 2) * j:
                measure_isolate.append(i - num_data + (width - 1))
                i -= gap

        for i in measure_isolate:
            circuit_tmp.append("MX", i)
        record = get_record(circuit_tmp)
        phasex1 = 1
        # 修正 qdic，补偿为stabilizer形式
        k = 1
        j = 1 - scale
        i = x2 - width * distance - width * j - gap * 1
        while i <= x2 + (width - 1) * distance + (width - 1) * j:
            record[-1][i - num_data - (width - 1)] *= record[-1][num_measure + k - 1]
            phasex1 *= record[-1][num_measure + k - 1]
            i += gap
            k += 1
        i = x2 + (width - 1) * distance + (width - 1) * j + 1
        while i <= x2 + width * distance + width * j:
            record[-1][i - num_data - width] *= record[-1][num_measure + k - 1]
            phasex1 *= record[-1][num_measure + k - 1]
            i += 1
            k += 1
        i = x2 + width * distance + width * j - gap * 1
        while i >= x2 - (width - 1) * distance - int(j * (width - 1 / 2) / (1 - scale) - 1 / 2) * j:
            record[-1][i - num_data + (width - 1)] *= record[-1][num_measure + k - 1]
            phasex1 *= record[-1][num_measure + k - 1]
            i -= gap

        j = scale - 1
        i = x2 - width * distance - width * j - gap * scale
        while i <= x2 + (width - 1) * distance + (width - 1) * j:
            record[-1][i - num_data - width] *= record[-1][num_measure + k - 1]
            i += gap
            k += 1
        i = x2 + (width - 1) * distance + (width - 1) * j + scale
        while i <= x2 + width * distance + width * j:
            record[-1][i - num_data + (width - 1)] *= record[-1][num_measure + k - 1]
            i += 1
            k += 1
        i = x2 + width * distance + width * j - gap * scale
        while i >= x2 - (width - 1) * distance - int(j * (width - 1 / 2) / (1 - scale) - 1 / 2) * j:
            record[-1][i - num_data + width] *= record[-1][num_measure + k - 1]
            i -= gap

        record[-1] = record[-1][0: num_measure]
        circuit_tmp.clear()

        # 填充
        isolate_on = []  # 记录的是包含于cut里面的data qubit的index
        isolate0 = []  # 记录的是包含于cut里面的与cut同类型的（即与最外圈的stabilizer同类型的）measure qubit的index
        for j in range(0, scale):
            i = x2 - width * distance - width * (scale - 1) + j
            while i <= x2 + (width - 1) * distance + (width - 1) * (scale - 1) + j:
                isolate0.append(i)
                i += gap
            i = x2 + (width - 1) * distance + (width - 1) * (scale - 1) + scale - j * gap
            while i <= x2 + width * distance + width * (scale - 1) - j * gap:
                isolate0.append(i)
                i += 1
            i = x2 + width * distance + width * (scale - 1) - gap * scale - j
            while i >= x2 - (width - 1) * distance - (width - 1) * (scale - 1) + gap * scale - j:
                isolate0.append(i)
                i -= gap
        for j in list(isolate0):
            isolate_on = list(set(qdic[j - num_data][1:]) | set(isolate_on))
        for i in isolate0:
            qdic[i - num_data] = [qdic[i - num_data][0], i - num_data, i - num_data - (width - 1), i - num_data + 1,
                                  i - num_data + width]
        qdic = turn_on(isolate_on)
        circuit = runcirc(qdic)
        record = get_record(circuit)
        if Z_L1:
            for j in range(0, scale):
                i = x2 - width * distance - width * (scale - 1) + j
                while i <= x2 + (width - 1) * distance + (width - 1) * (scale - 1) + j:
                    phasez1 *= record[-1][i - num_data]
                    i += gap
                i = x2 + (width - 1) * distance + (width - 1) * (scale - 1) + scale - j * gap
                while i <= x2 + width * distance + width * (scale - 1) - j * gap:
                    phasez1 *= record[-1][i - num_data]
                    i += 1
                i = x2 + width * distance + width * (scale - 1) - gap * scale - j
                while i >= x2 - (width - 1) * distance - (width - 1) * (scale - 1) + gap * scale - j:
                    phasez1 *= record[-1][i - num_data]
                    i -= gap
                i = x2 - (width - 1) * distance - (width - 1) * (scale - 1) - scale - j * gap
                while i >= x2 - width * distance - width * (scale - 1) - j * gap:
                    phasez1 *= record[-1][i - num_data]
                    i -= 1
            else:
                pass

        # 挖去剩下的
        isolate = []  # 记录的是包含于cut里面的data qubit的index
        isolate0 = []  # 记录的是包含于cut里面的与cut同类型的（即与最外圈的stabilizer同类型的）measure qubit的index
        for j in range(0, scale):
            i = x2 - (width - 1) * distance - (width - 1) * (scale - 1) - scale - j * gap
            while i >= x2 - width * distance - width * (scale - 1) - j * gap:
                isolate0.append(i)
                i -= 1
        for j in list(isolate0):
            isolate = list(set(qdic[j - num_data][1:]) | set(isolate))
        for r in isolate0:
            qdic[r - num_data] = [qdic[r - num_data][0]]
        qdic = turn_off(isolate)

        circuit = runcirc(qdic)
        circuit_tmp = circuit.copy()
        measure_isolate = []

        for j in [0, scale - 1]:
            i = x2 - (width - 1) * distance - (width - 1) * (scale - 1) - scale - j * gap
            while i >= x2 - width * distance - width * (scale - 1) - j * gap:
                measure_isolate.append(i - num_data + 1)
                i -= 1

        for i in measure_isolate:
            circuit_tmp.append("MX", i)
        record = get_record(circuit_tmp)

        k = 1
        i = x2 - (width - 1) * distance - (width - 1) * (scale - 1) - scale
        while i >= x2 - width * distance - width * (scale - 1):
            record[-1][i - num_data + width] *= record[-1][num_measure + k - 1]
            i -= 1
            k += 1

        j = scale - 1
        i = x2 - (width - 1) * distance - (width - 1) * (scale - 1) - scale - j * gap
        while i >= x2 - width * distance - width * (scale - 1) - j * gap:
            record[-1][i - num_data - (width - 1)] *= record[-1][num_measure + k - 1]
            phasex1 *= record[-1][num_measure + k - 1]
            i -= 1
            k += 1

        circuit_tmp.clear()
        record[-1] = record[-1][0: num_measure]

        # 填充
        isolate_on = []  # 记录的是包含于cut里面的data qubit的index
        isolate0 = []  # 记录的是包含于cut里面的与cut同类型的（即与最外圈的stabilizer同类型的）measure qubit的index
        for j in range(0, scale):
            i = x2 - (width - 1) * distance - (width - 1) * (scale - 1) - j * gap
            while i >= x2 - width * distance - width * (scale - 1) + scale - j * gap:
                isolate0.append(i)
                i -= 1
        for j in list(isolate0):
            isolate_on = list(set(qdic[j - num_data][1:]) | set(isolate_on))
        for i in isolate0:
            qdic[i - num_data] = [qdic[i - num_data][0], i - num_data, i - num_data - (width - 1), i - num_data + 1,
                                  i - num_data + width]
        qdic = turn_on(isolate_on)

        # 更新电路
        circuit = runcirc(qdic)
        record = get_record(circuit)

        if Z_L1:
            for j in range(0, scale):
                i = x2 - (width - 1) * distance - (width - 1) * (scale - 1) - j * gap
                while i >= x2 - width * distance - width * (scale - 1) + scale - j * gap:
                    phasez1 *= record[-1][i - num_data]
                    i -= 1
        else:
            pass

        phasex2 = 1
        # byproduct operator

        label = np.array([])  # 记录的是包含于X_L构成的LOOP内部且在上面的Xcut之外的X stabilizer对应的measure qubit的index
        for k in range(scale + 1, distance - (scale - 1)):
            j = 0
            while j < k:
                label = np.append(label, [x2 - width * k + j, x2 - (width - 1) * k + gap * j,
                                          x2 + width * k - j, x2 + (width - 1) * k - gap * j])
                j += 1
        label = [int(item) for item in label]
        for k in label:
            phasex2 *= record[-1][k]

        if X_L1:
            if phasex1 == 1:
                pass
            else:
                byproduct_err.append([z1 - num_data + width, 1, len(record) - 3])
            if phasex2 == 1:
                pass
            else:
                byproduct_err.append([x2 - num_data - (width - 1), 1, len(record) - 1])
        else:
            pass

        if Z_L2:
            if phasez1 == 1:
                pass
            else:
                byproduct_err.append([z1 - num_data + width, 0, len(record) - 2])
            if phasez2 == 1:
                pass
            else:
                byproduct_err.append([x2 + gap - num_data - (width - 1), 0, len(record) - 1])

        return circuit, qdic

    def Hadamard(self, scale: int, turn_off1: int, turn_off2: int, x_cut=True):  # 输入时按照turn_off1>turn_off2计算
        global circuit
        global qdic
        global record
        # global byproduct_err
        lenth = turn_off1 - turn_off2 + 2 * scale * gap
        p = int(lenth / 2)
        # 将要处理的logical_qubit对应的cut分离出来，通过在这个cut外围用turnoff手段挖loop，将cut隔离在一个子块里
        # 更新qdic
        isolate = []  # 记录的index of the measure qubit which is to be turned off
        i = turn_off1 - scale * gap - p
        while i <= turn_off1 - scale * gap + lenth - p - 1:
            isolate.append(i + 1)
            i += 1

        i = turn_off1 - scale * gap + lenth - p
        while i <= turn_off1 - scale * gap + lenth - p + (lenth - 1) * gap:
            isolate.append(i - width)
            i += gap

        i = turn_off1 - scale * gap + lenth - p + lenth * gap
        while i >= turn_off1 - scale * gap + lenth - p + lenth * gap - lenth - 1:
            isolate.append(i)
            i -= 1

        i = turn_off1 - scale * gap + lenth - p + lenth * gap - lenth
        while i >= turn_off1 - scale * gap - p:
            isolate.append(i - (width - 1))
            i -= gap
        for r in isolate:
            qdic[r - num_data] = [qdic[r - num_data][0]]
        circuit = runcirc(qdic)
        circuit_tmp = circuit.copy()
        for k in isolate:
            if not x_cut:
                circuit_tmp.append("M", k)
            else:
                circuit_tmp.append("MX", k)
        record = get_record(circuit_tmp)
        k = 0
        i = turn_off1 - scale * gap - p
        while i <= turn_off1 - scale * gap + lenth - p - 1:
            record[-1][i + width] *= record[-1][num_measure + k]
            record[-1][i - (width - 1)] *= record[-1][num_measure + k]
            i += 1
            k += 1

        i = turn_off1 - scale * gap + lenth - p
        while i <= turn_off1 - scale * gap + lenth - p + (lenth - 1) * gap:
            record[-1][i + width] *= record[-1][num_measure + k]
            record[-1][i + (width - 1)] *= record[-1][num_measure + k]
            i += gap
            k += 1

        i = turn_off1 - scale * gap + lenth - p + lenth * gap
        while i >= turn_off1 - scale * gap + lenth - p + lenth * gap - lenth - 1:
            record[-1][i - width] *= record[-1][num_measure + k]
            record[-1][i + (width - 1)] *= record[-1][num_measure + k]
            i -= 1
            k += 1

        i = turn_off1 - scale * gap + lenth - p + lenth * gap - lenth
        while i >= turn_off1 - scale * gap - p:
            record[-1][i - width] *= record[-1][num_measure + k]
            record[-1][i - (width - 1)] *= record[-1][num_measure + k]
            i -= gap
            k += 1

        record[-1] = record[-1][0: num_measure]

        #   形变下面的cut_operator------------------------------------------------------------------------------------------------
        label2 = np.array([])
        for i in range(0, scale):  # (cut内部的measure qubit的index，存储在label2中)
            for j in range(0, i):
                label2 = np.append(label2, [
                    turn_off2 - width * i + j, turn_off2 - (width - 1) * i + gap * j,
                    turn_off2 + width * i - j, turn_off2 + (width - 1) * i - gap * j])
        label2 = [int(item) for item in label2]
        label2.append(turn_off2)
        i2 = int(turn_off2 - (scale - 1) * gap - (width - 1) + lenth - p - 1)
        i1 = int(turn_off2 - (scale - 1) * gap - width - p + 1)
        index_z = []
        for r in range(0, scale + 2):
            for i in range(i1 + gap * r, i2 + gap * r + 1):
                index_z.append(i)
        for k in list(label2):
            index_z.remove(k)
            # 形变需要乘的stabilizer的测量结果的乘积
        stabilizer_product = 1
        for i in index_z:
            stabilizer_product *= record[-1][i - num_data]

        #   截取两洞之间没有cut的子surface code--------------------------------------------------------------------------------------
        i1 = turn_off1 - (scale - 1) * gap - width - (p - 1)
        i2 = turn_off1 - (scale - 1) * gap - (width - 1) + (lenth - p - 1)
        index = []  # 用于记录需要挖去的measure qubit的下标
        for r in range(0, scale + 1):
            for i in range(i1 + gap * r, i2 + gap * r + 1):
                index.append(i)

        i5 = turn_off1 - (scale - 1) * gap - (p - 1)
        i6 = turn_off1 - (scale - 1) * gap + (lenth - p - 1)
        for r in range(0, scale):
            for i in range(i5 + gap * r, i6 + gap * r + 1):
                index.append(i)

        i3 = turn_off2 - width - (p - 1)
        i4 = turn_off2 - (width - 1) + (lenth - p - 1)
        for r in range(0, scale + 1):
            for i in range(i3 + gap * r, i4 + gap * r + 1):
                index.append(i)

        i7 = turn_off2 - (p - 1)
        i8 = turn_off2 + (lenth - p - 1)
        for r in range(0, scale):
            for i in range(i7 + gap * r, i8 + gap * r + 1):
                index.append(i)

        small_lenth = turn_off1 - turn_off2 - 2 * (scale - 1)
        q = int(small_lenth / 2)
        i9 = turn_off1 + gap - (p - 1)
        i10 = turn_off1 + gap - q
        for r in range(0, small_lenth + 1):
            for i in range(i9 + gap * r, i10 + gap * r + 1):
                index.append(i)

        i11 = turn_off1 + gap + (width - 1) - (p - 1)
        i12 = turn_off1 + gap + (width - 1) - q
        for r in range(0, small_lenth):
            for i in range(i11 + gap * r, i12 + gap * r + 1):
                index.append(i)

        i13 = turn_off1 + gap + (lenth - p - 1)
        i14 = turn_off1 + gap + (small_lenth - q) + 1
        for r in range(0, small_lenth + 1):
            for i in range(i14 + gap * r, i13 + gap * r + 1):
                index.append(i)

        i15 = turn_off1 + gap + width + (lenth - p - 1)
        i16 = turn_off1 + gap + width + (small_lenth - q) + 1
        for r in range(0, small_lenth):
            for i in range(i16 + gap * r, i15 + gap * r + 1):
                index.append(i)
        # ------------------------------------------------------------------------------
        for r in index:
            qdic[r - num_data] = [qdic[r - num_data][0]]
        # ------------------------------------------------------------------------------
        # 记录的是围绕中间的sub surface code外的需要被测量用作与stabilizer测量结果作对比的 data qubit的index
        top_bottom_isolate = []
        left_right_isolate = []
        for i in range(turn_off1 - num_data + width - (q - 1), turn_off1 - num_data + width + small_lenth - q + 1):
            top_bottom_isolate.append(i)
        for i in range(turn_off2 - num_data + width - (q - 1), turn_off2 - num_data + width + small_lenth - q + 1):
            top_bottom_isolate.append(i)
        k1 = turn_off1 - num_data + width - q + gap
        while k1 <= turn_off1 - num_data + width - q + small_lenth * gap:
            left_right_isolate.append(k1)
            k1 += gap
        k2 = turn_off1 - num_data + width + small_lenth - q + 1 + gap
        while k2 <= turn_off1 - num_data + width + small_lenth - q + 1 + small_lenth * gap:
            left_right_isolate.append(k2)
            k2 += gap

        qdic = turn_off(top_bottom_isolate)
        qdic = turn_off(left_right_isolate)
        # -----------------------------------------------------------------------------------------------------------------------
        circuit = runcirc(qdic)
        circuit_tmp = circuit.copy()

        for k in top_bottom_isolate:
            if not x_cut:
                circuit_tmp.append("MX", k)
            else:
                circuit_tmp.append("M", k)
        for k in left_right_isolate:
            if not x_cut:
                circuit_tmp.append("M", k)
            else:
                circuit_tmp.append("MX", k)
        record = get_record(circuit_tmp)

        k = 0
        for i in range(turn_off1 - num_data + width - (q - 1), turn_off1 - num_data + width + small_lenth - q + 1):
            record[-1][i + (width - 1)] *= record[-1][num_measure + k]
            k += 1
        for i in range(turn_off2 - num_data + width - (q - 1), turn_off2 - num_data + width + small_lenth - q + 1):
            record[-1][i - width] *= record[-1][num_measure + k]
            k += 1
        k1 = turn_off1 - num_data + width - q + gap
        while k1 <= turn_off1 - num_data + width - q + small_lenth * gap:
            record[-1][i] *= record[-1][num_measure + k]
            k1 += gap
            k += 1
        k2 = turn_off1 - num_data + width + small_lenth - q + 1 + gap
        while k2 <= turn_off1 - num_data + width + small_lenth - q + 1 + small_lenth * gap:
            record[-1][i + 1] *= record[-1][num_measure + k]
            k2 += gap
            k += 1
        record[-1] = record[-1][0: num_measure]
        # -----------------------------------------------------------------------------------------------------------------------
        data_index = []  # 记录sub surface code中的数据比特的下标（从上到下的顺序）
        for r in range(0, small_lenth + 1):
            for i in range(turn_off1 + gap - num_data - q + 1 + r * gap,
                           turn_off1 + gap - num_data + 1 + (small_lenth - q) + 1 + r * gap):
                data_index.append(i)
            for i in range(turn_off1 + gap - num_data + width - q + 1 + r * gap,
                           turn_off1 + gap - num_data + width + (small_lenth - q) + 1 + r * gap):
                data_index.append(i)
        data_index2 = []  # 记录sub surface code中的数据比特的下标（从左到右的顺序）
        for i in range(0, small_lenth + 1):
            k1 = turn_off1 + gap - num_data - q + 1 + i
            while k1 <= turn_off1 + gap - num_data - q + 1 + small_lenth * gap + i:
                data_index2.append(k1)
                k1 += gap
            if i == small_lenth:
                k2 = turn_off1 + gap - num_data + width - q + 1 + i
                while k2 <= turn_off1 + gap - num_data + width - q + 1 + (small_lenth - 1) * gap + i:
                    data_index2.append(k2)
                    k2 += gap
            else:
                pass
        #  对挖出来的sub surface code中的每一个数据比特作用H门
        for r in data_index:
            circuit.append("H", r)
        #  对挖出来的sub surface code中的每一个数据比特作用SWAP门
        # （1）between patch data qubit and the measure qubit above it(从上到下)
        for k in data_index:
            circuit.append("SWAP", [k + num_data - width, k])
        # （2）between the measure qubit and the patch data qubit to its left(从左到右)
        for j in data_index2:
            circuit.append("SWAP", [j + num_data - 1, j])

        #  turn on至大的patch----------------------------------------------------------------------------------------------------
        turn_off1_new = turn_off1 + gap - q - (
                small_lenth - q) * gap - 1  # turn_off1 + gap - num_data - q + 1 - width - (small_lenth - q)*gap + num_data + (width - 1) - 1
        turn_off2_new = turn_off1_new + 2 * (scale - 1) + small_lenth
        label = np.array([])  # 记录的是包含于cut里面的measure qubit的index

        for k in range(0, scale):
            for j in range(0, k):
                label = np.append(label, [turn_off1_new - width * k + j, turn_off1_new - (width - 1) * k + gap * j,
                                          turn_off1_new + width * k - j, turn_off1_new + (width - 1) * k - gap * j,
                                          turn_off2_new - width * k + j, turn_off2_new - (width - 1) * k + gap * j,
                                          turn_off2_new + width * k - j, turn_off2_new + (width - 1) * k - gap * j])
        label = np.append(label, [turn_off1_new, turn_off2_new])
        label = [int(item) for item in label]

        label_data = np.array([])  # 记录的是包含于cut里面的data qubit的index
        if scale > 1:
            for k in range(0, scale):
                for j in range(0, k):
                    label_data = np.append(label_data,
                                           [turn_off1_new - width * k - num_data + 1 + j,
                                            turn_off1_new - (width - 1) * k - num_data + width + gap * j,
                                            turn_off1_new + width * k - num_data - j,
                                            turn_off1_new + (width - 1) * k - num_data - (width - 1) - gap * j,
                                            turn_off2_new - width * k - num_data + 1 + j,
                                            turn_off2_new - (width - 1) * k - num_data + width + gap * j,
                                            turn_off2_new + width * k - num_data - j,
                                            turn_off2_new + (width - 1) * k - gap * j])
        else:
            pass
        label_data = [int(item) for item in label_data]
        #  这里的index 在前面的挖成小patch时定义的index(挖去的measure qubit)
        for r in label:
            index.remove(r)
        index_data = []
        for j in index:
            if qdic[j - num_data] == -1:
                if x_cut:
                    pass
                else:  # set(qdic[j - num_data][1:])与set(index_data)的并集
                    index_data = list(set(qdic[j - num_data][1:]) | set(index_data))
            else:
                if not x_cut:
                    pass
                else:
                    index_data = list(set(qdic[j - num_data][1:]) | set(index_data))
        for r in label_data:
            index_data.remove(r)
        for k in index:
            for j in index_data:
                if j in qdic_const[k - num_data][1:]:
                    qdic[k - num_data].append(j)
        circuit = runcirc(qdic)
        record = get_record(circuit)

        #   形变下面的cut_operator------------------------------------------------------------------------------------------------
        label2 = np.array([])  # (cut内部的measure qubit的index，存储在label2中)
        for i in range(0, scale):
            j = 0
            while j < i:
                label2 = np.append(label2, [
                    turn_off2_new - width * i + j, turn_off2_new - (width - 1) * i + gap * j,
                    turn_off2_new + width * i - j, turn_off2_new + (width - 1) * i - gap * j])
        label2 = [int(item) for item in label2]
        label2.append(turn_off2_new)
        index_z = []
        r2 = int(turn_off1 + (width - 1) * scale - scale * gap)
        r1 = copy.deepcopy(r2 + (lenth - 2) * gap)
        for r in range(0, gap + lenth - p + (width - 1) * scale):
            i = r2 + r
            while i <= r1 + r:
                index_z.append(i)
                i += gap

        for k in list(label2):
            index_z.remove(k)
            # 形变需要乘的stabilizer的测量结果的乘积
        stabilizer_product = 1
        for i in index_z:
            stabilizer_product *= record[-1][i - num_data]

        #   move cut (step1) 没有添加byproduct--------------------------------------------------------------------------------
        isolate = []  # 记录的是包含于cut里面的data qubit的下标（to be applied turn_off function）
        isolate0 = []  # 记录的是包含于cut里面的与cut同类型的（即与最外圈的stabilizer同类型的）measure qubit的index
        for j in range(0, scale):
            i = turn_off2_new - (scale - 1) * (width - 1) + gap * scale - j
            while i <= turn_off2 + lenth - p - 1 - (scale - 1) + width * (scale - 1) - j:
                isolate0.append(i)
                i += gap
            i = turn_off2 + lenth - p - 1 + width * (scale - 1) - scale - j * gap
            while i >= turn_off2 + (scale - 1) - (width - 1) * (scale - 1) - j * gap:
                isolate0.append(i)
                i -= 1
            i = turn_off1_new + (scale - 1) * (width - 1) - gap * scale + j
            while i >= turn_off1 - p + 1 + (scale - 1) - width * (scale - 1) + j:
                isolate0.append(i)
                i -= gap
            i = turn_off1 - p + 1 + (scale - 1) - width * (scale - 1) + scale + j * gap
            while i <= turn_off1 - (scale - 1) + (width - 1) * (scale - 1) + j * gap:
                isolate0.append(i)
                i += 1

        for j in list(isolate0):
            isolate = list(set(qdic[j - num_data][1:]) | set(isolate))
        for r in isolate0:
            qdic[r - num_data] = [qdic[r - num_data][0]]
        qdic = turn_off(isolate)

        circuit = runcirc(qdic)
        circuit_tmp = circuit.copy()
        # 获取孤立比特X基测量结果
        measure_isolate = []
        for j in [1 - scale, scale - 1]:
            """
            let m = int((j + scale + 1)/2), 
            if j = (1 - scale), m = 1,
            if j = scale - 1, m = scale.
            """
            i = turn_off2_new - j * (width - 1) + gap * int((j + scale + 1) / 2)
            while i <= turn_off2 + lenth - p - 1 - (scale - 1) + width * j:
                measure_isolate.append(i - num_data - (width - 1))
                i += gap
            i = turn_off2 + lenth - p - 1 - (scale - 1) + width * j
            while i >= turn_off2 + (scale - 1) - int(j / 2 * (scale - 1) - (width - 1 / 2)) * j + 1:
                """
                 let m = int(j/2*(scale - 1) - (width - 1/2)), 
                 if j = 1 - scale, m = -width
                 if j = scale - 1, m = -(width - 1)
                 """
                measure_isolate.append(i - num_data)
                i -= 1
            i = turn_off1_new + j * (width - 1) - gap * int((j + scale + 1) / 2)
            while i >= turn_off2 + lenth - p - 1 - (scale - 1) + width * j:
                measure_isolate.append(i - num_data + width)
                i -= gap
            i = turn_off1 - p + 1 + (scale - 1) - width * j
            while i <= turn_off1 - (scale - 1) + int(j / 2 * (scale - 1) - (width - 1 / 2)) * j - 1:
                measure_isolate.append(i - num_data + 1)
                i += 1
        for i in range(0, scale - 1):
            measure_isolate.append(turn_off2 + (scale - 1) + (width - 1) * (scale - 1) - num_data - (width - 1))
            measure_isolate.append(turn_off1 - (scale - 1) - (width - 1) * (scale - 1) - num_data + width)

        for i in measure_isolate:
            circuit_tmp.append("MX", i)
        record = get_record(circuit_tmp)
        phase_cut2 = 1
        # 修正 qdic，补偿为stabilizer形式
        k = 1
        j = 1 - scale
        i = turn_off2_new - j * (width - 1) + gap * int((j + scale + 1) / 2)
        """
        let m = int((j + scale + 1) / 2, 
        if j = 1 - scale, m = 1
        if j = scale - 1, m = scale
        """
        while i <= turn_off2 + lenth - p - 1 - (scale - 1) + width * j:
            record[-1][i - num_data - width] *= record[-1][num_measure + k - 1]
            phase_cut2 *= record[-1][num_measure + k - 1]
            i += gap
            k += 1
        i = turn_off2 + lenth - p - 1 - (scale - 1) + width * j
        while i >= turn_off2 + (scale - 1) - int(j / 2 * (scale - 1) - (width - 1 / 2)) * j + 1:
            """
             let m = int(j/2*(scale - 1) - (width - 1/2)), 
             if j = 1 - scale, m = -width
             if j = scale - 1, m = -(width - 1)
             """
            record[-1][i - num_data - width] *= record[-1][num_measure + k - 1]
            phase_cut2 *= record[-1][num_measure + k - 1]
            i -= 1
            k += 1
        i = turn_off1_new + j * (width - 1) - gap * int((j + scale + 1) / 2)
        while i >= turn_off1 - p + 1 + (scale - 1) - width * j:
            record[-1][i - num_data + width] *= record[-1][num_measure + k - 1]
            phase_cut2 *= record[-1][num_measure + k - 1]
            i -= gap
            k += 1
        i = turn_off1 - p + 1 + (scale - 1) - width * j
        while i <= turn_off1 - (scale - 1) + int(j / 2 * (scale - 1) - (width - 1 / 2)) * j - 1:
            record[-1][i - num_data + width] *= record[-1][num_measure + k - 1]
            phase_cut2 *= record[-1][num_measure + k - 1]
            i += 1
            k += 1

        j = scale - 1
        i = turn_off2_new - j * (width - 1) + gap * int((j + scale + 1) / 2)
        while i <= turn_off2 + lenth - p - 1 - (scale - 1) + width * j:
            record[-1][i - num_data - (width - 1)] *= record[-1][num_measure + k - 1]
            i += gap
            k += 1
        i = turn_off2 + lenth - p - 1 - (scale - 1) + width * j
        while i >= turn_off2 + (scale - 1) - int(j / 2 * (scale - 1) - (width - 1 / 2)) * j + 1:
            record[-1][i - num_data + (width - 1)] *= record[-1][num_measure + k - 1]
            i -= 1
            k += 1
        i = turn_off1_new + j * (width - 1) - gap * int((j + scale + 1) / 2)
        while i >= turn_off2 + lenth - p - 1 - (scale - 1) + width * j:
            record[-1][i - num_data + (width - 1)] *= record[-1][num_measure + k - 1]
            i -= gap
            k += 1
        i = turn_off1 - p + 1 + (scale - 1) - width * j
        while i <= turn_off1 - (scale - 1) + int(j / 2 * (scale - 1) - (width - 1 / 2)) * j - 1:
            record[-1][i - num_data - (width - 1)] *= record[-1][num_measure + k - 1]
            i += 1
            k += 1
        for i in range(0, scale - 1):
            record[-1][turn_off2 + (scale - 1) + (width - 1) * (scale - 1) - num_data - width] *= record[-1][
                num_measure + k - 1]
            record[-1][turn_off1 - (scale - 1) - (width - 1) * (scale - 1) - num_data + width] *= record[-1][
                num_measure + k - 1]
            k += 1

        record[-1] = record[-1][0: num_measure]
        circuit_tmp.clear()

        # 填充
        isolate_on = []  # 记录的是包含于cut里面的data qubit的index
        isolate0 = []  # 记录的是包含于cut里面的与cut同类型的（即与最外圈的stabilizer同类型的）measure qubit的index
        for j in range(0, scale):
            i = turn_off2_new - (scale - 1) * (width - 1) - j
            while i <= turn_off2 + lenth - p - 1 - (scale - 1) + width * (scale - 1) - j:
                isolate0.append(i)
                i += gap
            i = turn_off2 + lenth - p - 1 - (scale - 1) + width * (scale - 1) - scale - j * gap
            while i >= turn_off2 + (scale - 1) + (width - 1) * (scale - 1) + scale - j * gap:
                isolate0.append(i)
                i -= 1
            i = turn_off1_new + (scale - 1) * (width - 1) + j
            while i >= turn_off1 - p + 1 + (scale - 1) - width * (scale - 1) + j:
                isolate0.append(i)
                i -= gap
            i = turn_off1 - p + 1 + (scale - 1) - width * (scale - 1) + scale + j * gap
            while i <= turn_off1 - (scale - 1) - (width - 1) * (scale - 1) - scale + j * gap:
                isolate0.append(i)
                i += 1
        for j in list(isolate0):
            isolate_on = list(set(qdic[j - num_data][1:]) | set(isolate_on))
        for i in isolate0:
            qdic[i - num_data] = [qdic[i - num_data][0], i - num_data, i - num_data - (width - 1), i - num_data + 1,
                                  i - num_data + width]
        qdic = turn_on(isolate_on)
        circuit = runcirc(qdic)
        record = get_record(circuit)

        #   move cut(step2)----------------------------------------------------------------------------------------------------
        isolate = []  # 记录的是包含于cut里面的data qubit的下标（to be applied turn_off function）
        isolate0 = []  # 记录的是包含于cut里面的与cut同类型的（即与最外圈的stabilizer同类型的）measure qubit的index
        for j in range(0, scale):
            i = turn_off2 + (scale - 1) + width * (scale - 1) - scale - j * gap
            while i >= turn_off2 + (width - 1) * scale - j * gap:
                isolate0.append(i)
                i -= 1
            i = turn_off1 - (scale - 1) - width * (scale - 1) + scale + j * gap
            while i <= turn_off1 - (width - 1) * scale + j * gap:
                isolate0.append(i)
                i += 1

        for j in list(isolate0):
            isolate = list(set(qdic[j - num_data][1:]) | set(isolate))
        for r in isolate0:
            qdic[r - num_data] = [qdic[r - num_data][0]]
        qdic = turn_off(isolate)

        circuit = runcirc(qdic)
        circuit_tmp = circuit.copy()
        # 获取孤立比特X基测量结果

        measure_isolate = []

        for j in [0, scale - 1]:
            i = turn_off2 + (scale - 1) + width * (scale - 1) - scale - j * gap
            while i >= turn_off2 + (width - 1) * scale - j * gap:
                measure_isolate.append(i - num_data + 1)
                i += gap
            i = turn_off1 - (scale - 1) - width * (scale - 1) + scale + j * gap
            while i <= turn_off1 - (width - 1) * scale + j * gap:
                measure_isolate.append(i - num_data)
                i -= gap
        for i in range(0, scale - 1):
            measure_isolate.append(turn_off2 + (width - 1) * scale - i * gap - num_data - (width - 1))
            measure_isolate.append(turn_off1 - (width - 1) * scale + i * gap - num_data + width)

        for i in measure_isolate:
            circuit_tmp.append("MX", i)
        record = get_record(circuit_tmp)
        # 修正 qdic，补偿为stabilizer形式
        k = 1
        for j in [0, scale - 1]:
            i = turn_off2 + (scale - 1) + width * (scale - 1) - scale - j * gap
            while i >= turn_off2 + (width - 1) * scale - j * gap:
                record[-1][i - num_data - int(j * (2 * width - 1) / (scale - 1) - width)] *= record[-1][
                    num_measure + k - 1]
                """
                let m = int(j * (2*width - 1)/(scale - 1) - width), 
                if j = 0, m = - width
                if j = scale - 1, m = width - 1
                """
                i -= 1
                k += 1
            i = turn_off1 - (scale - 1) - width * (scale - 1) + scale + j * gap
            while i <= turn_off1 - (width - 1) * scale + j * gap:
                record[-1][i - num_data + int(j * (2 * width - 1) / (scale - 1) - width)] *= record[-1][
                    num_measure + k - 1]
                i += 1
                k += 1
        for i in range(0, scale - 1):
            record[-1][turn_off2 + (width - 1) * scale - i * gap - num_data - width] *= record[-1][num_measure + k - 1]
            record[-1][turn_off1 - (width - 1) * scale + i * gap - num_data + width] *= record[-1][num_measure + k - 1]
            k += 1

        circuit_tmp.clear()
        record[-1] = record[-1][0: num_measure]

        # 填充
        isolate_on = []  # 记录的是包含于cut里面的data qubit的index
        isolate0 = []  # 记录的是包含于cut里面的与cut同类型的（即与最外圈的stabilizer同类型的）measure qubit的index
        for j in range(0, scale):
            i = turn_off2 + (scale - 1) + width * (scale - 1) - j * gap
            while i >= turn_off2 + width * (scale - 1) + 1 - j * gap:
                isolate0.append(i)
                i -= 1
            i = turn_off1 - (scale - 1) - width * (scale - 1) + j * gap
            while i <= turn_off1 - width * (scale - 1) - 1 - j * gap:
                isolate0.append(i)
                i += 1
        for j in list(isolate0):
            isolate_on = list(set(qdic[j - num_data][1:]) | set(isolate_on))
        for i in isolate0:
            qdic[i - num_data] = [qdic[i - num_data][0], i - num_data, i - num_data - (width - 1), i - num_data + 1,
                                  i - num_data + width]
        qdic = turn_on(isolate_on)

        # 更新电路
        circuit = runcirc(qdic)
        record = get_record(circuit)