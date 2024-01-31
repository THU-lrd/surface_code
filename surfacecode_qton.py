import numpy as np
import copy
from qton import Qcircuit

# s = stim.TableauSimulator()

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


def MX(circ: Qcircuit, qdict: list, index: int):
    global measure_val
    circ.h(index)
    for i in range(1, len(qdict[index - num_data])):
        circ.cx(index, qdict[index - num_data][i])
    circ.h(index)
    measure_val.append(circ.measure(index))
    # print("X-basis 测量比特结果:", circ.measure(index))
    return circ


def MZ(circ: Qcircuit, qdict: list, index: int):
    global measure_val
    for i in range(1, len(qdict[index - num_data])):
        circ.cx(qdict[index - num_data][i], index)
    measure_val.append(circ.measure(index))
    # print("Z-basis 测量比特结果:", circ.measure(index))
    return circ


def runcirc(qdict: np.array, circ: Qcircuit):
    global num_measure
    global num_data
    global measure_val

    measure_val = []
    for i in range(0, len(qdict)):
        if qdic[i][0] == -1:
            circ = MX(circ, qdict, i + num_data)
        else:
            circ = MZ(circ, qdict, i + num_data)
    return circ


def reset(circ: Qcircuit, index: int):
    if circ.measure(index) == 1:
        circ.x(index)
    else:
        pass


def init_quiescent_state(circ: Qcircuit):
    global record
    global qdic
    global measure_val
    record = get_record(circ)
    measure_val = []
    for i in range(0, len(record[-1])):
        if record[-1][i] == -1:
            print(i)
            if qdic[i][0] == -1:
                # circ.h(i + num_data)
                for j in range(1, len(qdic[i])):
                    # circ.x(qdic[i][j])
                    circ.cx(i + num_data, qdic[i][j])
                circ.h(i + num_data)
                circ.measure(i + num_data)
                reset(circ, i + num_data)
            else:
                for j in range(1, len(qdic[i])):
                    # circ.z(qdic[i][j])
                    circ.cx(qdic[i][j], i + num_data)
                circ.measure(i + num_data)
                reset(circ, i + num_data)
        else:
            pass
    circ = runcirc(qdic, circ)
    return circ


def get_record(circ: Qcircuit):
    global measure_val
    measure_val_tmp = copy.deepcopy(measure_val)
    for i in range(0, num_measure):
        reset(circ, i + num_data)
    for k in range(0, len(measure_val_tmp)):
        measure_val_tmp[k] = -2 * (measure_val_tmp[k] - 1 / 2)
    record.append(measure_val_tmp)
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
measure_val = []

"""byproduct_err非空时，其元素为长度为 3 的列表[a, b, c]，a代表发生x or z 错误的比特位置，
b = 0 代表错误类型为x， b = 1 代表错误类型为z， c代表发生错误的时间（c代表的是该错误发生在第c次测量和第c + 1次测量之间）"""
byproduct_err = []
