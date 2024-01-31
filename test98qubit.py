import surfacecode as sc
# import stim
# init(97, 98, 7, 8)
# init(7, 8, 3, 2)
sc.init(97, 98, 7, 8)
_ = sc.get_qdic(True)
circuit = sc.runcirc(sc.qdic)
# print()
_= sc.get_record(circuit)
print('第1次')
_= sc.get_record(circuit)
print('第2次')
print(sc.record[-1] == sc.record[-2])
_= sc.get_record(circuit)
print('第3次')
print(sc.record[-1] == sc.record[-2])
_= sc.get_record(circuit)
print('第4次')
print(sc.record[-1] == sc.record[-2])
_ = sc.get_record(circuit)
print('第5次')
print(sc.record[-1] == sc.record[-2])
_= sc.get_record(circuit)
print('第6次')
print(sc.record[-1] == sc.record[-2])
# print(circuit.diagram())
#  测试cut函数
# circ, qdic1, _ = sc.cut(120, 146, 1,False)
# print(sc.record[-1] == sc.record[-2])
#  测试cut_init函数
circ2, _, _ = sc.cut_init(1, 120, 146, False, False)
print('initial')
print(sc.record[-1] == sc.record[-2])
_= sc.get_record(circ2)
print('(after initial) stabilize the circuit(the result is not important,T anf F are both possible)')
print(sc.record[-1] == sc.record[-2])
# print(sc.record[-1] == sc.record[-2])
#  测试apply_cut，理论上该函数作用之后电路测量结果与作用前的测量结果相同
xl, zl = sc.apply_cut(1, 120, 146, False)
circuit1 = xl(circ2)
sc.get_record(circuit1)
print(sc.record[-1] == sc.record[-2])
circuit2 = zl(circuit1)
print(sc.record[-1] == sc.record[-2])
sc.get_record(circuit2)
print(sc.record[-1] == sc.record[-2])
# 测试cut_measure
# _, qdic2, _ = sc.cut_measure(1, 120, 146, True, False)
circuit, _ = sc.qmove(1, 120, 146, 172, True)
xl, zl = sc.apply_cut(1, 120, 172, False)
circuit1 = xl(circuit)
sc.get_record(circuit1)
print('xl after qmove')  # if the measurement result after applying logical operator after qmove is unchanged, my code is no-error
print(sc.record[-1] == sc.record[-2])
sc.get_record(circuit1)
print(sc.record[-1] == sc.record[-2])
sc.get_record(circuit1)
print(sc.record[-1] == sc.record[-2])
circuit2 = zl(circuit1)
print('zl after qmove')
print(sc.record[-1] == sc.record[-2])
sc.get_record(circuit2)
print(sc.record[-1] == sc.record[-2])
# print(sc.record)

# # 比较每相邻两次测量结果是否相同
# for i in range(0, len(sc.record) - 1):
#     print(sc.record[i] == sc.record[i+1])

# for i in sc.record:
#     print(len(i))









# get_record(circuit)
# l1 = copy.deepcopy(s.state_vector(endian='big'))
# circuit = runcirc(circuit, qdic)
# get_record(circuit)
# l2 = copy.deepcopy(s.state_vector(endian='big'))
# print(list(l2) == list(l1))

# ""
# circuit = runcirc(circuit, qdic)
# get_record(circuit)
# print(record[-1] == record[-2])
# print(qdic)
# l3 = copy.deepcopy(s.state_vector(endian='big'))
# print(list(l2) == list(l3))
# if (l3 == l2).all():
#     print('l3 = l2')
#     print(l3 == l2)
# else:
#     print('不相等')
#     print(l3 == l2)
# # print(s.state_vector(endian='big') == l2)
# ""
# circuit = runcirc(circuit, qdic)
# get_record(circuit)
# print(record[-1] == record[-2])
# print(qdic)
# # print(circuit.diagram())
# _ = runcirc(circuit, qdic)
# get_record(circuit)
# print(record[-1] == record[-2])
# print(qdic)
# # print(circuit.diagram())
# _ = runcirc(circuit, qdic)
# get_record(circuit)
# print(record[-1] == record[-2])
# print(qdic)
# _ = runcirc(circuit, qdic)
# get_record(circuit)
# print(record[-1] == record[-2])
# print(qdic)
# _ = runcirc(circuit, qdic)
# get_record(circuit)
# print(record[-1] == record[-2])
# print(qdic)
# print(circuit.diagram())
# circuit.diagram("detslice-svg")
# plt.show()
# print(list(s.canonical_stabilizers()))
# print(list(s.state_vector(endian='big')))