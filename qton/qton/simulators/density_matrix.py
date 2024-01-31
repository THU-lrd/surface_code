import numpy as np


__all__ = ["svec2dmat",
           "random_qubit",
           "expectation",
           "operate",
           "Qdensity_matrix",
           ]


def svec2dmat(svec):
    '''
    Statevector to density matrix.
    
    $$
    \rho = |\psi\rangle \langle\psi|
    $$
    
    -In(1):
        1. svec --- quantum statevector.
            type: numpy.ndarray, 1D, complex
    
    -Return(1):
        1. --- density matrix corresponding to the statevector.
            type: numpy.ndarray, 2D, complex
    '''
    # return np.enisum('i,j->ij', svec, svec.conj())
    return np.outer(svec, svec.conj())


def random_qubit(mixed=True):
    '''
    Returns a random single-qubit density matrix. 
    
    $$
    \rho = \frac{1}{2}(I + r_x \sigma_x + r_y \sigma_y + r_z \sigma_z)
    $$

    $r_x^2 + r_y^2 + r_z^2 < 1$ for mixed state.

    -In(1):
        1. mixed --- a mixed state?
            type: bool

    -Return(1):
        1. dmat --- single-qubit density matrix.
            type: numpy.ndarray, 2D, complex
    '''
    x, y, z = np.random.standard_normal(3)
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0:
        rx = ry = rz = 0.
    else:
        rx, ry, rz = x/r, y/r, z/r

    if mixed:
        e = np.random.random()
        rx, ry, rz = rx*e, ry*e, rz*e

    dmat = np.zeros((2, 2), complex)
    dmat[0, 0] = 1. + rz
    dmat[0, 1] = rx - 1j*ry
    dmat[1, 0] = rx + 1j*ry
    dmat[1, 1] = 1. - rz
    dmat *= 0.5
    return dmat


def expectation(oper, dmat):
    '''
    Expectation of quantum operations on a density matrix.

    $$
    \langle E \rangle = \sum_k {\rm Tr}(E_k\rho)
    $$
    
    -In(2):
        1. oper --- quantum operations.
            type: list, numpy.ndarray, 2D, complex
        2. dmat --- density matrix of system.
            type: numpy.ndarray, 2D, complex

    -Return(1):
        1. ans --- expectation value.
            type: complex
    '''
    if type(oper) is not list:
        oper = [oper]

    ans = 0.
    for i in range(len(oper)):
        # ans += np.einsum('ij,ji->', oper[i], dmat)
        ans += np.trace(np.matmul(oper[i], dmat))
    return ans


def operate(oper, dmat):
    '''
    Implement a single-qubit or double-qubit quantum operation.

    $$
    \rho' = \sum_k E_k \rho E_k^\dagger
    $$

    -In(2):
        1. oper --- the quantum operations.
            type: list, numpy.ndarray, 2D, complex
        2. dmat --- density matrix.
            type: numpy.ndarray, 2D, complex
            
    -Return(1):
        1. ans --- density matrix after implementation.
            type: numpy.ndarray, 2D, complex
    '''
    if type(oper) is not list:
        oper = [oper]

    n = len(oper)
    ans = np.zeros(dmat.shape, complex)
    for i in range(n):
        # ans += np.einsum('ij,jk,lk->il', oper[i], dmat, oper[i].conj())
        ans += np.matmul(oper[i],
                         np.matmul(dmat, oper[i].transpose().conj()))
    return ans


# alphabet
alp = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
]
ALP = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
]


from ._basic_qcircuit_ import _Basic_qcircuit_
from qton.operators.channels import *
from qton.operators.superop import to_superop
from qton.operators.channels import to_channel


class Qdensity_matrix(_Basic_qcircuit_):
    '''
    Quantum circuit represented by circuit density matrix.

    -Attributes(3):
        1. backend --- how to execute the circuit; 'statevector', 
            'unitary', or 'density_matrix'.
            type: str
        2. num_qubits --- number of qubits.
            type: int
        3. state --- circuit state representation.
            type: numpy.ndarray, complex        
    
    -Methods(70):
        1. __init__(self, num_qubits=1)
        2. _apply_(self, op, *qubits)
        3. _apply1_(self, op, *qubits)
        4. _apply2_(self, op, *qubits)
        5. apply(self, op, *qubits)
        6. measure(self, qubit, delete=False)
        7. add_qubit(self, num_qubits=1)
        8. sample(self, shots=1024, output='memory')
        9. copy(self)
        10. reduce(self, qubits)
        11. bit_flip(self, p, qubits)
        12. phase_flip(self, p, qubits)
        13. bit_phase_flip(self, p, qubits)
        14. depolarize(self, p, qubits)
        15. amplitude_damping(self, gamma, qubits)
        16. generalized_amplitude_damping(self, p, gamma, qubits)
        17. phase_damping(self, lamda, qubits)
        18. i(self, qubits)
        19. x(self, qubits)
        20. y(self, qubits)
        21. z(self, qubits)
        22. h(self, qubits)
        23. s(self, qubits)
        24. t(self, qubits)
        25. sdg(self, qubits)
        26. tdg(self, qubits)
        27. rx(self, theta, qubits)
        28. ry(self, theta, qubits)
        29. rz(self, theta, qubits)
        30. p(self, phi, qubits)
        31. u1(self, lamda, qubits)
        32. u2(self, phi, lamda, qubits)
        33. u3(self, theta, phi, lamda, qubits)
        34. u(self, theta, phi, lamda, gamma, qubits)
        35. swap(self, qubit1, qubit2)
        36. cx(self, qubits1, qubits2)
        37. cy(self, qubits1, qubits2)
        38. cz(self, qubits1, qubits2)
        39. ch(self, qubits1, qubits2)
        40. cs(self, qubits1, qubits2)
        41. ct(self, qubits1, qubits2)
        42. csdg(self, qubits1, qubits2)
        43. ctdg(self, qubits1, qubits2)
        44. crx(self, theta, qubits1, qubits2)
        45. cry(self, theta, qubits1, qubits2)
        46. crz(self, theta, qubits1, qubits2)
        47. cp(self, phi, qubits1, qubits2)
        48. fsim(self, theta, phi, qubits1, qubits2)
        49. cu1(self, lamda, qubits1, qubits2)
        50. cu2(self, phi, lamda, qubits1, qubits2)
        51. cu3(self, theta, phi, lamda, qubits1, qubits2)
        52. cu(self, theta, phi, lamda, gamma, qubits1, qubits2)
        53. cswap(self, qubit1, qubit2, qubit3)
        54. ccx(self, qubits1, qubits2, qubits3)
        55. ccy(self, qubits1, qubits2, qubits3)
        56. ccz(self, qubits1, qubits2, qubits3)
        57. cch(self, qubits1, qubits2, qubits3)
        58. ccs(self, qubits1, qubits2, qubits3)
        59. cct(self, qubits1, qubits2, qubits3)
        60. ccsdg(self, qubits1, qubits2, qubits3)
        61. cctdg(self, qubits1, qubits2, qubits3)
        62. ccrx(self, theta, qubits1, qubits2, qubits3)
        63. ccry(self, theta, qubits1, qubits2, qubits3)
        64. ccrz(self, theta, qubits1, qubits2, qubits3)
        65. ccp(self, phi, qubits1, qubits2, qubits3)
        66. cfsim(self, theta, phi, qubits1, qubits2, qubits3)
        67. ccu1(self, lamda, qubits1, qubits2, qubits3)
        68. ccu2(self, phi, lamda, qubits1, qubits2, qubits3)
        69. ccu3(self, theta, phi, lamda, qubits1, qubits2, qubits3)
        70. ccu(self, theta, phi, lamda, gamma, qubits1, qubits2, qubits3)
    '''
    backend = 'density_matrix'


    def __init__(self, num_qubits=1):
        super().__init__(num_qubits)
        self.state = np.zeros((2**num_qubits, 2**num_qubits), complex)
        self.state[0, 0] = 1.0
        return None


    def _apply_(self, op, *qubits):
        super()._apply_(op, *qubits)

        threshold = 4
        # when 'op.num_qubits' is smaller than 'threshold', 'superop'
        # method is choosed; if else, 'channel' method is choosed.
        if op.num_qubits < threshold:
            self._apply1_(op, *qubits)
        else:
            self._apply2_(op, *qubits)
        return None


    def _apply1_(self, op, *qubits):
        # '_apply1_' uses 'superop' to alter the density matrix.
        # Empirically, this works well when the 'op.num_qubits' < 4
        global alp
        alp += ALP

        a_idx = [*range(2*op.num_qubits, 4*op.num_qubits)]
        b_idx = [self.num_qubits-i-1 for i in qubits] + \
            [2*self.num_qubits-i-1 for i in qubits]
        if op.category != 'superop':
            op = to_superop(op)
        rep = op.represent.reshape([2]*4*op.num_qubits)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.tensordot(rep, self.state, axes=(a_idx, b_idx))

        s = ''.join(alp[:2*self.num_qubits])
        end = s
        start = ''
        for i in range(op.num_qubits):
            start += end[self.num_qubits-qubits[i]-1]
            s = s.replace(start[i], '')
        for i in range(op.num_qubits):
            start += end[2*self.num_qubits-qubits[i]-1]
            s = s.replace(start[op.num_qubits+i], '')
        start = start + s
        self.state = np.einsum(
            start+'->'+end, self.state).reshape(2**self.num_qubits, -1)
        return None


    def _apply2_(self, op, *qubits):
        # '_apply2_' uses 'channel' to alter the density matrix.
        # Empirically, this works well when the 'op.num_qubits' >= 4
        global alp, ALP

        a_idx = [*range(op.num_qubits, 2*op.num_qubits)]
        b_idx = [self.num_qubits-i-1 for i in qubits]
        c_idx = [i+self.num_qubits for i in b_idx]
        d_idx = [*range(op.num_qubits)]
        if op.category != 'channel':
            op = to_channel(op)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        state = np.zeros(self.state.shape, complex)
        for i in range(len(op.represent)):
            rep_a = op.represent[i].reshape([2]*2*op.num_qubits)
            rep_b = op.represent[i].transpose().conj().reshape(
                [2]*2*op.num_qubits)
            tmp = np.tensordot(rep_a, self.state, axes=(a_idx, b_idx))
            tmp = np.tensordot(tmp, rep_b, axes=(c_idx, d_idx))
            state += tmp

        end1 = s1 = ''.join(alp[:self.num_qubits])
        end2 = s2 = ''.join(ALP[:self.num_qubits])
        start1 = start2 = ''
        for i in range(op.num_qubits):
            start1 += end1[self.num_qubits-qubits[i]-1]
            start2 += end2[self.num_qubits-qubits[i]-1]
            s1 = s1.replace(start1[i], '')
            s2 = s2.replace(start2[i], '')
        start = start1 + s1 + s2 + start2
        end = end1 + end2
        self.state = np.einsum(
            start+'->'+end, state).reshape(2**self.num_qubits, -1)
        return None


    def measure(self, qubit, delete=False):
        super().measure(qubit, delete)
        state = self.state.reshape([2]*2*self.num_qubits)
        dic = locals()

        string00 = ':, '*(self.num_qubits-qubit-1) + '0, ' + \
            ':, '*(self.num_qubits-1) + '0, ' + ':, '*qubit
        string11 = ':, '*(self.num_qubits-qubit-1) + '1, ' + \
            ':, '*(self.num_qubits-1) + '1, ' + ':, '*qubit

        string01 = ':, '*(self.num_qubits-qubit-1) + '0, ' + \
            ':, '*(self.num_qubits-1) + '1, ' + ':, '*qubit
        string10 = ':, '*(self.num_qubits-qubit-1) + '1, ' + \
            ':, '*(self.num_qubits-1) + '0, ' + ':, '*qubit

        exec('reduced0 = state[' + string00 + ']', dic)
        measured = dic['reduced0'].reshape(2**(self.num_qubits-1), -1)
        probability0 = np.trace(measured)

        if np.random.random() < probability0:
            bit = 0
            if delete:
                self.state = measured
                self.num_qubits -= 1
            else:
                exec('state[' + string01 + '] = 0.', dic)
                exec('state[' + string10 + '] = 0.', dic)
                exec('state[' + string11 + '] = 0.', dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= probability0
        else:
            bit = 1
            if delete:
                exec('reduced1 = state[' + string11 + ']', dic)
                self.state = dic['reduced1'].reshape(
                    2**(self.num_qubits-1), -1)
                self.num_qubits -= 1
            else:
                exec('state[' + string00 + '] = 0.', dic)
                exec('state[' + string01 + '] = 0.', dic)
                exec('state[' + string10 + '] = 0.', dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= (1. - probability0)
        return bit


    def reduce(self, qubits):
        '''
        Recuced density matrix after partial trace over given qubits.

        Circuit will remove the given qubits.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
                
        -Influenced(2):
            1. self.state --- qubit density matrix.
                type: numpy.ndarray, 2D, complex
            2. self.num_qubit --- number of qubits.
                type: int
        '''
        if type(qubits) is int:
            q = [qubits]
        else:
            q = list(qubits)

        if max(q) >= self.num_qubits:
            raise Exception('Qubit index oversteps.')

        if len(q) > len(set(q)):
            raise Exception('Duplicate qubits in input.')

        if self.num_qubits - len(q) < 1:
            raise Exception('Must keep one qubit at least.')

        global alp, ALP

        s = (alp+ALP)[:2*self.num_qubits]
        for i in q:
            s[2*self.num_qubits-i-1] = s[self.num_qubits-i-1]
        start = ''.join(s)

        for i in q:
            s[self.num_qubits-i-1] = ''
            s[2*self.num_qubits-i-1] = ''
        end = ''.join(s)

        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.einsum(
            start+'->'+end, self.state).reshape([2**(self.num_qubits-len(q))]*2)
        self.num_qubits -= len(q)
        return None


#
# Kraus channel methods.
#


    def bit_flip(self, p, qubits):
        '''
        Bit flip channel.

        -In(2):
            1. p --- the probability for not flipping.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
                
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Bit_flip([p]), qubits)
        return None

    def phase_flip(self, p, qubits):
        '''
        Phase flip channel.
        
        -In(2):
            1. p --- the probability for not flipping.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Phase_flip([p]), qubits)
        return None

    def bit_phase_flip(self, p, qubits):
        '''
        Bit phase flip channel.
        
        -In(2):
            1. p --- the probability for not flipping.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Bit_phase_flip([p]), qubits)
        return None

    def depolarize(self, p, qubits):
        '''
        Depolarizing channel.
    
        -In(2):
            1. p --- the probability for depolarization.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Depolarize([p]), qubits)
        return None

    def amplitude_damping(self, gamma, qubits):
        '''
        Amplitude damping channel.
        
        -In(2):
            1. gamma --- probability such as losing a photon.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Amplitude_damping([gamma]), qubits)
        return None

    def generalized_amplitude_damping(self, p, gamma, qubits):
        '''
        Generalized amplitude damping channel.
        
        -In(3):
            1. p --- the probability for acting normal amplitude damping.
                type: float
            2. gamma --- probability such as losing a photon.
                type: float
            3. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Generalized_amplitude_damping([p, gamma]), qubits)
        return None

    def phase_damping(self, lamda, qubits):
        '''
        Phase damping channel.
        
        -In(2):
            1. lamda --- probability such as a photon from the system has been 
                scattered(without loss of energy).
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Phase_damping([lamda]), qubits)
        return None
