__version__ = '2.1.1'
__author__ = 'Yunheng Ma'


__all__ = ["Qcircuit",
           "save",
           "load",
          ]


from .simulators.statevector import Qstatevector
from .simulators.unitary import Qunitary
from .simulators.density_matrix import Qdensity_matrix


def Qcircuit(num_qubits, backend='statevector'):
    '''
    Create new quantum circuit instance.
    
    -In(2):
        1. num_qubits --- number of qubits.
            type: int
        2. backend --- how to execute the circuit; 'statevector', 
            'unitary', or 'density_matrix'.
            type: str
    
    -Return(1):
        1. --- quantum circuit instance.
            type: qton circuit instance.
    '''
    if backend == 'statevector':
        return Qstatevector(num_qubits)
    if backend == 'unitary':
        return Qunitary(num_qubits)
    elif backend == 'density_matrix':
        return Qdensity_matrix(num_qubits)
    else:
        raise Exception('Unrecognized backend.')


def save(var, filename=''):
    '''
    Save variable to a file.

    -In(2):
        1. var --- variable to save.
            type: any
        2. filename --- name of file to save in.
            type: str

    -Return(1):
        1. filename --- name of file to save in.
            type: str
    '''
    import pickle   
    if filename == '':
        import time
        filename = str(time.time()) + '.save'
    with open(filename, 'wb') as f:
        pickle.dump(var, f)
    return filename


def load(filename=''):
    '''
    Load variable from a file.
    
    -In(1):
        1. filename --- name of saved file.
            type: str

    -Return(1):
        1. var --- variable saved before.
            type: any
    '''
    import pickle
    if filename == '':
        raise Exception('Must specify a file to load.')
    else:
        with open(filename, 'rb') as f:
            var = pickle.load(f)
    return var