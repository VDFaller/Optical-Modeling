import numpy as np
def JtM(Jones):
    A = 1/np.sqrt(2)*np.mat([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])
    kron = np.kron(Jones, np.conj(Jones))
    M = A * kron * A.H
    
    return M.real

linear_h = np.mat([[1, 0], [0, 0]])
linear_v = np.mat([[0, 0], [0, 1]])
linear_p45 = .5 * np.mat([[1, 1], [1, 1]])
linear_n45 = .5 * np.mat([[1, -1], [-1, 1]])
rcp = .5 * np.mat([[1, 1j], [-1j, 1]])
lcp = .5 * np.mat([[1, -1j], [1j, 1]])

qwp_v = np.exp(1j*np.pi/4)*np.mat([[1, 0], [0, -1j]])
qwp_h = np.exp(1j*np.pi/4)*np.mat([[1, 0], [0, 1j]])