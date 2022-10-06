import numpy as np

Z_VECTOR_P = np.array([[1], [0], [0]], dtype=complex)
Z_VECTOR_N = np.array([[0], [1], [0]], dtype=complex)
X_VECTOR_P = np.array(
    [[1.0/np.sqrt(2.0)], [1.0/np.sqrt(2.0)], [0]], dtype=complex)
X_VECTOR_N = np.array(
    [[1.0/np.sqrt(2.0)], [-1.0/np.sqrt(2.0)], [0]], dtype=complex)
Y_VECTOR_P = np.array([[1.0/np.sqrt(2.0)], [complex(0, 1) /
                                            np.sqrt(2.0)], [0]], dtype=complex)
Y_VECTOR_N = np.array([[1.0/np.sqrt(2.0)], [-complex(0, 1) /
                                            np.sqrt(2.0)], [0]], dtype=complex)
Y_MATRIX_PI_2 = np.array([[1.0/np.sqrt(2.0), -1.0/np.sqrt(2.0), 0],
                          [1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0), 0], [0, 0, 1]], dtype=complex)
Y_MATRIX_PI_2 = np.array([[1.0/np.sqrt(2.0), -1.0/np.sqrt(2.0), 0],
                          [1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0), 0], [0, 0, 1]], dtype=complex)


def USFQgenerator(deltatheta):
    USFQ = np.zeros([3, 3], dtype=complex)
    USFQ[0][0] = 2.0/3.0+1.0/3.0*np.cos(np.sqrt(3.0)/2.0*deltatheta)
    USFQ[0][1] = -1.0/(np.sqrt(3.0))*np.sin(np.sqrt(3.0)/2.0*deltatheta)
    USFQ[0][2] = 2.0/3.0*np.sqrt(2.0)*np.sin(np.sqrt(3.0) /
                                             4.0*deltatheta)*np.sin(np.sqrt(3.0)/4.0*deltatheta)
    USFQ[1][0] = 1.0/(np.sqrt(3.0))*np.sin(np.sqrt(3.0)/2.0*deltatheta)
    USFQ[1][1] = np.cos(np.sqrt(3.0)/2.0*deltatheta)
    USFQ[1][2] = -np.sqrt(2.0/3.0)*np.sin(np.sqrt(3.0)/2.0*deltatheta)
    USFQ[2][0] = 2.0/3.0*np.sqrt(2.0)*np.sin(np.sqrt(3.0) /
                                             4.0*deltatheta)*np.sin(np.sqrt(3.0)/4.0*deltatheta)
    USFQ[2][1] = np.sqrt(2.0/3.0)*np.sin(np.sqrt(3.0)/2.0*deltatheta)
    USFQ[2][2] = 1.0/3.0+2.0/3.0*np.cos(np.sqrt(3.0)/2.0*deltatheta)
    return USFQ


def UFRgenerator(omegaq, t, alpha):
    UFR = np.zeros([3, 3], dtype=complex)
    UFR[0][0] = 1.0
    UFR[1][1] = np.exp(-complex(0, 1)*omegaq*t)
    UFR[2][2] = np.exp(-complex(0, 1)*omegaq*t*(2-alpha))
    return UFR


def UGgenerator(USFQ, UFR, Subsequence, times):
    UG = np.eye(3, dtype=complex)
    for i in range(len(Subsequence)):
        if Subsequence[i] == 0:
            UG = np.matmul(UFR, UG)
        else:
            UG = np.matmul(USFQ, UG)
            UG = np.matmul(UFR, UG)
    UG_temp = UG
    for i in range(times-1):
        UG = np.matmul(UG_temp, UG)
    return UG


def Fedelity(UG, matrix):
    F = 0
    F += abs(np.matmul(Z_VECTOR_P.transpose(), np.matmul(
        matrix.transpose(), np.matmul(UG, Z_VECTOR_P))))*abs(np.matmul(Z_VECTOR_P.transpose(), np.matmul(
            matrix.transpose(), np.matmul(UG, Z_VECTOR_P))))

    F += abs(np.matmul(Z_VECTOR_N.transpose(), np.matmul(
        matrix.transpose(), np.matmul(UG, Z_VECTOR_N))))*abs(np.matmul(Z_VECTOR_N.transpose(), np.matmul(
            matrix.transpose(), np.matmul(UG, Z_VECTOR_N))))

    F += abs(np.matmul(X_VECTOR_P.transpose(), np.matmul(
        matrix.transpose(), np.matmul(UG, X_VECTOR_P))))*abs(np.matmul(X_VECTOR_P.transpose(), np.matmul(
            matrix.transpose(), np.matmul(UG, X_VECTOR_P))))

    F += abs(np.matmul(X_VECTOR_N.transpose(), np.matmul(
        matrix.transpose(), np.matmul(UG, X_VECTOR_N))))*abs(np.matmul(X_VECTOR_N.transpose(), np.matmul(
            matrix.transpose(), np.matmul(UG, X_VECTOR_N))))

    F += abs(np.matmul(Y_VECTOR_N.transpose(), np.matmul(
        matrix.transpose(), np.matmul(UG, Y_VECTOR_P))))*abs(np.matmul(Y_VECTOR_N.transpose(), np.matmul(
            matrix.transpose(), np.matmul(UG, Y_VECTOR_P))))

    F += abs(np.matmul(Y_VECTOR_P.transpose(), np.matmul(
        matrix.transpose(), np.matmul(UG, Y_VECTOR_N))))*abs(np.matmul(Y_VECTOR_P.transpose(), np.matmul(
            matrix.transpose(), np.matmul(UG, Y_VECTOR_N))))
    F = float(F/6.0)
    return F


def Y_deltatheta(deltatheta):
    return np.array([[np.cos(deltatheta/2), -np.sin(deltatheta/2), 0],
                     [np.sin(deltatheta/2), np.cos(deltatheta/2), 0], [0, 0, 1]], dtype=complex)
