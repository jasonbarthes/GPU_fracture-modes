# import igl
# from scipy.sparse import csc_matrix
#
# def massmatrix_tets(V,T):
#     # Libigl python binding of massmatrix does not work for tet meshes, so
#     # this is a quick wrapper doing a lumped tet mass matrix
#     vol = (igl.volume(V, T)/4.0).repeat(4)
#     i = T.flatten()
#     j = i
#
#     M = csc_matrix((vol, (i, j)), (V.shape[0], V.shape[0]))
#     return M

import cupy as cp
from cupyx.scipy.sparse import coo_matrix

def massmatrix_tets(V, T):
    V_gpu = cp.asarray(V)
    T_gpu = cp.asarray(T)

    v0 = V_gpu[T_gpu[:, 0]]
    v1 = V_gpu[T_gpu[:, 1]]
    v2 = V_gpu[T_gpu[:, 2]]
    v3 = V_gpu[T_gpu[:, 3]]

    vol = cp.abs(cp.einsum('ij,ij->i', cp.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0

    vol = (vol / 4.0).repeat(4)

    i = T_gpu.flatten()
    j = i

    # vol = cp.asarray(vol, dtype=cp.float32)
    # i = cp.asarray(i, dtype=cp.int32)
    # j = cp.asarray(j, dtype=cp.int32)
    #
    # assert i.min() >= 0 and i.max() < V.shape[0]
    # assert j.min() >= 0 and j.max() < V.shape[0]

    M_coo = coo_matrix((vol, (i, j)), shape=(V.shape[0], V.shape[0]))

    #Convert to CSC
    M = M_coo.tocsc()

    return M