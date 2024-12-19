from scipy.sparse import csc_matrix
from sksparse.cholmod import cholesky
import os

# Check if the environment variable is set
cholmod_gpu = os.getenv('CHOLMOD_USE_GPU')

# Output the value to verify
if cholmod_gpu == '1':
    print("CHOLMOD GPU acceleration is enabled.")
else:
    print("CHOLMOD GPU acceleration is not enabled.")

def sparse_sqrt(A):
    # Given positive semi definite square A, find a square R
    # such that R.T @ R = A
    decomp = cholesky(csc_matrix(A), beta=1e-12,
        ordering_method='natural')
    L,D = decomp.L_D()
    D.data[D.data<0.] = 0.
    return D.sqrt() @ L.T

#If we want this, the size cannot be too big because it's O(n**3). So I finally gave up doing this on GPU
# import cupy as cp
#
#
# def sparse_sqrt(A):
#
#     A_dense = A.toarray() if hasattr(A, 'toarray') else A
#     A_gpu = cp.asarray(A_dense)
#
#     # A = R.T @ R
#     R = cp.linalg.cholesky(A_gpu)
#     return R.T
