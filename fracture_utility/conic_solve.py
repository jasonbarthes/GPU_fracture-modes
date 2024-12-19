# # Include existing libraries
# import numpy as np
# import sys
#
# # Mosek for the conic solve
# import mosek
#
#
# def conic_solve(D, M, Us, c, d, verbose=False):
#     # This uses Mosek to solve the conic problem
#     #           argmin     ||Du||_{2,1}
#     #           s.t.       u' M Us = 0
#     #           and        u' M c = 1
#     #
#     # Unfortunately MOSEK's conic API is quite complicated so we actually have
#     # to write it as
#     #          argmin    sum (z_e)                <--- linear
#     #           s.t.     ze >= sqrt(sum(Yd^2))    <--- cone (linear if d=1)
#     #           and      Y = Du                   <--- linear
#     #           and      u' M Us = 0              <--- linear
#     #           and      u' M c = 1               <--- linear
#
#
#     # From Mosek template, apparently we have to add this
#     def streamprinter(text):
#         sys.stdout.write(text)
#         sys.stdout.flush()
#
#     # Mosek boilerplate
#     with mosek.Env() as env:
#         if verbose:
#             env.set_Stream(mosek.streamtype.log, streamprinter)
#         with env.Task(0,0) as task:
#             if verbose:
#                 task.set_Stream(mosek.streamtype.log, streamprinter)
#
#             # Dimensions and degrees of freedom
#             p = D.shape[0] // d
#             n = D.shape[1]
#             ndofs = n+ p*d+p
#             task.appendvars(ndofs)
#             task.putvarboundlistconst([*range(ndofs)], mosek.boundkey.fr, 0., 0.)
#             task.appendcons(p*d+n+1+len(Us))
#
#             # Objective function
#             task.putclist([*range(   n+ p*d,    n+ p*d+p)],
#                     [1.] * (p))
#
#
#
#             #Set up equality constraint Y = Du
#             nrows = 0
#             #D
#             task.putaijlist(nrows+D.row, D.col, D.data)
#             #-Y
#             task.putaijlist([*range(nrows, nrows+ p*d)],
#                 [*range(   n,    n+ p*d)], [-1.]*( p*d))
#             task.putconboundlistconst([*range(nrows,nrows+ p*d)],
#                 mosek.boundkey.fx, 0., 0.)
#             nrows +=  p*d
#
#
#
#             # Set up orthogonality constraint wrt Us
#             for U in Us:
#                 UtM = M*U
#                 task.putaijlist([nrows]*len(UtM), [*range(  0,    n)], UtM)
#                 task.putconbound(nrows, mosek.boundkey.fx, 0., 0.)
#                 nrows += 1
#             # Set up norm-1 constraint wrt c
#             ctM = M*c
#             task.putaijlist([nrows]*len(ctM), [*range(  0,    n)], ctM)
#             task.putconbound(nrows, mosek.boundkey.fx, 1., 1.)
#             nrows += 1
#
#
#             # Set up cone ze >= sqrt(Yes^2)
#             for e in range(p):
#                 coneinds = [   n+ p*d+e]
#                 for dim in range(d):
#                     coneinds.extend([   n+ p*dim+e])
#                 task.appendcone(mosek.conetype.quad, 0., coneinds)
#
#             # Solve
#             task.putobjsense(mosek.objsense.minimize)
#             task.optimize()
#             xx = [0.] * ndofs
#             task.getxx(mosek.soltype.itr, xx)
#
#             return np.asarray(xx)[0:n] # Extract just the u part from the solution

import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import csc_matrix
import numpy as np
import mosek
import sys

def conic_solve(D, M, Us, c, d, verbose=False):
    # Convert inputs to GPU-compatible formats
    D_gpu = cp_csr_matrix(D)  # Assume D is a sparse matrix
    M_gpu = cp_csr_matrix(M)  # Assume M is a sparse matrix
    Us_gpu = [cp.asarray(U) for U in Us]  # Assume Us is a list of dense matrices
    c_gpu = cp.asarray(c)  # Assume c is a dense vector
    d_gpu = d  # Scalar does not need conversion

    # Mosek template, must be executed on the CPU
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    # Transferred computation logic
    with mosek.Env() as env:
        if verbose:
            env.set_Stream(mosek.streamtype.log, streamprinter)
        with env.Task(0, 0) as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, streamprinter)

            # Get dimensions and degrees of freedom
            p = D_gpu.shape[0] // d_gpu
            n = D_gpu.shape[1]
            ndofs = n + p * d_gpu + p
            task.appendvars(ndofs)
            task.putvarboundlistconst([*range(ndofs)], mosek.boundkey.fr, 0.0, 0.0)
            task.appendcons(p * d_gpu + n + 1 + len(Us_gpu))

            # Set objective function
            task.putclist([*range(n + p * d_gpu, n + p * d_gpu + p)],
                          [1.0] * p)

            # Set constraints: Y = D * u
            nrows = 0
            # Convert D_gpu to COO format
            D_coo = D_gpu.tocoo()

            # Use coo_matrix properties to get rows, columns, and data
            task.putaijlist(D_coo.row.get(), D_coo.col.get(), D_coo.data.get())
            task.putaijlist([*range(nrows, nrows + p * d_gpu)],
                            [*range(n, n + p * d_gpu)], [-1.0] * (p * d_gpu))
            task.putconboundlistconst([*range(nrows, nrows + p * d_gpu)],
                                      mosek.boundkey.fx, 0.0, 0.0)
            nrows += p * d_gpu

            # Set orthogonality constraints
            for U_gpu in Us_gpu:
                UtM_gpu = M_gpu.dot(U_gpu)
                task.putaijlist([nrows] * len(UtM_gpu), [*range(0, n)], UtM_gpu.get())
                task.putconbound(nrows, mosek.boundkey.fx, 0.0, 0.0)
                nrows += 1

            # Set norm-1 constraint

            # Ensure c_gpu_dense is a dense array
            c_gpu_dense = cp.asarray(c_gpu)

            # Use the sparse matrix .dot method
            ctM_gpu = M_gpu.dot(c_gpu_dense)

            task.putaijlist([nrows] * len(ctM_gpu), [*range(0, n)], ctM_gpu.get())
            task.putconbound(nrows, mosek.boundkey.fx, 1.0, 1.0)
            nrows += 1

            # Set quadratic cone constraint ze >= sqrt(Yes^2)
            for e in range(p):
                coneinds = [n + p * d_gpu + e]
                for dim in range(d_gpu):
                    coneinds.extend([n + p * dim + e])
                task.appendcone(mosek.conetype.quad, 0.0, coneinds)

            # Solve the problem
            task.putobjsense(mosek.objsense.minimize)
            task.optimize()

            xx = [0.0] * ndofs
            task.getxx(mosek.soltype.itr, xx)

            # Return GPU data to CPU
            return np.asarray(xx)[0:n]
