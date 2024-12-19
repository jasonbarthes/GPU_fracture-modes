# # Include existing libraries
# import numpy as np
# from scipy.sparse import coo_matrix, csr_matrix, kron, eye
# from scipy.sparse.linalg import eigsh
# from scipy.sparse.csgraph import connected_components
# # Libigl
# import igl
#
#
# # Local includes
# from . explode_mesh import explode_mesh
# from . conic_solve import conic_solve
# from . massmatrix_tets import massmatrix_tets
# from . sparse_sqrt import sparse_sqrt
# from . tictoc import tic, toc
#
#
# # @profile
# def compute_fracture_modes(vertices,elements,parameters):
#     # Takes as input an (unexploded) tetrahedral mesh and a number of modes, returns a matrix UU dim x #T by #parameters.num_modes with computed fracture modes.
#     if parameters.verbose:
#         print("Starting fracture mode computation")
#         print("We will find ",parameters.num_modes," unique fracture modes")
#         print("Our input (unexploded) mesh has ",vertices.shape[0]," vertices and ",elements.shape[0]," tetrahedra.")
#         tic()
#
#     # Step 1: Compute traditional Laplacian eigenmodes.
#     blockdiag_kron = eye(parameters.d)
#     laplacian_unexploded = igl.cotmatrix(vertices,elements)
#
#     dense_matrix_cpu = massmatrix_tets(vertices,elements).get()  # 将 CuPy 数据转回 CPU
#     massmatrix_unexploded = kron(blockdiag_kron, dense_matrix_cpu, format='csc')
#
#     # massmatrix_unexploded = kron(blockdiag_kron,massmatrix_tets(vertices,elements),format='csc')
#     Q_unexploded = kron(blockdiag_kron,laplacian_unexploded,format='csc')
#     # print(-Q_unexploded)
#     # print(massmatrix_unexploded)
#     # unexploded_Q_eigenvalues, unexploded_Q_eigenmodes = eigsh(-Q_unexploded,parameters.num_modes,massmatrix_unexploded,which='SM') # <- our bottleneck outside of conic solves
#     unexploded_Q_eigenvalues, unexploded_Q_eigenmodes = eigsh(-Q_unexploded,parameters.num_modes,massmatrix_unexploded,which='LM', sigma=0)
#     unexploded_Q_eigenmodes = np.real(unexploded_Q_eigenmodes)
#
#     # Step 2: Explode mesh, get unexploded-to-exploded matrix, get discontinuity and exploded Laplacian matrices
#     exploded_vertices, exploded_elements, discontinuity_matrix, unexploded_to_exploded_matrix, tet_to_vertex_matrix, tet_neighbors = explode_mesh(vertices,elements,num_quad=1)
#     discontinuity_matrix_full = kron(parameters.omega*blockdiag_kron,discontinuity_matrix,format='coo')
#     unexploded_to_exploded_matrix_full = kron(blockdiag_kron,unexploded_to_exploded_matrix,format='csc')
#     tet_to_vertex_matrix_full = kron(blockdiag_kron,tet_to_vertex_matrix,format='csc')
#     laplacian_exploded = igl.cotmatrix(exploded_vertices,exploded_elements)
#     massmatrix_exploded = massmatrix_tets(exploded_vertices,exploded_elements).get()
#     Q = kron(blockdiag_kron,laplacian_exploded,format='csc')
#     R = coo_matrix(sparse_sqrt(-Q))
#     M = kron(blockdiag_kron,massmatrix_exploded,format='csc')
#
#
#
#     # Step 3: Solve iteratively to find all modes
#
#     # Initialization
#     UU = unexploded_to_exploded_matrix_full @ unexploded_Q_eigenmodes
#
#
#     # We convert everything into per-tet quantities
#     UU = tet_to_vertex_matrix_full.T @ UU
#     Q = coo_matrix(tet_to_vertex_matrix_full.T @ Q @ tet_to_vertex_matrix_full)
#     #we need .get for gpu_sqrt
#     R = coo_matrix(sparse_sqrt(-Q))
#     M = coo_matrix(tet_to_vertex_matrix_full.T @ M @ tet_to_vertex_matrix_full)
#     discontinuity_matrix_full = coo_matrix(discontinuity_matrix_full @ tet_to_vertex_matrix_full)
#
#
#
#     if parameters.verbose:
#         t_before_modes = toc(silence=True)
#         print("Building matrices before starting mode computation: ", round(t_before_modes,2)," seconds.")
#
#
#
#     # "Outer" loop to find all modes
#     Us = []
#     ts = []
#     labels_full = np.zeros((elements.shape[0],parameters.num_modes))
#     for k in range(parameters.num_modes):
#         if parameters.verbose:
#             tic()
#         iter_num = 0
#         diff = 1.0
#         c = UU[:,k] # initialize to exploded laplacian mode
#         # "Inner" loop to find each mode
#         while diff>parameters.tol and iter_num<parameters.max_iter:
#             cprev = c
#             # Solve conic problem
#             Ui = conic_solve(discontinuity_matrix_full, M, Us, c, parameters.d)
#             c = Ui / np.sqrt(np.dot(Ui, M @ Ui))
#             diff = np.max(np.abs(c-cprev))
#             iter_num = iter_num + 1
#         # Now, identify pieces:
#         tet_tet_distances = np.linalg.norm(np.reshape(c,(-1,parameters.d),order='F')[tet_neighbors[:,0],:] - np.reshape(c,(-1,parameters.d),order='F')[tet_neighbors[:,1],:],axis=1)
#         actual_neighbors = tet_neighbors[(tet_tet_distances<0.1),:]
#
#         tet_adjacency_matrix = csr_matrix((np.ones(actual_neighbors.shape[0]),(actual_neighbors[:,0],actual_neighbors[:,1])),shape=(exploded_elements.shape[0],exploded_elements.shape[0]),dtype=int)
#         n_components,labels = connected_components(tet_adjacency_matrix)
#         labels_full[:,k] = labels
#         Us.append(c)
#         UU[:,k] = c
#         if parameters.verbose:
#             t_mode = toc(silence=True)
#             ts.append(t_mode)
#             print("Computed unique mode number", k+1, "using", iter_num, "iterations and", round(t_mode, 3), "seconds. This mode breaks the shape into",n_components,"pieces.")
#     if parameters.verbose:
#         print("Average time per mode: ", round(sum(ts)/len(ts),3)," seconds")
#
#     modes = UU
#
#     # Placeholder return
#     return exploded_vertices,exploded_elements,modes,labels_full,tet_to_vertex_matrix,tet_neighbors,M,unexploded_to_exploded_matrix

import numpy as np
import cupy as cp
from cupyx.scipy.sparse import coo_matrix, csc_matrix as cp_csc_matrix, csr_matrix as cp_csr_matrix
from scipy.sparse import coo_matrix as cpu_coo_matrix, csc_matrix as cpu_csc_matrix, csr_matrix as cpu_csr_matrix, kron, eye
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
import igl

from .explode_mesh import explode_mesh
from .conic_solve import conic_solve
from .massmatrix_tets import massmatrix_tets
from .sparse_sqrt import sparse_sqrt
from .tictoc import tic, toc
from concurrent.futures import ProcessPoolExecutor


def compute_fracture_modes(vertices, elements, parameters):
    if parameters.verbose:
        print("Starting fracture mode computation")
        print("We will find ", parameters.num_modes, " unique fracture modes")
        print("Our input (unexploded) mesh has ", vertices.shape[0], " vertices and ", elements.shape[0], " tetrahedra.")
        tic()

    # Step 1: Compute Laplacian eigenmodes (CPU)
    # igl.cotmatrix is a CPU function
    laplacian_unexploded = igl.cotmatrix(vertices, elements)  # CPU scipy.sparse matrix

    # massmatrix_tets returns GPU csc_matrix
    massmatrix_gpu = massmatrix_tets(vertices, elements)  # GPU csc_matrix

    # eigsh runs on the CPU, so convert massmatrix back to CPU
    massmatrix_cpu = cpu_csc_matrix(
        (massmatrix_gpu.data.get(), massmatrix_gpu.indices.get(), massmatrix_gpu.indptr.get()),
        shape=massmatrix_gpu.shape)

    blockdiag_kron = eye(parameters.d)
    massmatrix_unexploded = kron(blockdiag_kron, massmatrix_cpu, format='csc')
    Q_unexploded = kron(blockdiag_kron, laplacian_unexploded, format='csc')

    # Perform eigendecomposition on the CPU
    unexploded_Q_eigenvalues, unexploded_Q_eigenmodes = eigsh(-Q_unexploded, parameters.num_modes,
                                                              massmatrix_unexploded, which='LM', sigma=0)
    unexploded_Q_eigenmodes = np.real(unexploded_Q_eigenmodes)

    # Step 2: Explode mesh (modify explode_mesh to return GPU data directly)
    (exploded_vertices_gpu, exploded_elements_gpu,
     discontinuity_matrix_gpu, unexploded_to_exploded_matrix_gpu,
     tet_to_vertex_matrix_gpu, tet_neighbors_gpu) = explode_mesh(vertices, elements, num_quad=1, return_gpu=True)

    # Now we have GPU data for discontinuity_matrix, unexploded_to_exploded_matrix, and tet_to_vertex_matrix
    # However, laplacian_exploded and massmatrix_exploded still require CPU igl operations

    exploded_vertices = exploded_vertices_gpu.get()  # laplacian_exploded requires CPU igl.cotmatrix
    exploded_elements = exploded_elements_gpu.get()
    laplacian_exploded = igl.cotmatrix(exploded_vertices, exploded_elements)  # CPU
    Q = kron(blockdiag_kron, laplacian_exploded, format='csc')

    # Obtain massmatrix_exploded from GPU massmatrix_tets
    massmatrix_exploded_gpu = massmatrix_tets(exploded_vertices, exploded_elements)  # GPU csc_matrix
    massmatrix_exploded_cpu = cpu_csc_matrix((massmatrix_exploded_gpu.data.get(), massmatrix_exploded_gpu.indices.get(),
                                              massmatrix_exploded_gpu.indptr.get()),
                                             shape=massmatrix_exploded_gpu.shape)
    M = kron(blockdiag_kron, massmatrix_exploded_cpu, format='csc')

    # Perform kron on the CPU for discontinuity_matrix_full due to GPU limitations
    discontinuity_matrix_cpu = cpu_csr_matrix((discontinuity_matrix_gpu.data.get(),
                                               discontinuity_matrix_gpu.indices.get(),
                                               discontinuity_matrix_gpu.indptr.get()),
                                              shape=discontinuity_matrix_gpu.shape)
    discontinuity_matrix_full = kron(parameters.omega * blockdiag_kron, discontinuity_matrix_cpu, format='coo')

    unexploded_to_exploded_matrix_cpu = cpu_csr_matrix((unexploded_to_exploded_matrix_gpu.data.get(),
                                                        unexploded_to_exploded_matrix_gpu.indices.get(),
                                                        unexploded_to_exploded_matrix_gpu.indptr.get()),
                                                       shape=unexploded_to_exploded_matrix_gpu.shape)
    unexploded_to_exploded_matrix_full = kron(blockdiag_kron, unexploded_to_exploded_matrix_cpu, format='csc')

    tet_to_vertex_matrix_cpu = cpu_csr_matrix((tet_to_vertex_matrix_gpu.data.get(),
                                               tet_to_vertex_matrix_gpu.indices.get(),
                                               tet_to_vertex_matrix_gpu.indptr.get()),
                                              shape=tet_to_vertex_matrix_gpu.shape)
    tet_to_vertex_matrix_full = kron(blockdiag_kron, tet_to_vertex_matrix_cpu, format='csc')

    # Compute R and subsequent operations on the CPU
    R = cpu_coo_matrix(sparse_sqrt(-Q))

    # Step 3: Solve iteratively on the CPU
    UU = unexploded_to_exploded_matrix_full @ unexploded_Q_eigenmodes
    UU = tet_to_vertex_matrix_full.T @ UU
    Q = cpu_coo_matrix(tet_to_vertex_matrix_full.T @ Q @ tet_to_vertex_matrix_full)
    R = cpu_coo_matrix(sparse_sqrt(-Q))
    M = cpu_coo_matrix(tet_to_vertex_matrix_full.T @ M @ tet_to_vertex_matrix_full)
    discontinuity_matrix_full = cpu_coo_matrix(discontinuity_matrix_full @ tet_to_vertex_matrix_full)

    if parameters.verbose:
        t_before_modes = toc(silence=True)
        print("Building matrices before starting mode computation: ", round(t_before_modes, 2), " seconds.")

    Us = []
    ts = []
    labels_full = np.zeros((elements.shape[0], parameters.num_modes))
    tet_neighbors = tet_neighbors_gpu.get()  # connected_components requires CPU data
    for k in range(parameters.num_modes):
        if parameters.verbose:
            tic()
        iter_num = 0
        diff = 1.0
        c = UU[:, k]

        # Use conic_solve in the CPU loop
        while diff > parameters.tol and iter_num < parameters.max_iter:
            cprev = c
            Ui = conic_solve(discontinuity_matrix_full, M, Us, c, parameters.d)  # CPU operation
            c = Ui / np.sqrt(np.dot(Ui, M @ Ui))
            diff = np.max(np.abs(c - cprev))
            iter_num += 1

        # Identify pieces on CPU
        tet_tet_distances = np.linalg.norm(np.reshape(c, (-1, parameters.d), order='F')[tet_neighbors[:, 0], :] -
                                           np.reshape(c, (-1, parameters.d), order='F')[tet_neighbors[:, 1], :], axis=1)
        actual_neighbors = tet_neighbors[tet_tet_distances < 0.1, :]

        tet_adjacency_matrix = cpu_csr_matrix((np.ones(actual_neighbors.shape[0]),
                                               (actual_neighbors[:, 0], actual_neighbors[:, 1])),
                                              shape=(exploded_elements.shape[0], exploded_elements.shape[0]),
                                              dtype=int)
        n_components, labels = connected_components(tet_adjacency_matrix)
        labels_full[:, k] = labels
        Us.append(c)
        UU[:, k] = c
        if parameters.verbose:
            t_mode = toc(silence=True)
            ts.append(t_mode)
            print("Computed unique mode number", k + 1, "using", iter_num, "iterations and", round(t_mode, 3),
                  "seconds. This mode breaks the shape into", n_components, "pieces.")

    if parameters.verbose:
        print("Average time per mode: ", round(sum(ts) / len(ts), 3), " seconds")

    modes = UU

    return exploded_vertices, exploded_elements, modes, labels_full, tet_to_vertex_matrix_cpu, tet_neighbors, M, unexploded_to_exploded_matrix_cpu



