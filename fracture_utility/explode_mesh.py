# # Include existing libraries
# import numpy as np
# from scipy.sparse import csr_matrix
#
# # Libigl
# import igl
# # from cupyx.scipy.sparse import csr_matrix
#
# # @profile
# def explode_mesh(vertices,elements,num_quad=3):
#     # Vertices will have the following order:
#     # [tet_1_vert_1,tet_1_vert_2,....,tet_n_vert_3,tet_n_vert_4]
#     # We can get these indeces by reshaping (row-first) the elements
#     vert_indeces = np.squeeze(np.reshape(elements,(-1,1)))
#     exploded_vert_indeces = np.linspace(0,4*elements.shape[0]-1,4*elements.shape[0],dtype=int)
#     exploded_elements = np.reshape(exploded_vert_indeces,(-1,4))
#     exploded_vertices = vertices[vert_indeces,:]
#
#     # Build the matrix that takes any scalar-valued function on tets and maps it to the four vertices of the tet in the exploded mesh
#     I = exploded_vert_indeces
#     J = np.kron(np.linspace(0,elements.shape[0]-1,elements.shape[0],dtype=int),np.array([1.0,1.0,1.0,1.0],dtype=int))
#     vals = np.ones(I.shape)
#     tet_to_vertex_matrix = csr_matrix((vals,(I,J)),shape=(4*elements.shape[0],elements.shape[0]))
#
#     # Make unexploded to exploded matrix
#     J = vert_indeces
#     I = exploded_vert_indeces
#     vals = np.ones((I.shape[0]))
#     unexploded_to_exploded_matrix = csr_matrix((vals,(I,J)),shape=(4*elements.shape[0],vertices.shape[0]))
#
#     # Make discontinuity matrix (use quadrature weights)
#     if num_quad==3:
#         quad_weights = np.array([[2/3,1/3,1/3],[1/3,2/3,1/3],[1/3,1/3,2/3]])
#     elif num_quad==1:
#         quad_weights = np.array([[1.0,0.0,0.0]])
#     # There will be three nodes per internal face
#     TT, TTi = igl.tet_tet_adjacency(elements)
#     # TT #T by #4 adjacency matrix, the element i,j is the id of the tet adjacent to the j face of tet i
#     # TTi #T by #4 adjacency matrix, the element i,j is the id of face of the tet TT(i,j) that is adjacent to tet i
#     # the first face of a tet is [0,1,2], the second [0,1,3], the third [1,2,3], and the fourth [2,0,3].
#
#     # Quickly let's get indeces of neighboring tets.
#     tet_neighbors_j = np.reshape(TT,(-1,1))
#     tet_neighbors_i = np.reshape(np.kron(np.linspace(0,elements.shape[0]-1,elements.shape[0],dtype=int),np.array([1.0,1.0,1.0,1.0],dtype=int)),(-1,1))
#     tet_neighbors = np.hstack((tet_neighbors_i,tet_neighbors_j))
#     tet_neighbors = tet_neighbors[tet_neighbors_j[:,0]>-1,:]
#     # that's it, keep going building D
#
#     I = np.zeros(6*num_quad*4*exploded_elements.shape[0])
#     J = np.zeros(6*num_quad*4*exploded_elements.shape[0])
#     vals = np.zeros(6*num_quad*4*exploded_elements.shape[0])
#     num_tets = elements.shape[0]
#     # the matrix row ordering will be
#     # I = 4*num_tets*qi + 4*i + nn
#
#     # Now we have to figure out which *vertices* match
#     tet_face_ordering = np.array([[0,1,2],
#                                   [0,1,3],
#                                   [1,2,3],
#                                   [2,0,3]])
#     all_faces = np.vstack((elements[:,[0,1,2]],elements[:,[0,1,3]],elements[:,[1,2,3]],elements[:,[2,0,3]]))
#
#     areas = igl.doublearea(vertices,all_faces)
#
#     #areas[vertices[all_faces[:,0],2]>0] = 100*areas[vertices[all_faces[:,0],2]>0]
#     # This for loop is obviously suboptimal, should be vectorized. However,
#     # it is far from the bottleneck (dominated by eigsh call), so it is not
#     # very important to do it.
#     for i in range(elements.shape[0]):
#         tet_1 = i
#         for nn in range(4):
#             tet_2 = TT[i,nn]
#             if tet_2>-1 and tet_1>tet_2: # internal faces only, and only once
#                 face_area = areas[nn*elements.shape[0] + i]
#                 # tet_1 and tet_2 are neighbors. The face nn of tet_1 neighbors the face TTi[i,j] of tet_2
#                 # So the six exploded vertices we are dealing with are 4*tet_1 + tet_face_ordering[nn,:] and 4*tet_2 + tet_face_ordering[TTi[i,nn],:]
#                 # What we want to know is which of these six vertices are duplicates
#                 # In the unexploded mesh, these vertices are (in the same order) elements[tet_1,tet_face_ordering[nn,:]] and elements[tet_2,tet_face_ordering[TTi[i,nn],:]]
#                 unexploded_indeces_tet_1 = elements[tet_1,tet_face_ordering[nn,:]]
#                 unexploded_indeces_tet_2 = elements[tet_2,tet_face_ordering[TTi[i,nn],:]]
#                 exploded_indeces_tet_1 = 4*tet_1 + tet_face_ordering[nn,:]
#                 exploded_indeces_tet_2 = 4*tet_2 + tet_face_ordering[TTi[i,nn],:]
#
#                 # Then all we need is a mapping giving the equality relationship between unexploded_indeces_tet_1 and 2. We can do this by sorting them
#                 argsort_1 = np.argsort(unexploded_indeces_tet_1)
#                 argsort_2 = np.argsort(unexploded_indeces_tet_2)
#
#                 # There should be 3*6 new non-zero entries in the matrix from this internal face, in 3 different rows
#                 for qi in range(num_quad):
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 0] = 4*num_tets*qi + 4*i + nn
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 1] = 4*num_tets*qi + 4*i + nn
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 2] = 4*num_tets*qi + 4*i + nn
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 3] = 4*num_tets*qi + 4*i + nn
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 4] = 4*num_tets*qi + 4*i + nn
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 5] = 4*num_tets*qi + 4*i + nn
#                     J[ 6*(qi + num_quad*nn + num_quad*4*i) + 0] = exploded_indeces_tet_1[argsort_1[0]]
#                     J[ 6*(qi + num_quad*nn + num_quad*4*i) + 1] = exploded_indeces_tet_2[argsort_2[0]]
#                     J[ 6*(qi + num_quad*nn + num_quad*4*i) + 2] = exploded_indeces_tet_1[argsort_1[1]]
#                     J[ 6*(qi + num_quad*nn + num_quad*4*i) + 3] = exploded_indeces_tet_2[argsort_2[1]]
#                     J[ 6*(qi + num_quad*nn + num_quad*4*i) + 4] = exploded_indeces_tet_1[argsort_1[2]]
#                     J[ 6*(qi + num_quad*nn + num_quad*4*i) + 5] = exploded_indeces_tet_2[argsort_2[2]]
#                     vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 0] = quad_weights[qi,0]*face_area
#                     vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 1] = -quad_weights[qi,0]*face_area
#                     vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 2] = quad_weights[qi,1]*face_area
#                     vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 3] = -quad_weights[qi,1]*face_area
#                     vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 4] = quad_weights[qi,2]*face_area
#                     vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 5] = -quad_weights[qi,2]*face_area
#             else:
#                 for qi in range(num_quad):
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 0] = -1
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 1] = -1
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 2] = -1
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 3] = -1
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 4] = -1
#                     I[ 6*(qi + num_quad*nn + num_quad*4*i) + 5] = -1
#
#     J = J[I>-1]
#     vals = vals[I>-1]
#     I = I[I>-1]
#
#     discontinuity_matrix = csr_matrix((vals,(I,J)),shape=(num_quad*4*exploded_elements.shape[0],exploded_vertices.shape[0]))
#
#     num_nonzeros = np.diff(discontinuity_matrix.indptr)
#     discontinuity_matrix =  discontinuity_matrix[num_nonzeros != 0]
#
#     return exploded_vertices, exploded_elements, discontinuity_matrix, unexploded_to_exploded_matrix, tet_to_vertex_matrix, tet_neighbors

# import numpy as np
# import igl
# import cupy as cp
# from cupyx.scipy.sparse import coo_matrix, csr_matrix as cp_csr_matrix
# from scipy.sparse import csr_matrix as cpu_csr_matrix
#
# def explode_mesh(vertices, elements, num_quad=3, return_gpu=False):
#     # 将输入转为 GPU 数组
#     elements_gpu = cp.asarray(elements)
#     vertices_gpu = cp.asarray(vertices)
#
#     # 构造 exploded_vertices 和 exploded_elements
#     vert_indeces = cp.squeeze(cp.reshape(elements_gpu, (-1, 1)))
#     exploded_vert_indeces = cp.arange(4 * elements_gpu.shape[0], dtype=cp.int32)
#     exploded_elements_gpu = cp.reshape(exploded_vert_indeces, (-1, 4))
#     exploded_vertices_gpu = vertices_gpu[vert_indeces, :]
#
#     # 构造 tet_to_vertex_matrix (GPU上)
#     I = exploded_vert_indeces
#     J = cp.kron(
#         cp.arange(elements_gpu.shape[0], dtype=cp.int32),
#         cp.ones(4, dtype=cp.int32)
#     )
#     vals = cp.ones(I.shape, dtype=cp.float32)
#     tet_to_vertex_coo = coo_matrix(
#         (vals, (I, J)),
#         shape=(4 * elements_gpu.shape[0], elements_gpu.shape[0])
#     )
#     tet_to_vertex_matrix_gpu = tet_to_vertex_coo.tocsr()
#
#     # 构造 unexploded_to_exploded_matrix (GPU上)
#     J = vert_indeces
#     I = exploded_vert_indeces
#     vals = cp.ones(I.shape, dtype=cp.float32)
#     unexploded_to_exploded_coo = coo_matrix(
#         (vals, (I, J)),
#         shape=(4 * elements_gpu.shape[0], vertices_gpu.shape[0])
#     )
#     unexploded_to_exploded_matrix_gpu = unexploded_to_exploded_coo.tocsr()
#
#     # 构造 quad_weights
#     if num_quad == 3:
#         quad_weights = cp.array([[2/3, 1/3, 1/3],
#                                  [1/3, 2/3, 1/3],
#                                  [1/3, 1/3, 2/3]], dtype=cp.float32)
#     elif num_quad == 1:
#         quad_weights = cp.array([[1.0, 0.0, 0.0]], dtype=cp.float32)
#
#     # 计算邻接 (CPU)
#     TT, TTi = igl.tet_tet_adjacency(elements)
#     TT_gpu = cp.asarray(TT)
#     TTi_gpu = cp.asarray(TTi)
#
#     # 计算 areas (GPU)
#     tet_face_ordering = cp.array([
#         [0, 1, 2],
#         [0, 1, 3],
#         [1, 2, 3],
#         [2, 0, 3]
#     ], dtype=cp.int32)
#
#     all_faces = cp.vstack((
#         elements_gpu[:, [0, 1, 2]],
#         elements_gpu[:, [0, 1, 3]],
#         elements_gpu[:, [1, 2, 3]],
#         elements_gpu[:, [2, 0, 3]]
#     ))
#     v0 = vertices_gpu[all_faces[:, 0]]
#     v1 = vertices_gpu[all_faces[:, 1]]
#     v2 = vertices_gpu[all_faces[:, 2]]
#     areas_gpu = cp.linalg.norm(cp.cross(v1 - v0, v2 - v0), axis=1)
#
#     # 内部面筛选
#     num_tets = elements_gpu.shape[0]
#     tet_1_full = cp.repeat(cp.arange(num_tets, dtype=cp.int32), 4)
#     nn_full = cp.tile(cp.arange(4, dtype=cp.int32), num_tets)
#     tet_2_full = TT_gpu.flatten()
#     tti_full = TTi_gpu.flatten()
#
#     valid_mask = (tet_2_full > -1) & (tet_1_full > tet_2_full)
#     tet_1 = tet_1_full[valid_mask]
#     tet_2 = tet_2_full[valid_mask]
#     nn = nn_full[valid_mask]
#     tti = tti_full[valid_mask]
#     face_area = areas_gpu[nn * num_tets + tet_1]
#
#     # 获取相应的三角面顶点索引
#     unexploded_indeces_tet_1 = elements_gpu[tet_1[:, None], tet_face_ordering[nn, :]]
#     unexploded_indeces_tet_2 = elements_gpu[tet_2[:, None], tet_face_ordering[tti, :]]
#
#     exploded_indeces_tet_1 = 4 * tet_1[:, None] + tet_face_ordering[nn, :]
#     exploded_indeces_tet_2 = 4 * tet_2[:, None] + tet_face_ordering[tti, :]
#
#     argsort_1 = cp.argsort(unexploded_indeces_tet_1, axis=1)
#     argsort_2 = cp.argsort(unexploded_indeces_tet_2, axis=1)
#
#     M = tet_1.size
#     I = cp.zeros(M * num_quad * 6, dtype=cp.int32)
#     J = cp.zeros(M * num_quad * 6, dtype=cp.int32)
#     vals = cp.zeros(M * num_quad * 6, dtype=cp.float32)
#
#     for qi in range(num_quad):
#         offset = qi * M * 6
#         face_rows = 4 * num_tets * qi + 4 * tet_1 + nn
#
#         I[offset+0::6] = face_rows
#         I[offset+1::6] = face_rows
#         I[offset+2::6] = face_rows
#         I[offset+3::6] = face_rows
#         I[offset+4::6] = face_rows
#         I[offset+5::6] = face_rows
#
#         J[offset + 0::6] = exploded_indeces_tet_1[cp.arange(M), argsort_1[:, 0]]
#         J[offset + 1::6] = exploded_indeces_tet_2[cp.arange(M), argsort_2[:, 0]]
#         J[offset + 2::6] = exploded_indeces_tet_1[cp.arange(M), argsort_1[:, 1]]
#         J[offset + 3::6] = exploded_indeces_tet_2[cp.arange(M), argsort_2[:, 1]]
#         J[offset + 4::6] = exploded_indeces_tet_1[cp.arange(M), argsort_1[:, 2]]
#         J[offset + 5::6] = exploded_indeces_tet_2[cp.arange(M), argsort_2[:, 2]]
#
#         vals[offset+0::6] = quad_weights[qi,0]*face_area
#         vals[offset+1::6] = -quad_weights[qi,0]*face_area
#         vals[offset+2::6] = quad_weights[qi,1]*face_area
#         vals[offset+3::6] = -quad_weights[qi,1]*face_area
#         vals[offset+4::6] = quad_weights[qi,2]*face_area
#         vals[offset+5::6] = -quad_weights[qi,2]*face_area
#
#     discontinuity_matrix_coo = coo_matrix((vals, (I, J)), shape=(num_quad * 4 * num_tets, exploded_vertices_gpu.shape[0]))
#     discontinuity_matrix_gpu = discontinuity_matrix_coo.tocsr()
#     num_nonzeros_gpu = cp.diff(discontinuity_matrix_gpu.indptr)
#     valid_rows = (num_nonzeros_gpu != 0)
#     discontinuity_matrix_gpu = discontinuity_matrix_gpu[valid_rows.get(), :]
#
#     # tet_neighbors
#     tet_neighbors_j = TT_gpu.reshape(-1,1)
#     tet_neighbors_i = cp.kron(
#         cp.arange(num_tets, dtype=cp.int32),
#         cp.ones(4, dtype=cp.int32)
#     ).reshape(-1,1)
#     tet_neighbors_gpu = cp.hstack((tet_neighbors_i, tet_neighbors_j))
#     tet_neighbors_gpu = tet_neighbors_gpu[tet_neighbors_j[:,0]>-1,:]
#
#     if return_gpu:
#         # 直接返回 GPU 数据
#         return (exploded_vertices_gpu, exploded_elements_gpu,
#                 discontinuity_matrix_gpu, unexploded_to_exploded_matrix_gpu,
#                 tet_to_vertex_matrix_gpu, tet_neighbors_gpu)
#     else:
#         # 转为 CPU 数据
#         exploded_vertices = exploded_vertices_gpu.get()
#         exploded_elements = exploded_elements_gpu.get()
#         tet_neighbors = tet_neighbors_gpu.get()
#
#         def gpu_csr_to_cpu_csr(gpu_csr_mat):
#             return cpu_csr_matrix((gpu_csr_mat.data.get(), gpu_csr_mat.indices.get(), gpu_csr_mat.indptr.get()),
#                                   shape=gpu_csr_mat.shape)
#
#         discontinuity_matrix = gpu_csr_to_cpu_csr(discontinuity_matrix_gpu)
#         unexploded_to_exploded_matrix = gpu_csr_to_cpu_csr(unexploded_to_exploded_matrix_gpu)
#         tet_to_vertex_matrix = gpu_csr_to_cpu_csr(tet_to_vertex_matrix_gpu)
#
#         return (exploded_vertices, exploded_elements, discontinuity_matrix,
#                 unexploded_to_exploded_matrix, tet_to_vertex_matrix, tet_neighbors)

import numpy as np
import cupy as cp
from cupyx.scipy.sparse import coo_matrix, csr_matrix as cp_csr_matrix
from scipy.sparse import csr_matrix as cpu_csr_matrix


def explode_mesh(vertices, elements, num_quad=3, return_gpu=False):
    # Convert input to GPU arrays
    elements_gpu = cp.asarray(elements)
    vertices_gpu = cp.asarray(vertices)
    num_tets = elements_gpu.shape[0]

    # Construct exploded_vertices and exploded_elements
    vert_indeces = cp.squeeze(cp.reshape(elements_gpu, (-1, 1)))
    exploded_vert_indeces = cp.arange(4 * num_tets, dtype=cp.int32)
    exploded_elements_gpu = cp.reshape(exploded_vert_indeces, (-1, 4))
    exploded_vertices_gpu = vertices_gpu[vert_indeces, :]

    # Construct tet_to_vertex_matrix (on GPU)
    I = exploded_vert_indeces
    J = cp.kron(
        cp.arange(num_tets, dtype=cp.int32),
        cp.ones(4, dtype=cp.int32)
    )
    vals = cp.ones(I.shape, dtype=cp.float32)
    tet_to_vertex_coo = coo_matrix(
        (vals, (I, J)),
        shape=(4 * num_tets, num_tets)
    )
    tet_to_vertex_matrix_gpu = tet_to_vertex_coo.tocsr()

    # Construct unexploded_to_exploded_matrix (on GPU)
    J = vert_indeces
    I = exploded_vert_indeces
    vals = cp.ones(I.shape, dtype=cp.float32)
    unexploded_to_exploded_coo = coo_matrix(
        (vals, (I, J)),
        shape=(4 * num_tets, vertices_gpu.shape[0])
    )
    unexploded_to_exploded_matrix_gpu = unexploded_to_exploded_coo.tocsr()

    # Construct quad_weights (still on GPU)
    if num_quad == 3:
        quad_weights = cp.array([[2 / 3, 1 / 3, 1 / 3],
                                 [1 / 3, 2 / 3, 1 / 3],
                                 [1 / 3, 1 / 3, 2 / 3]], dtype=cp.float32)
    elif num_quad == 1:
        quad_weights = cp.array([[1.0, 0.0, 0.0]], dtype=cp.float32)
    else:
        raise ValueError("num_quad currently only supports 1 or 3")

    # -------------------- Full GPU adjacency computation --------------------
    # Each tetrahedron has 4 faces: local vertex indices for each face
    num_tets = elements_gpu.shape[0]

    face_patterns = cp.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3]
    ], dtype=cp.int32)

    tet_ids = cp.repeat(cp.arange(num_tets, dtype=cp.int32), 4)  # (4*num_tets,)

    # Extend face_patterns to (4*num_tets, 3)
    tile_face_patterns = cp.tile(face_patterns, (num_tets, 1))  # (4*num_tets, 3)

    # Extend tet_ids to (4*num_tets, 3) for broadcasting
    tet_ids_expanded = tet_ids[:, None].repeat(3, axis=1)  # (4*num_tets, 3)

    # Perform advanced indexing without broadcasting errors
    faces_unordered = elements_gpu[tet_ids_expanded, tile_face_patterns]  # (4*num_tets, 3)
    # Sort face vertices (each face vertex is sorted) to get unique representation
    faces_sorted = cp.sort(faces_unordered, axis=1)

    # Combine (face vertices, tet_id) for sorting and grouping
    # Structure is (N*4, 4): First 3 columns are sorted face vertices, last column is the tet id
    combined = cp.hstack([faces_sorted, tet_ids.reshape(-1, 1)])

    # Sort first 3 columns (face vertices) to group duplicates
    # cp.lexsort uses the last sequence as the primary key, so reverse the order for sorting columns 0, 1, 2
    # lexsort requires tuple and reverses the order of the keys
    keys = cp.vstack((combined[:, 2], combined[:, 1], combined[:, 0]))

    # Use cp.lexsort
    sorted_indices = cp.lexsort(keys)
    combined_sorted = combined[sorted_indices]

    # Find adjacent duplicate faces
    same_face = cp.all(combined_sorted[:-1, :3] == combined_sorted[1:, :3], axis=1)

    adj_indices = cp.where(same_face)[0]
    # Each duplicate face corresponds to two adjacent tets
    tet1 = combined_sorted[adj_indices, 3]
    tet2 = combined_sorted[adj_indices + 1, 3]

    # Construct list of adjacent tets (store in both directions if needed, here store one set)
    # If users need TT, TTi-like data structures similar to igl, further processing is required.
    # Output a (tet1, tet2) relationship.
    # The original code uses tet_neighbors as a list of (tet_i, tet_j) pairs, which we generate directly:
    tet_neighbors_gpu = cp.vstack((tet1, tet2)).T
    # This gives all internal adjacent tet pairs

    # -------------------- Area calculation --------------------
    all_faces = cp.vstack((
        elements_gpu[:, [0, 1, 2]],
        elements_gpu[:, [0, 1, 3]],
        elements_gpu[:, [1, 2, 3]],
        elements_gpu[:, [2, 0, 3]]
    ))
    v0 = vertices_gpu[all_faces[:, 0]]
    v1 = vertices_gpu[all_faces[:, 1]]
    v2 = vertices_gpu[all_faces[:, 2]]
    areas_gpu = cp.linalg.norm(cp.cross(v1 - v0, v2 - v0), axis=1)

    # Internal face filtering: valid tet pairs are already in tet_neighbors_gpu
    # We need the corresponding face indices nn and tti. While the original used TT and TTi,
    # here, we reconstruct nn and tti based on matching face indices.
    # Simplified for consistency with original logic grouping tet_tet_distances requires nn/tti:

    # Use backtracking to find the original face index
    face_idx_1 = sorted_indices[adj_indices]  # First entry
    face_idx_2 = sorted_indices[adj_indices + 1]  # Second entry

    # face_idx // 4 gives tet number, face_idx % 4 gives face index nn
    tet_1 = tet_ids[face_idx_1]
    nn = face_idx_1 % 4
    tet_2 = tet_ids[face_idx_2]
    tti = face_idx_2 % 4  # Corresponding face index in tet_2
    M = tet_1.size
    face_area = areas_gpu[nn * num_tets + tet_1]

    # -------------------- Vectorized filling of I, J, vals --------------------
    # Original code loops over qi to fill I, J, vals; we complete in one go
    # Pattern has 6 entries per constraint:
    #    (tet_1 vertex[0], tet_2 vertex[0]),
    #    (tet_1 vertex[1], tet_2 vertex[1]),
    #    (tet_1 vertex[2], tet_2 vertex[2])
    # Each face has 3 vertex pairs, total 6 entries (positive and negative)

    # Use tet_face_ordering to extract corresponding face vertices:
    tet_face_ordering = face_patterns
    exploded_indeces_tet_1 = 4 * tet_1[:, None] + tet_face_ordering[nn, :]
    exploded_indeces_tet_2 = 4 * tet_2[:, None] + tet_face_ordering[tti, :]

    # Sort 3 vertices per face for matching with opposing tet face vertices
    argsort_1 = cp.argsort(exploded_indeces_tet_1, axis=1)
    argsort_2 = cp.argsort(exploded_indeces_tet_2, axis=1)

    # Generate 6 entries per constraint: i.e., (x1, -x2, y1, -y2, z1, -z2)
    # num_quad repetitions; each qi affects face_rows
    Q = num_quad
    # face_rows: from original code face_rows = 4 * num_tets * qi + 4 * tet_1 + nn
    # Construct (Q, M) matrix and flatten
    qi_array = cp.arange(Q, dtype=cp.int32)[:, None]  # Shape (Q, 1)
    face_rows_2d = 4 * num_tets * qi_array + 4 * tet_1 + nn  # (Q, M)
    face_rows_all = face_rows_2d.ravel()  # Flatten to (Q * M)

    # Need 6 entries per qi, total length = M * Q * 6
    total_len = M * Q * 6
    I_all = cp.repeat(face_rows_all, 6)  # Repeat each (Q * M) entry 6 times

    # Fill J and vals considering index patterns
    # Sequence: (tet_1[v0], tet_2[v0]), (tet_1[v1], tet_2[v1]), (tet_1[v2], tet_2[v2])
    idx_arange = cp.arange(M)
    t1v0 = exploded_indeces_tet_1[idx_arange, argsort_1[:, 0]]
    t1v1 = exploded_indeces_tet_1[idx_arange, argsort_1[:, 1]]
    t1v2 = exploded_indeces_tet_1[idx_arange, argsort_1[:, 2]]
    t2v0 = exploded_indeces_tet_2[idx_arange, argsort_2[:, 0]]
    t2v1 = exploded_indeces_tet_2[idx_arange, argsort_2[:, 1]]
    t2v2 = exploded_indeces_tet_2[idx_arange, argsort_2[:, 2]]

    # Repeat these values for all qi
    t1v0_all = cp.repeat(t1v0, Q)
    t1v1_all = cp.repeat(t1v1, Q)
    t1v2_all = cp.repeat(t1v2, Q)
    t2v0_all = cp.repeat(t2v0, Q)
    t2v1_all = cp.repeat(t2v1, Q)
    t2v2_all = cp.repeat(t2v2, Q)

    # Similarly, repeat face_area for qi and scale by corresponding weights
    face_area_all = cp.repeat(face_area, Q)  # Shape (M * Q,)
    q_idx = cp.tile(cp.arange(Q), M)  # Repeat pattern [0,1,...,Q-1] M times

    # Compute vals using qi and face weights
    w0 = quad_weights[q_idx, 0] * face_area_all
    w1 = quad_weights[q_idx, 1] * face_area_all
    w2 = quad_weights[q_idx, 2] * face_area_all

    # Sequence follows original code (6 entries):
    # 0: t1v0 -> +w0
    # 1: t2v0 -> -w0
    # 2: t1v1 -> +w1
    # 3: t2v1 -> -w1
    # 4: t1v2 -> +w2
    # 5: t2v2 -> -w2
    J_all = cp.empty(total_len, dtype=cp.int32)
    vals_all = cp.empty(total_len, dtype=cp.float32)

    # Expand (M * Q) to (M * Q * 6); each (M * Q) block repeats pattern
    base_idx = cp.arange(M * Q)
    J_all[0::6] = t1v0_all
    J_all[1::6] = t2v0_all
    J_all[2::6] = t1v1_all
    J_all[3::6] = t2v1_all
    J_all[4::6] = t1v2_all
    J_all[5::6] = t2v2_all

    vals_all[0::6] = w0
    vals_all[1::6] = -w0
    vals_all[2::6] = w1
    vals_all[3::6] = -w1
    vals_all[4::6] = w2
    vals_all[5::6] = -w2

    # Construct discontinuity_matrix (on GPU)
    discontinuity_matrix_coo = coo_matrix((vals_all, (I_all, J_all)),
                                          shape=(num_quad * 4 * num_tets, exploded_vertices_gpu.shape[0]))
    discontinuity_matrix_gpu = discontinuity_matrix_coo.tocsr()
    num_nonzeros_gpu = cp.diff(discontinuity_matrix_gpu.indptr)
    valid_rows = (num_nonzeros_gpu != 0)
    discontinuity_matrix_gpu = discontinuity_matrix_gpu[valid_rows.get(), :]

    if return_gpu:
        # Directly return GPU data
        return (exploded_vertices_gpu, exploded_elements_gpu,
                discontinuity_matrix_gpu, unexploded_to_exploded_matrix_gpu,
                tet_to_vertex_matrix_gpu, tet_neighbors_gpu)
    else:
        # Convert to CPU data
        exploded_vertices = exploded_vertices_gpu.get()
        exploded_elements = exploded_elements_gpu.get()
        tet_neighbors = tet_neighbors_gpu.get()

        def gpu_csr_to_cpu_csr(gpu_csr_mat):
            return cpu_csr_matrix((gpu_csr_mat.data.get(), gpu_csr_mat.indices.get(), gpu_csr_mat.indptr.get()),
                                  shape=gpu_csr_mat.shape)

        discontinuity_matrix = gpu_csr_to_cpu_csr(discontinuity_matrix_gpu)
        unexploded_to_exploded_matrix = gpu_csr_to_cpu_csr(unexploded_to_exploded_matrix_gpu)
        tet_to_vertex_matrix = gpu_csr_to_cpu_csr(tet_to_vertex_matrix_gpu)

        return (exploded_vertices, exploded_elements, discontinuity_matrix,
                unexploded_to_exploded_matrix, tet_to_vertex_matrix, tet_neighbors)
