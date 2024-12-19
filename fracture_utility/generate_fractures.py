# Include existing libraries
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, kron
from scipy.sparse.linalg import lsqr, spsolve
from scipy.sparse.csgraph import connected_components
# Libigl
import igl
import tetgen
from scipy.stats import multivariate_normal


import time
import sys
import os
import gpytoolbox
from gpytoolbox.copyleft import lazy_cage

from .fracture_modes_parameters import fracture_modes_parameters
from .fracture_modes import fracture_modes



def generate_fractures(input_dir,num_modes=20,num_impacts=80,output_dir=None,verbose=True,compressed=True,cage_size=4000,volume_constraint=(1/50)):
    """Randomly generate different fractures of a given object and write them to an output directory.

    Parameters
    ----------
    input_dir : str
        Path to a mesh file in .obj, .ply, or any other libigl-readable format
    num_modes : int (optional, default 20)
        Number of modes to consider (more modes will give more diversity to the fractures but will also be slower)
    num_impacts : int (optional, default 80)
        How many different random fractures to output
    output_dir : str (optional, default None)
        Path to the directory where all the fractures will be written
    compressed : bool (optional, default True)
        Whether to write the fractures as compressed .npy files instead of .obj. Needs to use `decompress.py` to decompress them afterwards.
    cage_size : int (optional, default 4000)
        Number of faces in the simulation mesh used
    volume_constraint : double (optional, default 0)
        Will only consider fractures with minimum piece volume larger than volume_constraint times the volume of the input. Values over 0.01 may severely delay runtime.
    """

    # directory = os.fsencode(input_dir)
    np.random.seed(0)
    # for file in os.listdir(directory):
    filename = input_dir
    t0 = time.time()
    #try:
    t00 = time.time()
    v_fine, f_fine = igl.read_triangle_mesh(filename)
    # Let's normalize it so that parameter choice makes sense
    v_fine = gpytoolbox.normalize_points(v_fine)
    t01 = time.time()
    reading_time = t01-t00
    if verbose:
        print("Read shape in",round(reading_time,3),"seconds.")
    # Build cage mesh (this may actually be the bottleneck...)
    t10 = time.time()
    v, f = lazy_cage(v_fine,f_fine,num_faces=cage_size,grid_size=256)
    t11 = time.time()
    cage_time = t11-t10
    if verbose:
        print("Built cage in",round(cage_time,3),"seconds.")
    # Tetrahedralize cage mesh
    t20 = time.time()
    tgen = tetgen.TetGen(v,f)
    nodes, elements =  tgen.tetrahedralize(minratio=1.5)
    t21 = time.time()
    tet_time = t21-t20
    if verbose:
        print("Tetrahedralization in ",round(tet_time,3),"seconds.")

    # Initialize fracture mode class
    t30 = time.time()
    modes = fracture_modes(nodes,elements)
    # Set parameters for call to fracture modes
    params = fracture_modes_parameters(num_modes=num_modes,verbose=False,d=1)
    # Compute fracture modes. This should be the bottleneck:
    modes.compute_modes(parameters=params)
    modes.impact_precomputation(v_fine=v_fine,f_fine=f_fine)

    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)



    if compressed:
        modes.write_generic_data_compressed(output_dir)
        modes.write_segmented_modes_compressed(output_dir)
    else:
        modes.write_segmented_modes(output_dir,pieces=True)
    t31 = time.time()
    mode_time = t31-t30
    if verbose:
        print("Modes computed in ",round(mode_time,3),"seconds.")
    # # Generate random contact points on the surface
    B,FI = igl.random_points_on_mesh(1000*num_impacts,v,f)
    B = np.vstack((B[:,0],B[:,0],B[:,0],B[:,1],B[:,1],B[:,1],B[:,2],B[:,2],B[:,2])).T
    P = B[:,0:3]*v[f[FI,0],:] + B[:,3:6]*v[f[FI,1],:] + B[:,6:9]*v[f[FI,2],:]
    sigmas = np.random.rand(1000*num_impacts)*1000

    vols = igl.volume(modes.vertices,modes.elements)
    total_vol = np.sum(vols)

    t40 = time.time()
    # Loop to generate many possible fractures
    all_labels = np.zeros((modes.precomputed_num_pieces,num_impacts),dtype=int)
    running_num = 0
    for i in range(P.shape[0]):
        t400 = time.time()
        modes.impact_projection(contact_point=P[i,:],direction=np.array([1.0]),threshold=sigmas[i],num_modes_used=20)
        min_volume = volume_constraint*total_vol/(modes.n_pieces_after_impact)
        current_min_volume = total_vol
        for i in range(modes.n_pieces_after_impact):
            current_min_volume = min(current_min_volume,np.sum(vols[modes.tet_labels_after_impact==i]))
        valid_volume = (current_min_volume >= min_volume)
        t401 = time.time()
        # if verbose:
        #     print("Impact simulation: ",round(t401-t400,3),"seconds.")
        new = not (modes.piece_labels_after_impact.tolist() in all_labels.T.tolist())
        #print(modes.piece_labels_after_impact.tolist() in all_labels.T.tolist())
        if (modes.n_pieces_after_impact>1 and modes.n_pieces_after_impact<100 and new and valid_volume):
            all_labels[:,running_num] = modes.piece_labels_after_impact
            write_output_name = os.path.join(output_dir,"fractured_") +  str(running_num)
            running_num = running_num + 1
            if not os.path.exists(write_output_name):
                        os.mkdir(write_output_name)
            if compressed:
                modes.write_segmented_output_compressed(filename=write_output_name)
            else:
                modes.write_segmented_output(filename=write_output_name,pieces=True)
            t402 = time.time()
            # if verbose:
            #     print("Writing: ",round(t402-t401,3),"seconds.")
        if running_num >= num_impacts:
            break
    #print(all_labels)
    t41 = time.time()
    impact_time = t41-t40
    if verbose:
        print("Impacts computed in ",round(impact_time,3),"seconds.")
    t1 = time.time()
    total_time = t1-t0
    if verbose:
        print("Generated",running_num,"fractures for object",filename_without_extension,"and wrote them into",output_dir + "/","in",round(total_time,3),"seconds.")
    # except:
    #     if verbose:
    #         print("Error encountered.")

# from concurrent.futures import ThreadPoolExecutor
# import numpy as np
# import os
# import igl
# import tetgen
# from scipy.sparse import csr_matrix
# from .fracture_modes_parameters import fracture_modes_parameters
# from .fracture_modes import fracture_modes
# import cupy as cp
# import cupyx.scipy.sparse as cp_sparse
#
# from gpytoolbox.copyleft import lazy_cage
# # Import other necessary modules (placeholders for now)
# # from .fracture_modes import fracture_modes
# # from .fracture_modes_parameters import fracture_modes_parameters
#
#
# def process_single_impact(index, contact_point, sigma, modes, volume_constraint, total_vol, stream=None):
#     """
#     单次断裂生成任务 (使用 GPU 和 CUDA 流)
#     """
#     try:
#         # 使用指定 CUDA 流
#         if stream is None:
#             stream = cp.cuda.Stream(non_blocking=True)
#
#         with stream:
#             # 切换到 GPU 上的矩阵和向量操作
#             contact_point_gpu = cp.asarray(contact_point)
#             sigma_gpu = cp.asarray(sigma)
#
#             # 使用 GPU 进行 impact_projection
#             modes.impact_projection(contact_point=contact_point_gpu, direction=cp.array([1.0]), threshold=sigma_gpu, num_modes_used=20)
#
#             # 验证体积约束是否满足
#             min_volume = volume_constraint * total_vol / modes.n_pieces_after_impact
#             current_min_volume = total_vol
#             for i in range(modes.n_pieces_after_impact):
#                 current_min_volume = min(
#                     current_min_volume,
#                     cp.sum(modes.volumes[modes.tet_labels_after_impact == i])  # GPU 加速
#                 )
#             valid_volume = current_min_volume >= min_volume
#
#             # 如果断裂有效且满足体积约束
#             if valid_volume and modes.n_pieces_after_impact > 1:
#                 return modes.piece_labels_after_impact, modes.n_pieces_after_impact
#             else:
#                 return None
#
#     except Exception as e:
#         print(f"Error in task {index}: {e}")
#         return None
#     finally:
#         # 确保同步流
#         if stream is not None:
#             stream.synchronize()
#
# def generate_fractures(input_dir, num_modes=20, num_impacts=80, output_dir=None, verbose=True, compressed=True,
#                        cage_size=4000, volume_constraint=(1 / 50), num_threads=16):
#     """
#     使用 GPU 和 CUDA 流优化的断裂生成
#     """
#     np.random.seed(0)
#     v_fine, f_fine = igl.read_triangle_mesh(input_dir)
#     v_fine = cp.asarray(v_fine)  # 转换到 GPU
#     f_fine = cp.asarray(f_fine)
#
#     # 调用 lazy_cage 前，将数据从 CuPy 转为 NumPy
#     v_fine_numpy = v_fine.get()
#     f_fine_numpy = f_fine.get()
#
#     # 调用 lazy_cage（需要 NumPy 数据）
#     v, f = lazy_cage(v_fine_numpy, f_fine_numpy, num_faces=cage_size)
#
#     # 确保输入 `tetgen.TetGen` 时为 NumPy 数组
#     v = np.asarray(v)  # 将 CuPy 或其他可能的格式转换为 NumPy 数组
#     f = np.asarray(f)
#
#     # 四面体化
#     tgen = tetgen.TetGen(v, f)
#     nodes, elements = tgen.tetrahedralize(minratio=1.5)
#     nodes = cp.asarray(nodes)  # 转换到 GPU
#     elements = cp.asarray(elements)
#
#     # 初始化 fracture_modes 对象
#     modes = fracture_modes(nodes, elements)
#     params = fracture_modes_parameters(num_modes=num_modes, verbose=False, d=1)
#     modes.compute_modes(parameters=params)
#     modes.impact_precomputation(v_fine=v_fine, f_fine=f_fine)
#
#     # 随机生成冲击点
#     B, FI = igl.random_points_on_mesh(1000 * num_impacts, v, f)
#     P = B[:, 0:3] * v[f[FI, 0], :] + B[:, 3:6] * v[f[FI, 1], :] + B[:, 6:9] * v[f[FI, 2], :]
#     sigmas = cp.random.rand(1000 * num_impacts) * 1000  # GPU 上生成随机数
#
#     # 初始化结果存储
#     results = []
#     total_vol = cp.sum(igl.volume(modes.vertices, modes.elements))  # GPU 化体积计算
#
#     # 并行化执行，每个线程分配一个 CUDA 流
#     streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_threads)]
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         futures = [
#             executor.submit(process_single_impact, i, P[i], sigmas[i], modes, volume_constraint, total_vol, streams[i % num_threads])
#             for i in range(len(P))
#         ]
#
#         for future in futures:
#             result = future.result()
#             if result is not None:
#                 results.append(result)
#
#     # 输出结果
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#
#     for idx, (labels, n_pieces) in enumerate(results):
#         fracture_dir = os.path.join(output_dir, f"fracture_{idx}")
#         os.makedirs(fracture_dir, exist_ok=True)
#
#         # 保存断裂模式
#         modes.write_segmented_output_compressed(fracture_dir)
#
#     if verbose:
#         print(f"Generated {len(results)} fractures and saved to {output_dir}.")