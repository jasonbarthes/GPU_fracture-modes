## Credits
This project is based on [Breaking Good: Fracture Modes for Realtime Destruction](https://github.com/sgsellan/fracture-modes) by [Silvia Sellán].

---

### Project Overview

This project is focused on optimizing the performance of the "Breaking Good: Fracture Modes for Realtime Destruction" project. As indicated by the author in the original paper, GPU acceleration could enhance its performance. This version incorporates GPU computation, transitioning from CPU-based to GPU-based calculations wherever possible. Below are the details of the GPU-optimized version:

---

### Current Status of the Project

#### GPU-Optimized or Parallelized Components:

1. **Tetrahedral Mass Matrix Calculation (`massmatrix_tets`):**
   - Replaced CPU-based `numpy` and `scipy` implementations with `CuPy`.
   - All computations for tetrahedral mass matrix calculations now execute on the GPU.

2. **Exploded Tetrahedral Mesh (`explode_mesh`):**
   - Implemented a GPU-accelerated version to handle vertex and tetrahedral index explosion, adjacency computation, and discontinuity matrix construction.

3. **Eigenvalue Decomposition Data Transfer:**
   - While eigenvalue decomposition still runs on the CPU using `scipy.eigsh`, the data generation occurs on the GPU, reducing bottlenecks.

4. **Impact Calculation (`fracture_modes`):**
   - Utilized the GPU for adjacency-based distance calculations between tetrahedra, matrix operations, and vector computations.
   
5. **Parallel Computation of Fracture Modes:**
   - Adopted `ThreadPoolExecutor` to parallelize mode computation, reducing processing time.

---

#### CPU-Dependent Components (Limitations for Full GPU Transition):

1. **Mode Computation:**
   - The current algorithm requires iterative dependency on prior mode results (stored in `Us`), which prevents straightforward parallelization. Attempts to fully parallelize (e.g., via multi-threading or multi-processing) led to incorrect object segmentation.
   - A potential redesign could leverage Gram-Schmidt orthogonalization to generate an orthogonal seed set for parallel mode calculations. However, this would require ensuring orthogonality throughout the computation.

2. **Sparse Matrix Decomposition (`sparse_sqrt`):**
   - Cholesky decomposition of large sparse matrices does not scale well on the GPU due to its complexity (O(n^3)).
   - Enabled `CHOLMOD_USE_GPU` for partial GPU acceleration but limited by memory size constraints.

3. **Eigenvalue Decomposition (`eigsh`):**
   - The `scipy.sparse.linalg.eigsh` function is not GPU-compatible, necessitating CPU execution.

4. **Conic Programming Solver (`conic_solve`):**
   - Relies on the MOSEK solver, which supports only CPU computation.

5. **Sparse Kronecker Product and Connected Components:**
   - Operations such as `kron` and `connected_components` are not supported on GPUs and rely on `scipy`.

6. **Mesh Boolean Operations and Post-Processing:**
   - Complex mesh operations like `cotmatrix` and `remove_unreferenced` from `igl` are CPU-bound.

---

#### If you compile the original version of the project provided by the author, you might find that its model processing speed is similar to mine. This is because GPU acceleration has not become the computational bottleneck here. In this scenario, all tasks transferred to the GPU are not granular enough, and their computation speeds are very fast, making the CPU-bound calculations, which cannot be replaced by the GPU, the main factor slowing down progress. To achieve significant acceleration, it is necessary to avoid these CPU-only libraries from the design phase and build an architecture that supports parallelization.

---

### How to Build
1. First, ensure that you have completed the configuration requirements of the "Breaking Good: Fracture Modes for Realtime Destruction" project. I have replaced the provided `requirements.txt` with a version where some library versions are corrected.

2. Ensure that your computer has the correct version of CUDA installed.

3. I have included the file `manyfaces.obj` in this project. If you wish to use your own model, you can rename it to `manyfaces.obj`.

4. My optimization is entirely based on the provided GUI, meaning I have not attempted to use this project as a library in other projects. The command-line execution method is `python scripts/fracture_gui.py manyfaces.obj`. If you want to change the model, you can replace `manyfaces` with your file name. Before running this command, you can enable CHOLMOD GPU acceleration by setting `$env:CHOLMOD_USE_GPU = "1"`. These options can also be configured in your IDE's project debugging settings.

5. Although not recommended, you can try selecting excessively high `faces in cage` values. In some cases, such as the current kevlar board model, which has a relatively simple shape, the interpolation method of `lazy_cage` might cause a face explosion. This could result in the input face count increasing slightly, leading to tenfold or even hundredfold processing face counts, potentially causing memory overflow. For the current object, I recommend you set this parameter to a number smaller than 8000.
