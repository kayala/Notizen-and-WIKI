* CUDA stands for Compute Unified Device Architecture. CUDA is a heterogeneous programming language from NVIDIA that exposes GPU for general purpose program. Heterogeneous programming means the code runs on two different platform: host (CPU) and devices (NVDIA GPU devices).
* Host: Host is the CPU is available in the system. System memory(D RAM) associated with host is called Host memory.
* Devices: Device is GPU and GPU memory is called device memory.
* Kernel: Kernel is a function executed in the GPU by a single thread.
* Kernel launch: It is a CPU instructing GPU to execute code in parallel.
* Execution Configuration: Number of threads that runs in GPU and how to define it.
* CUDA cores: It is the floating point unit of NVDIA graphics card that can perform a floating point map.
* Workflow
  * The host is in control of the execution. The program loads sequentially till it reaches the kernel. The host does kernal launch. it reaches another kernel
  * host-to-device transfer: Memory is allocated to the device memory using cudaMalloc. Data from host memory is copied to device using cudaMemcpy. The communication of device and host is via PCI bus.
  * kernal-load: Load the GPU program and execute, caching data on-chip for performance.
  * device-to-host transfer: Copy the results from device memory to host memory using cudaMemcpy. The memory is cleared from device memory using cudaFree.
* __global__: is a indicates that the function runs on device(GPU) and is called from Host (CPU). It defines kernal code.
* hello_world<<<1,1>>(): The <<<M,T>>> signifies the kernal launch. It needs to be called for code that has __global__ in it. The kernel launches with a grid of M thread blocks. Each thread block has T parallel threads.
* cudaDeviceSynchronize(): The cuda code launches asynchronously and host might not always wait for device to complete its execution. The cudaDeviceSyncronize waits for device to complete execution.
* compile: 'nvcc -o hello helloworld.cu'
* Hierarchy of threads
  * Threads: Threads are a single process running in one of the cuda cores. Each kernel call creates a single grid.
  * Blocks: Blocks are collections of threads.
  * grid: Grids are Collections of blocks.
* Inbuild variables
  * threadIdx.x: We can get index of the current thread within its block with using inbuilt variable.
  * blockIdx.x: We can get index of the current block in the grid.
  * blockDim.x: Number of threads in the block can be gotten with using inbuilt variable.
  * gridDim.x: We can size of the grid using inbuilt variable.
* Components of GPU
  * global memory: It is the same as RAM in CPU. It can be accessed by both device and host.
  * Streaming Multiprocessor: The device that does actual computation. In layman, term, Its CUDA cores.