# NSTU Parallel Programming 5 semester

### Lab 1

#### OpenMP

```bash
cd lab1
make
```

### Lab 2

#### NVIDIA CUDA

> requires NVIDIA GPU with [Compute Capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported):

- \>= 3.5 for recursive algorithm (Dynamic parallelism)

```bash
cd lab2
make recursive
```

- \>= 2.0 for iterative algorithm (Slower than sequential)

```bash
cd lab2
make
```