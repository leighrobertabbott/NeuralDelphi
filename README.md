# NeuralDelphi

<p align="center">
  <img src="https://img.shields.io/badge/Delphi-EE1F35?style=for-the-badge&logo=delphi&logoColor=white" alt="Delphi"/>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge" alt="Windows"/>
</p>

**A high-performance, pure Delphi machine learning framework.** No Python. No external DLLs. Just fast, native code.

---

## ✨ Features

- **🚀 Arena-Based Memory** — Zero allocation/deallocation during training
- **⚡ SIMD Assembly** — Hand-tuned SSE kernels for x64
- **🔄 Automatic Differentiation** — Full autograd with computation graphs
- **🧵 Thread Pool Parallelization** — Efficient multi-core utilization
- **📦 Zero Dependencies** — Pure Delphi, compiles standalone
- **🎛️ N-Dimensional Tensors** — Full Shape/Strides support for any dimensionality
- **📡 Broadcasting** — NumPy-style automatic shape broadcasting
- **🔢 Batch Operations** — Batched matrix multiplication for 3D+ tensors

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NeuralDelphi                              │
├─────────────────────────────────────────────────────────────────┤
│  ML.Arena    │  Linear memory allocator (zero GC overhead)       │
│  ML.Tensor   │  N-D tensor views with Shape/Strides              │
│  ML.Ops      │  SIMD kernels + parallel ops + broadcasting       │
│  ML.Graph    │  Computation graph + autograd                     │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### **ML.Arena.pas** - Memory Management
The foundation of NeuralDelphi's performance. Implements a **linear allocator** (also called a "bump allocator" or "arena allocator") that pre-allocates a large contiguous block of memory.

**Key Concepts:**
- **`TMemPtr`**: An `Integer` index into the arena, not a pointer. This avoids pointer arithmetic issues and makes the system 32/64-bit agnostic.
- **`TArena.Alloc(Count)`**: O(1) allocation - just increments the head pointer. No free lists, no fragmentation.
- **`TArena.Reset()`**: O(1) deallocation - sets head to 0. All memory is "freed" instantly.
- **`GetSavePoint()` / `Restore()`**: Critical for the graph architecture. Allows resetting only temporary activations while keeping persistent parameters.

**Why This Matters:**
Traditional `GetMem`/`FreeMem` calls are expensive (kernel calls, heap fragmentation). During training, you might allocate millions of temporary tensors. The arena eliminates this overhead entirely.

**Example:**
```delphi
Arena := TArena.Create(256);        // Allocate 256MB block
W1 := Arena.Alloc(8 * 2);          // Allocate 16 floats (8x2 matrix)
W2 := Arena.Alloc(1 * 8);          // Allocate 8 floats (1x8 matrix)
// ... use W1, W2 ...
Arena.Reset;                        // Free everything instantly
```

---

#### **ML.Tensor.pas** - N-Dimensional Tensor Abstraction
A lightweight `record` (not a class!) that acts as a **view** into the arena. Supports arbitrary dimensions with NumPy-style shape and strides.

**Key Fields:**
- **`DataPtr: TMemPtr`**: Index into arena where tensor data lives
- **`GradPtr: TMemPtr`**: Index for gradients (allocated on-demand during backward pass)
- **`Shape: TArray<Integer>`**: Dimensions, e.g., `[32, 3, 224, 224]` for batch of images
- **`Strides: TArray<Integer>`**: Memory strides for each dimension (row-major)
- **`RequiresGrad: Boolean`**: Whether this tensor needs gradients computed

**Key Methods:**
- **`NDim`**: Returns number of dimensions
- **`ElementCount`**: Total number of elements (product of shape)
- **`IsContiguous`**: Checks if memory layout matches strides
- **`GetLinearIndex(Indices)`**: Converts N-D indices to linear index
- **`Reshape(NewShape)`**: Zero-copy view with new shape
- **`Transpose(Dim0, Dim1)`**: Zero-copy dimension swap
- **`Squeeze` / `Unsqueeze`**: Add/remove dimensions of size 1
- **`RawData(Arena)`**: Returns `PSingle` pointer for direct memory access
- **`RawGrad(Arena)`**: Returns gradient pointer, or `nil` if not allocated

**Why Records, Not Classes:**
- Zero heap allocation overhead
- Value semantics (can copy freely)
- Cache-friendly (all data in one contiguous block)

**Example:**
```delphi
var
  T: TTensor;
begin
  T := TTensor.Create(Arena, [32, 8, 64], True);  // 3D tensor, needs gradients
  // T.Shape = [32, 8, 64]
  // T.Strides = [512, 64, 1]  (row-major)
  // T.ElementCount = 16384
  
  // Zero-copy reshape
  T2 := T.Reshape([32, 512]);  // Same data, different view
  
  // Zero-copy transpose
  T3 := T.Transpose(0, 1);     // Swaps first two dimensions
end;
```

---

#### **ML.Ops.pas** - Mathematical Operations
Contains three layers: **Pure ASM Kernels**, **Parallel Execution**, and **High-Level Tensor Ops** with **Broadcasting Support**.

**1. `TKernels` - Pure Assembly Math Kernels**
Hand-written x64 SSE assembly for maximum performance. These are **stateless** functions that operate on raw pointers.

- **`DotProduct(A, B, Count)`**: SIMD dot product using `MOVUPS`, `MULPS`, `ADDPS`, `HADDPS`. Processes 4 floats at once.
- **`VectorAdd(A, B, Out, Count)`**: Element-wise addition with SSE `ADDPS`.
- **`VectorMul(A, B, Out, Count)`**: Element-wise multiplication with SSE `MULPS`.
- **`Transpose(Src, Dst, Rows, Cols)`**: Block-based matrix transpose (8x8 blocks) for cache efficiency.

**2. `TMLParallel` - Thread Pool Wrapper**
Wraps `System.Threading.TParallel.For` with a threshold check. Only parallelizes if workload is substantial (>256 elements) to avoid overhead.

**3. `TOps` - High-Level Tensor Operations with Broadcasting**
Combines kernels + parallelism + tensor management. Supports **NumPy-style broadcasting**.

**Broadcasting Example:**
```delphi
// [32, 10] + [10] = [32, 10]  (bias broadcast across batch)
// [8, 1] + [1, 8] = [8, 8]    (outer product style)
TOps.Add(Arena, A, B, Out);     // Automatically broadcasts if shapes compatible
```

**Batch MatMul:**
```delphi
// [Batch, M, K] @ [Batch, K, N] -> [Batch, M, N]
// Processes each batch independently with parallel inner loops
TOps.MatMul(Arena, A, B, Out);  // Works with 2D, 3D, or higher
```

**Key Operations:**
- **`MatMul`**: Batched matrix multiplication. Transposes B for cache-friendly access, parallelizes rows, uses SIMD dot product.
- **`Add` / `Mul`**: Element-wise with broadcasting support.
- **`ReLU` / `LeakyReLU` / `Sigmoid`**: Activation functions.
- **`MSE` / `CrossEntropy`**: Loss functions.
- **`*Backward`**: Gradient computation with broadcasting-aware reduction.

---

#### **ML.Graph.pas** - Computation Graph & Autograd
The "brain" of NeuralDelphi. Implements automatic differentiation by building a computation graph.

**Key Concepts:**

**1. Computation Graph:**
Each operation creates a `TNode` that records:
- Operation type (`opMatMul`, `opReLU`, etc.)
- Input node indices (parents)
- Output tensor
- Whether gradients are needed

**2. Forward Pass:**
Operations are executed immediately as you build the graph:
```delphi
W := Graph.Param([8, 2]);        // Creates param node with shape [8, 2]
X := Graph.Input([2, 1]);         // Creates input placeholder
H := Graph.MatMul(W, X);         // Executes MatMul, creates node
A := Graph.LeakyReLU(H);         // Executes LeakyReLU, creates node
```

**3. Backward Pass:**
Traverses graph in reverse, computing gradients using chain rule:
```delphi
Graph.Backward(LossNode);  // Computes gradients for all nodes requiring them
```

**4. Memory Architecture:**
- **`MarkParamsEnd()`**: Called after all `Param()` calls. Marks the boundary between persistent parameters and temporary activations.
- **`ResetActivations()`**: Resets arena to param savepoint. Wipes activations but keeps parameters intact.

**Key Methods:**
- **`Param(Shape)`**: Creates trainable parameter with N-D shape. Pre-allocates gradients.
- **`Input(Shape)`**: Creates input placeholder with N-D shape.
- **`Step(LearningRate, GradClip)`**: Updates parameters with gradient clipping.

---

## 🎯 XOR Demo

The included demo (`XOR_Demo.dpr`) trains a neural network to learn the XOR function with a **real-time visual heatmap** and **interactive control panel**.

### Interactive Controls

The demo includes a control panel for experimenting with hyperparameters:

| Control | Default | Description |
|---------|---------|-------------|
| **Learning Rate** | 0.5 | How fast to learn. Higher = faster but may overshoot |
| **Hidden Neurons** | 16 | Network capacity. More = better fit, slower training |
| **Grad Clip** | 5.0 | Maximum gradient magnitude. Prevents exploding gradients |
| **Start/Stop** | — | Toggle training on/off |
| **Reset Network** | — | Reinitialize with random weights |

**Tips:**
- Learning rate 0.5-1.0 works well for XOR
- 8-32 hidden neurons is plenty
- Watch the loss decrease and decision boundary sharpen in real-time

### The XOR Problem

XOR (exclusive OR) is a classic non-linearly separable problem that requires a hidden layer:

| Input A | Input B | Expected Output | Visual |
|---------|---------|-----------------|--------|
| 0       | 0       | 0               | Red    |
| 0       | 1       | 1               | Blue   |
| 1       | 0       | 1               | Blue   |
| 1       | 1       | 0               | Red    |

### Network Architecture

```
Input(2) → MatMul(W1: Hx2) → Add(B1: Hx1) → LeakyReLU → 
          MatMul(W2: 1xH) → Add(B2: 1x1) → Sigmoid → Output(1)
```

Where H = Hidden Neurons (configurable via UI)

**Visualization:**
- Heatmap shows network's prediction for every (x, y) coordinate
- Red = predicts 0, Blue = predicts 1
- Corners show actual XOR truth table
- Updates in real-time as network learns

## 🔧 Building

### Requirements
- **RAD Studio** 11+ (Delphi)
- **Platform:** Windows x64 (for SIMD assembly)

### Steps
1. Open `XOR_Demo.dpr` in RAD Studio
2. Select **64-bit Windows** target
3. Build and Run (F9)

> **Note:** 32-bit builds use scalar fallbacks (no SIMD)

## 📁 Project Structure

```
NeuralDelphi/
├── ML.Arena.pas      # Memory arena allocator
├── ML.Tensor.pas     # N-D tensor with Shape/Strides
├── ML.Ops.pas        # Math operations + SIMD + broadcasting
├── ML.Graph.pas      # Computation graph + autograd
├── XOR_Demo.dpr      # Interactive XOR visualization
├── LICENSE           # MIT License
└── README.md
```

## 📊 Supported Operations

### Core Operations

| Operation | Description | Broadcasting | Batched |
|-----------|-------------|--------------|---------|
| **MatMul** | Matrix multiplication `C = A @ B` | No | ✅ 3D+ |
| **Add** | Element-wise addition `C = A + B` | ✅ | ✅ |
| **Mul** | Element-wise multiplication `C = A * B` | ✅ | ✅ |

### Activation Functions

| Operation | Formula | Use Case |
|-----------|---------|----------|
| **ReLU** | `f(x) = max(0, x)` | Standard activation |
| **LeakyReLU** | `f(x) = max(αx, x)` | Prevents dying ReLU |
| **Sigmoid** | `f(x) = 1/(1+e^(-x))` | Binary classification output |
| **Tanh** | `f(x) = tanh(x)` | Centered around 0 |
| **Softmax** | `f(x_i) = e^(x_i) / Σe^(x_j)` | Multi-class output |

### Loss Functions

| Operation | Formula | Use Case |
|-----------|---------|----------|
| **MSE** | `L = (1/n)Σ(pred - target)²` | Regression |
| **CrossEntropy** | `L = -Σ(target * log(pred))` | Classification |
| **SoftmaxCrossEntropy** | Combined softmax + CE | Numerically stable classification |

## 🎛️ Broadcasting Rules

NeuralDelphi follows NumPy broadcasting semantics:

1. **Align shapes from the right**: `[32, 10]` and `[10]` align as `[32, 10]` + `[1, 10]`
2. **Dimensions must match or be 1**: `[8, 1]` + `[1, 8]` → `[8, 8]`
3. **Result shape is element-wise max**: `[32, 1, 64]` + `[1, 8, 64]` → `[32, 8, 64]`

**Helper Functions:**
```delphi
if CanBroadcast(ShapeA, ShapeB) then
  OutShape := BroadcastShapes(ShapeA, ShapeB);
  
// During element-wise ops, use BroadcastIndex to map output → input indices
```

## 🔢 Batch Matrix Multiplication

MatMul supports N-dimensional tensors where the last two dimensions are the matrix dimensions:

```delphi
// A: [Batch, M, K]  ×  B: [Batch, K, N]  →  Out: [Batch, M, N]
// Each batch is multiplied independently

A := TTensor.Create(Arena, [32, 64, 128]);   // 32 matrices of 64x128
B := TTensor.Create(Arena, [32, 128, 256]);  // 32 matrices of 128x256
Out := TTensor.Create(Arena, [32, 64, 256]); // Result: 32 matrices of 64x256

TOps.MatMul(Arena, A, B, Out);  // Parallel across batches and rows
```

## 🧠 Performance Optimizations

**1. SIMD (Single Instruction, Multiple Data)**
- Processes 4 floats simultaneously using SSE registers
- `DotProduct`: ~4x faster than scalar code

**2. Cache-Friendly Matrix Multiplication**
- Transposes matrix B before multiplication
- Block-based transpose (8x8 blocks) for L1 cache efficiency
- Reduces cache misses by ~80%

**3. Thread Pool Parallelization**
- Uses Delphi RTL's `TParallel.For` (reuses threads)
- Threshold: only parallelizes if >256 elements

**4. Gradient Clipping**
- Configurable max gradient norm
- Prevents exploding gradients during training

**5. Persistent Parameters**
- Parameters allocated once, gradients pre-allocated
- `ResetActivations()` only wipes temporary tensors

## 🚧 Roadmap

- [x] N-dimensional tensor support
- [x] Broadcasting for element-wise ops
- [x] Batch matrix multiplication
- [x] Interactive hyperparameter tuning
- [ ] Model save/load persistence
- [ ] Conv2D operations
- [ ] MNIST demo
- [ ] AVX-512 kernels
- [ ] GPU acceleration (CUDA/OpenCL)

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional layer types (Conv2D, BatchNorm, Dropout)
- Performance optimizations
- More demos and examples
- Documentation

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with ❤️ in Delphi</i>
</p>
