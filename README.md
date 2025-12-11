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

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NeuralDelphi                              │
├─────────────────────────────────────────────────────────────────┤
│  ML.Arena    │  Linear memory allocator (zero GC overhead)       │
│  ML.Tensor   │  Lightweight tensor views into arena              │
│  ML.Ops      │  SIMD kernels + parallel operations               │
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

#### **ML.Tensor.pas** - Tensor Abstraction
A lightweight `record` (not a class!) that acts as a **view** into the arena. Think of it like a pointer + metadata.

**Key Fields:**
- **`DataPtr: TMemPtr`**: Index into arena where tensor data lives
- **`GradPtr: TMemPtr`**: Index for gradients (allocated on-demand during backward pass)
- **`Rows, Cols: Integer`**: Shape information
- **`RequiresGrad: Boolean`**: Whether this tensor needs gradients computed

**Key Methods:**
- **`RawData(Arena)`**: Returns `PSingle` pointer for direct memory access (used by SIMD kernels)
- **`RawGrad(Arena)`**: Returns gradient pointer, or `nil` if not allocated
- **`CreateTensor()`**: Factory method that allocates memory in arena and returns a tensor view

**Why Records, Not Classes:**
- Zero heap allocation overhead
- Value semantics (can copy freely)
- Cache-friendly (all data in one contiguous block)

**Example:**
```delphi
var
  T: TTensor;
begin
  T := TTensor.CreateTensor(Arena, 8, 2, True);  // 8x2 matrix, needs gradients
  // T.DataPtr now points to 16 floats in the arena
  // T.GradPtr = -1 (not allocated yet)
end;
```

---

#### **ML.Ops.pas** - Mathematical Operations
Contains three layers: **Pure ASM Kernels**, **Parallel Execution**, and **High-Level Tensor Ops**.

**1. `TKernels` - Pure Assembly Math Kernels**
Hand-written x64 SSE assembly for maximum performance. These are **stateless** functions that operate on raw pointers.

- **`DotProduct(A, B, Count)`**: SIMD dot product using `MOVUPS`, `MULPS`, `ADDPS`, `HADDPS`. Processes 4 floats at once.
- **`VectorAdd(A, B, Out, Count)`**: Element-wise addition with SSE `ADDPS`.
- **`VectorMul(A, B, Out, Count)`**: Element-wise multiplication with SSE `MULPS`.
- **`Transpose(Src, Dst, Rows, Cols)`**: Block-based matrix transpose (8x8 blocks) for cache efficiency.

**Why Separate Kernels:**
- Can't use inline ASM inside anonymous methods (Delphi limitation)
- Kernels are reusable across different operations
- Easy to optimize independently

**2. `TMLParallel` - Thread Pool Wrapper**
Wraps `System.Threading.TParallel.For` with a threshold check. Only parallelizes if workload is substantial (>256 elements) to avoid overhead.

**3. `TOps` - High-Level Tensor Operations**
Combines kernels + parallelism + tensor management. Each operation:
- Validates tensor shapes
- Allocates output tensor in arena
- Calls appropriate kernels (SIMD or scalar)
- Parallelizes outer loops when beneficial

**Key Operations:**
- **`MatMul`**: Matrix multiplication. Transposes B for cache-friendly access, parallelizes rows, uses SIMD dot product for inner loop.
- **`Add` / `Mul`**: Element-wise operations using SIMD kernels.
- **`ReLU` / `LeakyReLU` / `Sigmoid`**: Activation functions (scalar, but could be SIMD-optimized).
- **`MSE` / `CrossEntropy`**: Loss functions.
- **`*Backward`**: Gradient computation for each operation (chain rule).

**Example:**
```delphi
// Forward pass
TOps.MatMul(Arena, W, X, Out);        // Out = W @ X (uses SIMD + parallel)
TOps.LeakyReLU(Arena, Out, Activated); // Activated = LeakyReLU(Out)

// Backward pass
TOps.MatMulBackward(Arena, W, X, OutGrad, WGrad, XGrad);  // Computes dW, dX
```

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
W := Graph.Param(8, 2);        // Creates param node, allocates memory
X := Graph.Input(2, 1);         // Creates input placeholder
H := Graph.MatMul(W, X);       // Executes MatMul, creates node
A := Graph.LeakyReLU(H);       // Executes LeakyReLU, creates node
```

**3. Backward Pass:**
Traverses graph in reverse, computing gradients using chain rule:
```delphi
Graph.Backward(LossNode);  // Computes gradients for all nodes requiring them
```

**4. Memory Architecture:**
- **`MarkParamsEnd()`**: Called after all `Param()` calls. Marks the boundary between persistent parameters and temporary activations.
- **`ResetActivations()`**: Resets arena to param savepoint. Wipes activations but keeps parameters intact. This is the key optimization that eliminates save/restore overhead.

**Key Methods:**
- **`Param(Rows, Cols)`**: Creates trainable parameter. Pre-allocates gradients so they persist across `ResetActivations()`.
- **`Input(Rows, Cols)`**: Creates input placeholder (value set later via `SetInputValue()`).
- **`MatMul(A, B)`**: Creates matrix multiplication node, executes forward pass.
- **`Backward(LossNode)`**: Computes gradients for all nodes that need them.
- **`Step(LearningRate)`**: Updates parameters: `W -= lr * dW`.

**Example:**
```delphi
// Build network (once)
W := Graph.Param(8, 2);
B := Graph.Param(8, 1);
Graph.MarkParamsEnd();  // Mark: everything before this is persistent

// Training loop
for i := 1 to 1000 do
begin
  Graph.ResetActivations();  // Wipe activations, keep W and B
  X := Graph.Input(2, 1);
  H := Graph.MatMul(W, X);
  H := Graph.Add(H, B);
  Y := Graph.LeakyReLU(H);
  Loss := Graph.MSE(Y, Target);
  
  Graph.ZeroGrad();      // Zero param gradients
  Graph.Backward(Loss);  // Compute gradients
  Graph.Step(0.01);      // Update: W -= 0.01 * dW
end;
```

### Memory Model

```
┌───────────────────────────────────────┬─────────────────────────┐
│         PERSISTENT PARAMS             │  TEMPORARY ACTIVATIONS  │
│  (weights, biases, gradients)         │  (reset each iteration) │
│                                       │                         │
│  ← MarkParamsEnd()                    │  ← ResetActivations()   │
└───────────────────────────────────────┴─────────────────────────┘
```

## 🎯 XOR Demo

The included demo (`XOR_Demo.dpr`) trains a neural network to learn the XOR function in real-time with a visual heatmap.

### The XOR Problem

XOR (exclusive OR) is a classic non-linearly separable problem that requires a hidden layer:

| Input A | Input B | Expected Output | Visual |
|---------|---------|-----------------|--------|
| 0       | 0       | 0               | Red    |
| 0       | 1       | 1               | Blue   |
| 1       | 0       | 1               | Blue   |
| 1       | 1       | 0               | Red    |

**Why XOR is Hard:**
- A single-layer perceptron cannot learn XOR (it's not linearly separable)
- Requires at least one hidden layer with non-linear activation
- Tests that the network can learn non-linear decision boundaries

### Network Architecture

```
Input(2) → MatMul(W1: 8x2) → Add(B1: 8x1) → LeakyReLU → 
          MatMul(W2: 1x8) → Add(B2: 1x1) → Sigmoid → Output(1)
```

**Layer Breakdown:**
- **Input Layer**: 2 neurons (XOR inputs)
- **Hidden Layer**: 8 neurons with LeakyReLU activation
  - `W1`: 8×2 weight matrix (16 parameters)
  - `B1`: 8×1 bias vector (8 parameters)
- **Output Layer**: 1 neuron with Sigmoid activation (probability)
  - `W2`: 1×8 weight matrix (8 parameters)
  - `B2`: 1×1 bias scalar (1 parameter)
- **Total Parameters**: 33 trainable weights/biases

**Training:**
- **Learning Rate**: 0.5 (higher for faster convergence on small dataset)
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: SGD (Stochastic Gradient Descent) - `W -= lr * dW`
- **Dataset**: 4 samples (all XOR combinations), repeated each epoch

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
├── ML.Tensor.pas     # Tensor record (view into arena)
├── ML.Ops.pas        # Math operations + SIMD kernels
├── ML.Graph.pas      # Computation graph + autograd
├── XOR_Demo.dpr      # Interactive XOR visualization
├── LICENSE           # MIT License
└── README.md
```

## 🧠 How It All Works Together

### Training Step Flow

Here's what happens during a single training iteration:

```
1. ResetActivations()
   └─> Arena.Restore(ParamSavePoint)
       └─> Wipes temporary tensors, keeps W, B, dW, dB

2. Build Forward Pass
   └─> Graph.Input()     → Allocates input tensor
   └─> Graph.MatMul()    → Calls TOps.MatMul()
       └─> Transposes B for cache efficiency
       └─> Parallel.ForEach(row) → TKernels.DotProduct() (SIMD)
   └─> Graph.LeakyReLU() → Element-wise activation
   └─> Graph.MSE()       → Computes loss

3. Backward Pass
   └─> Graph.ZeroGrad()  → Zeros param gradients (they persist!)
   └─> Graph.Backward()  → Traverses graph in reverse
       └─> For each node: calls TOps.*Backward()
           └─> Uses chain rule: dA = dOut * dOut/dA
           └─> Accumulates gradients: dW += gradient

4. Parameter Update
   └─> Graph.Step(lr)    → W -= lr * dW, B -= lr * dB
```

### Memory Layout Example

After `MarkParamsEnd()`, the arena looks like:

```
┌─────────────────────────────────────────────────────────────┐
│ Offset │ Size │ Content                                      │
├────────┼──────┼──────────────────────────────────────────────┤
│ 0      │ 16   │ W1[8x2] data                                │
│ 16     │ 8    │ B1[8x1] data                                │
│ 24     │ 8    │ W2[1x8] data                                │
│ 32     │ 1    │ B2[1x1] data                                │
│ 33     │ 16   │ W1 gradients (pre-allocated)                │
│ 49     │ 8    │ B1 gradients                                │
│ 57     │ 8    │ W2 gradients                                │
│ 65     │ 1    │ B2 gradients                                │
├────────┼──────┼──────────────────────────────────────────────┤
│ 66     │ ← FParamSavePoint (MarkParamsEnd() saved this)      │
│        │      │                                              │
│        │      │ ← ResetActivations() restores to here        │
│        │      │                                              │
│ 66+    │ 2    │ Input[2x1] (temporary)                      │
│ 68+    │ 8    │ Hidden[8x1] (temporary)                      │
│ 76+    │ 1    │ Output[1x1] (temporary)                     │
│ 77+    │ 1    │ Loss[1x1] (temporary)                       │
│        │      │ ... gradients for activations ...            │
└─────────────────────────────────────────────────────────────┘
```

### Performance Optimizations

**1. SIMD (Single Instruction, Multiple Data)**
- Processes 4 floats simultaneously using SSE registers
- `DotProduct`: ~4x faster than scalar code
- `VectorAdd`/`VectorMul`: ~4x faster for element-wise ops

**2. Cache-Friendly Matrix Multiplication**
- Transposes matrix B before multiplication
- Accesses memory sequentially (row-major)
- Reduces cache misses by ~80% vs naive implementation

**3. Thread Pool Parallelization**
- Uses Delphi RTL's `TParallel.For` (reuses threads)
- Parallelizes outer loops (rows of output matrix)
- Threshold: only parallelizes if >256 elements (avoids overhead)

**4. Zero-Copy Operations**
- Tensors are views, not copies
- Operations write directly to arena
- No intermediate allocations

**5. Persistent Parameters**
- Parameters allocated once, gradients pre-allocated
- `ResetActivations()` only wipes temporary tensors
- Eliminates save/restore overhead (was ~100 array operations per iteration)

## 📊 Supported Operations

### Core Operations

| Operation | Description | Forward | Backward |
|-----------|-------------|---------|----------|
| **MatMul** | Matrix multiplication `C = A @ B` | SIMD dot product, parallel rows, transposed B for cache | `dA = dC @ B^T`, `dB = A^T @ dC` |
| **Add** | Element-wise addition `C = A + B` | SIMD `ADDPS` | `dA = dC`, `dB = dC` (broadcast) |
| **Mul** | Element-wise multiplication `C = A * B` | SIMD `MULPS` | `dA = dC * B`, `dB = dC * A` |

### Activation Functions

| Operation | Formula | Use Case |
|-----------|---------|----------|
| **ReLU** | `f(x) = max(0, x)` | Standard activation, fast, can cause "dying ReLU" |
| **LeakyReLU** | `f(x) = max(αx, x)` where α=0.01 | Prevents dying ReLU, allows small negative gradients |
| **Sigmoid** | `f(x) = 1/(1+e^(-x))` | Output layer for binary classification, smooth gradient |
| **Tanh** | `f(x) = tanh(x)` | Centered around 0, stronger gradients than sigmoid |
| **Softmax** | `f(x_i) = e^(x_i) / Σe^(x_j)` | Multi-class classification, outputs probability distribution |

### Loss Functions

| Operation | Formula | Use Case |
|-----------|---------|----------|
| **MSE** | `L = (1/n)Σ(pred - target)²` | Regression tasks, smooth gradients |
| **CrossEntropy** | `L = -Σ(target * log(pred))` | Classification, but numerically unstable |
| **SoftmaxCrossEntropy** | Combined softmax + cross-entropy | **Recommended** for classification. Numerically stable, simple gradient: `pred - target` |

### Backward Operations

All operations have corresponding `*Backward` methods that compute gradients using the **chain rule**:

- **Chain Rule**: If `y = f(x)` and `z = g(y)`, then `dz/dx = dz/dy * dy/dx`
- **Accumulation**: Gradients accumulate (add) when a tensor is used in multiple operations
- **Lazy Allocation**: Gradients are only allocated when `RequiresGrad = True` and `Backward()` is called

## 🎓 Design Decisions & Trade-offs

### Why Records Instead of Classes?
- **Zero heap allocation**: Records are value types, stored on stack or inline
- **Cache-friendly**: All tensor metadata in one small struct
- **No vtable overhead**: Direct method calls, no virtual dispatch
- **Trade-off**: Can't use inheritance, but we don't need it

### Why Arena Instead of Standard Allocator?
- **Speed**: O(1) allocation vs O(log n) for heap allocators
- **No fragmentation**: Contiguous memory, perfect for SIMD
- **Predictable performance**: No GC pauses, no heap walks
- **Trade-off**: Can't free individual tensors (but we don't need to in training loops)

### Why Inline Assembly Instead of Compiler Intrinsics?
- **Delphi's SIMD support is limited**: No direct access to SSE intrinsics like C++
- **Full control**: We can optimize exactly how we want
- **Portability**: x64 gets SIMD, x86 gets scalar fallback (same code path)
- **Trade-off**: Platform-specific, but ML frameworks are typically platform-specific anyway

### Why Separate Kernels from Operations?
- **Delphi limitation**: Can't use inline ASM inside anonymous methods
- **Reusability**: Kernels can be called from anywhere
- **Testability**: Can unit test kernels independently
- **Trade-off**: Slight indirection, but negligible performance impact

### Why ResetActivations Instead of Full Reset?
- **Performance**: Eliminates ~100 array copy operations per iteration
- **Memory efficiency**: Parameters stay in place, no save/restore
- **Simplicity**: No need to track parameter arrays separately
- **Trade-off**: Slightly more complex arena management, but worth it

### Why Pre-allocate Parameter Gradients?
- **Persistence**: Gradients must survive `ResetActivations()`
- **Performance**: Allocate once, reuse forever
- **Simplicity**: No need to check if gradient exists during backward pass
- **Trade-off**: Uses more memory upfront, but negligible for typical networks

## 🚧 Roadmap

- [ ] Model save/load persistence
- [ ] Batch training support
- [ ] Conv2D operations
- [ ] MNIST demo
- [ ] AVX-512 kernels
- [ ] GPU acceleration (CUDA/OpenCL)

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional layer types
- Performance optimizations
- More demos and examples
- Documentation

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with ❤️ in Delphi</i>
</p>

