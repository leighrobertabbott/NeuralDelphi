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

The included demo trains a neural network to learn the XOR function in real-time:

| Input A | Input B | Expected Output |
|---------|---------|-----------------|
| 0       | 0       | 0 (Red)         |
| 0       | 1       | 1 (Blue)        |
| 1       | 0       | 1 (Blue)        |
| 1       | 1       | 0 (Red)         |

**Network Architecture:**
```
Input(2) → Dense(8) → LeakyReLU → Dense(1) → Sigmoid → Output
```

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

## 🧠 Core Concepts

### Arena Allocation
Traditional allocators are too slow for ML training loops. The arena pre-allocates a contiguous memory block:

```delphi
Arena := TArena.Create(256);  // 256MB block
Ptr := Arena.Alloc(1000);     // O(1) allocation
Arena.Reset;                   // O(1) free everything
```

### Computation Graph
Operations are recorded on a "tape" for automatic differentiation:

```delphi
W := Graph.Param(8, 2);       // Trainable weights
X := Graph.Input(2, 1);       // Input placeholder
H := Graph.MatMul(W, X);      // Forward: H = W @ X
A := Graph.LeakyReLU(H);      // Forward: A = LeakyReLU(H)

Graph.Backward(LossNode);      // Backward: compute all gradients
Graph.Step(0.01);              // Update: W -= lr * dW
```

### SIMD Kernels
Critical operations use hand-written x64 assembly:

```delphi
// SSE dot product - 4 floats at once
class function TKernels.DotProduct(const PtrA, PtrB: PSingle; K: Integer): Single;
asm
  XORPS XMM7, XMM7       // Accumulator = 0
@Loop:
  MOVUPS XMM0, [RAX]     // Load 4 floats from A
  MOVUPS XMM1, [RCX]     // Load 4 floats from B
  MULPS  XMM0, XMM1      // Multiply packed
  ADDPS  XMM7, XMM0      // Accumulate
  ...
end;
```

## 📊 Supported Operations

| Category | Operations |
|----------|------------|
| **Core** | MatMul, Add, Mul |
| **Activations** | ReLU, LeakyReLU, Sigmoid, Tanh, Softmax |
| **Loss** | MSE, CrossEntropy, SoftmaxCrossEntropy |

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

