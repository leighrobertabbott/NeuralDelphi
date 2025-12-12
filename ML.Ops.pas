unit ML.Ops;

{$R-} // Disable range checking for performance

interface

uses
  System.Math,
  System.SysUtils,
  System.Classes,
  System.Threading,
  {$IFDEF MSWINDOWS}
  Winapi.Windows,
  {$ENDIF}
  ML.Arena,
  ML.Tensor;

type
  PSingleArray = ^TSingleArray;
  TSingleArray = array[0..MaxInt div SizeOf(Single) - 1] of Single;

  // ==========================================================================
  // MATH KERNELS - Pure ASM, no closures, maximum performance
  // ==========================================================================
  TKernels = class
  public
    // SIMD Dot Product: returns sum of A[i]*B[i] for i=0..Count-1
    class function DotProduct(A, B: PSingle; Count: Integer): Single; static;
    // SIMD Vector Add: Out[i] = A[i] + B[i]
    class procedure VectorAdd(A, B, Output: PSingle; Count: Integer); static;
    // SIMD Vector Mul: Out[i] = A[i] * B[i]
    class procedure VectorMul(A, B, Output: PSingle; Count: Integer); static;
    // Transpose matrix: Dst[j,i] = Src[i,j]
    class procedure Transpose(Src, Dst: PSingleArray; Rows, Cols: Integer); static;
    
    // AVX-512 versions (x64 only)
    {$IFDEF CPUX64}
    class function DotProductAVX512(A, B: PSingle; Count: Integer): Single; static;
    class procedure VectorAddAVX512(A, B, Output: PSingle; Count: Integer); static;
    class procedure VectorMulAVX512(A, B, Output: PSingle; Count: Integer); static;
    // SSE versions (x64 fallback when AVX-512 not available)
    class function DotProductSSE(A, B: PSingle; Count: Integer): Single; static;
    class procedure VectorAddSSE(A, B, Output: PSingle; Count: Integer); static;
    class procedure VectorMulSSE(A, B, Output: PSingle; Count: Integer); static;
    {$ENDIF}
  end;

  // ==========================================================================
  // PARALLEL EXECUTION - Uses RTL Thread Pool
  // ==========================================================================
  TMLParallel = class
  public
    // ForEach with operation-specific threshold
    // Threshold determines when to use parallel processing vs sequential
    // Lower threshold = more aggressive parallelization (for heavy operations like Conv2D)
    // Higher threshold = more conservative (for light operations like ReLU)
    class procedure ForEach(AStart, AEnd: Integer; const AProc: TProc<Integer>; 
      Threshold: Integer = 256); static;
  end;

  // ==========================================================================
  // CPU FEATURE DETECTION
  // ==========================================================================
  TCPUFeatures = record
    HasSSE: Boolean;
    HasAVX: Boolean;
    HasAVX512: Boolean;
  end;

  // ==========================================================================
  // TENSOR OPERATIONS - High-level API combining Kernels + Parallelism
  // ==========================================================================
  TOps = class
    // Core operations
    class procedure MatMul(Arena: TArena; const A, B: TTensor; var OutT: TTensor); static;
    class procedure Add(Arena: TArena; const A, B: TTensor; var OutT: TTensor); static;
    class procedure Mul(Arena: TArena; const A, B: TTensor; var OutT: TTensor); static;
    class procedure ReLU(Arena: TArena; const A: TTensor; var OutT: TTensor); static;
    class procedure LeakyReLU(Arena: TArena; const A: TTensor; var OutT: TTensor; Alpha: Single = 0.01); static;
    
    // Extended operations
    class procedure Sigmoid(Arena: TArena; const A: TTensor; var OutT: TTensor); static;
    class procedure Tanh(Arena: TArena; const A: TTensor; var OutT: TTensor); static;
    class procedure Softmax(Arena: TArena; const A: TTensor; var OutT: TTensor); static;
    
    // Loss functions
    class function MSE(Arena: TArena; const Pred, Target: TTensor): Single; static;
    class function CrossEntropy(Arena: TArena; const Pred, Target: TTensor): Single; static;
    // Combined Softmax + CrossEntropy (numerically stable, simple gradient: Pred - Target)
    class function SoftmaxCrossEntropy(Arena: TArena; const Logits, Target: TTensor;
      var SoftmaxOut: TTensor): Single; static;
    
    // Backward operations
    class procedure MatMulBackward(Arena: TArena; const A, B, OutGrad: TTensor;
      var AGrad, BGrad: TTensor); static;
    class procedure AddBackward(Arena: TArena; const OutGrad: TTensor;
      var AGrad, BGrad: TTensor); static;
    class procedure MulBackward(Arena: TArena; const A, B, OutGrad: TTensor;
      var AGrad, BGrad: TTensor); static;
    class procedure ReLUBackward(Arena: TArena; const A, OutGrad: TTensor;
      var AGrad: TTensor); static;
    class procedure LeakyReLUBackward(Arena: TArena; const A, OutGrad: TTensor;
      var AGrad: TTensor; Alpha: Single = 0.01); static;
    class procedure SigmoidBackward(Arena: TArena; const Out, OutGrad: TTensor;
      var AGrad: TTensor); static;
    class procedure TanhBackward(Arena: TArena; const Out, OutGrad: TTensor;
      var AGrad: TTensor); static;
    class procedure SoftmaxBackward(Arena: TArena; const Out, OutGrad: TTensor;
      var AGrad: TTensor); static;
    // Combined backward: gradient is simply (Softmax - Target)
    class procedure SoftmaxCrossEntropyBackward(Arena: TArena; const SoftmaxOut, Target, OutGrad: TTensor;
      var AGrad: TTensor); static;
    
    // Convolution operations
    class procedure Conv2D(Arena: TArena; const Input, Weight: TTensor;
      Padding, Stride: Integer; var OutT: TTensor); static;
    class procedure Conv2DBackward(Arena: TArena; const Input, Weight, OutGrad: TTensor;
      Padding, Stride: Integer; var InputGrad, WeightGrad: TTensor); static;
    
    // Batch Normalization
    class procedure BatchNorm(Arena: TArena; const Input, Gamma, Beta: TTensor;
      Epsilon: Single; IsTraining: Boolean; var RunningMean, RunningVar: TArray<Single>;
      Momentum: Single; var OutT: TTensor); static;
    class procedure BatchNormBackward(Arena: TArena; const Input, OutGrad, Gamma: TTensor;
      Epsilon: Single; var InputGrad, GammaGrad, BetaGrad: TTensor); static;
    
    // Fused operations
    class procedure MatMulAdd(Arena: TArena; const A, B, Bias: TTensor; var OutT: TTensor); static;
    class procedure MatMulAddBackward(Arena: TArena; const A, B, Bias, OutGrad: TTensor;
      var AGrad, BGrad, BiasGrad: TTensor); static;
    class procedure AddReLU(Arena: TArena; const A, B: TTensor; var OutT: TTensor); static;
    class procedure AddReLUBackward(Arena: TArena; const A, B, OutGrad: TTensor;
      var AGrad, BGrad: TTensor); static;
    class procedure MatMulReLU(Arena: TArena; const A, B: TTensor; var OutT: TTensor); static;
    class procedure MatMulReLUBackward(Arena: TArena; const A, B, OutGrad: TTensor;
      var AGrad, BGrad: TTensor); static;
    
    // Pooling operations
    class procedure MaxPool2D(Arena: TArena; const Input: TTensor;
      PoolSize, Stride: Integer; var OutT: TTensor; var MaxIndices: TArray<Integer>); static;
    class procedure MaxPool2DBackward(Arena: TArena; const OutGrad: TTensor;
      const MaxIndices: TArray<Integer>; var InputGrad: TTensor); static;
    
    // Regularization
    class procedure Dropout(Arena: TArena; const Input: TTensor;
      DropRate: Single; IsTraining: Boolean; var OutT: TTensor; var Mask: TArray<Byte>); static;
    class procedure DropoutBackward(Arena: TArena; const OutGrad: TTensor;
      const Mask: TArray<Byte>; DropRate: Single; var AGrad: TTensor); static;
  end;

implementation

// ============================================================================
// CPU FEATURE DETECTION VARIABLE (declared early for use in methods)
// ============================================================================

var
  CPUFeatures: TCPUFeatures;

// ============================================================================
// MATH KERNELS IMPLEMENTATION - Pure Pascal (inline ASM not available in x64 Delphi)
// ============================================================================

class function TKernels.DotProduct(A, B: PSingle; Count: Integer): Single;
var
  i: Integer;
  Sum: Single;
  PA, PB: PSingleArray;
begin
  PA := PSingleArray(A);
  PB := PSingleArray(B);
  Sum := 0;
  for i := 0 to Count - 1 do
    Sum := Sum + PA^[i] * PB^[i];
  Result := Sum;
end;

// Pascal implementation for all platforms
class procedure TKernels.VectorAdd(A, B, Output: PSingle; Count: Integer);
var
  i: Integer;
  PA, PB, POut: PSingleArray;
begin
  PA := PSingleArray(A);
  PB := PSingleArray(B);
  POut := PSingleArray(Output);
  for i := 0 to Count - 1 do
    POut^[i] := PA^[i] + PB^[i];
end;

// Pascal implementation for all platforms
class procedure TKernels.VectorMul(A, B, Output: PSingle; Count: Integer);
var
  i: Integer;
  PA, PB, POut: PSingleArray;
begin
  PA := PSingleArray(A);
  PB := PSingleArray(B);
  POut := PSingleArray(Output);
  for i := 0 to Count - 1 do
    POut^[i] := PA^[i] * PB^[i];
end;

class procedure TKernels.Transpose(Src, Dst: PSingleArray; Rows, Cols: Integer);
const
  BLOCK_SIZE = 8;  // 8x8 blocks stay in L1 cache
var
  i, j, ii, jj: Integer;
  iEnd, jEnd: Integer;
begin
  // Block-based transpose for cache efficiency
  // Process 8x8 blocks to maximize L1 cache hits
  ii := 0;
  while ii < Rows do
  begin
    iEnd := ii + BLOCK_SIZE;
    if iEnd > Rows then iEnd := Rows;
    
    jj := 0;
    while jj < Cols do
    begin
      jEnd := jj + BLOCK_SIZE;
      if jEnd > Cols then jEnd := Cols;
      
      // Transpose this block
      for i := ii to iEnd - 1 do
        for j := jj to jEnd - 1 do
          Dst^[j * Rows + i] := Src^[i * Cols + j];
          
      Inc(jj, BLOCK_SIZE);
    end;
    Inc(ii, BLOCK_SIZE);
  end;
end;

// ============================================================================
// PARALLEL EXECUTION
// ============================================================================

class procedure TMLParallel.ForEach(AStart, AEnd: Integer; const AProc: TProc<Integer>; 
  Threshold: Integer);
var
  i: Integer;
begin
  if AStart > AEnd then Exit;
  
  // TParallel.For has significant overhead (~1-5ms per call)
  // Only use for genuinely large workloads
  // Threshold is operation-specific:
  //   - Conv2D: 32 (heavy work per position: nested loops, memory access)
  //   - MatMul: 64-128 (moderate work per row: SIMD dot products)
  //   - Element-wise ops: 1000+ (light work: simple arithmetic)
  if (AEnd - AStart) < Threshold then
  begin
    for i := AStart to AEnd do
      AProc(i);
  end
  else
    TParallel.For(AStart, AEnd, AProc);
end;

// ============================================================================
// TOps IMPLEMENTATION
// ============================================================================

class procedure TOps.MatMul(Arena: TArena; const A, B: TTensor; var OutT: TTensor);
var
  PA, PB, POut, PBT: PSingleArray;
  PABatch, PBBatch, POutBatch: PSingle;
  ARows, ACols, BRows, BCols: Integer;
  SavePoint: Integer;
  TempPtr: TMemPtr;
  ANdim, BNdim: Integer;
  i, BatchIdx, BatchSize: Integer;
  MatrixSizeA, MatrixSizeB, MatrixSizeOut: Integer;
  UseAVX512, UseSSE: Boolean;
begin
  // Capture CPUFeatures for use in anonymous method
  UseAVX512 := CPUFeatures.HasAVX512;
  UseSSE := CPUFeatures.HasSSE;
  ANdim := A.NDim;
  BNdim := B.NDim;
  
  // MatMul operates on last 2 dimensions
  // A: [..., M, K], B: [..., K, N] -> Out: [..., M, N]
  // Batch dimensions (leading dims) must match
  
  if ANdim < 2 then
    raise Exception.Create('MatMul: A must have at least 2 dimensions');
  if BNdim < 2 then
    raise Exception.Create('MatMul: B must have at least 2 dimensions');
    
  // Check batch dimensions match (all but last 2)
  if ANdim > 2 then
  begin
    if BNdim <> ANdim then
      raise Exception.Create('MatMul: Batch dimension count mismatch');
    for i := 0 to ANdim - 3 do
      if A.Shape[i] <> B.Shape[i] then
        raise Exception.CreateFmt('MatMul: Batch dimension %d mismatch: %d vs %d',
          [i, A.Shape[i], B.Shape[i]]);
  end;
  
  // Get matrix dimensions (last 2 dims)
  ARows := A.Shape[ANdim - 2];
  ACols := A.Shape[ANdim - 1];
  BRows := B.Shape[BNdim - 2];
  BCols := B.Shape[BNdim - 1];
  
  if ACols <> BRows then
    raise Exception.CreateFmt('MatMul: Inner dimension mismatch: %d != %d',
      [ACols, BRows]);
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  // Compute batch size (product of all dims except last 2)
  BatchSize := 1;
  for i := 0 to ANdim - 3 do
    BatchSize := BatchSize * A.Shape[i];
  
  MatrixSizeA := ARows * ACols;
  MatrixSizeB := BRows * BCols;
  MatrixSizeOut := ARows * BCols;
  
  // Save arena state for cleanup
  SavePoint := Arena.GetSavePoint;
  
  // Allocate temp for B transpose (one matrix at a time)
  TempPtr := Arena.Alloc(BRows * BCols);
  PBT := PSingleArray(Arena.GetPtr(TempPtr));
  
  // Process each batch
  for BatchIdx := 0 to BatchSize - 1 do
  begin
    PABatch := @PA^[BatchIdx * MatrixSizeA];
    PBBatch := @PB^[BatchIdx * MatrixSizeB];
    POutBatch := @POut^[BatchIdx * MatrixSizeOut];
    
    // Transpose B for cache-friendly access
    TKernels.Transpose(PSingleArray(PBBatch), PBT, BRows, BCols);
    
    // Parallel MatMul for this batch: each row of A dot-products with all columns of B
    TMLParallel.ForEach(0, ARows - 1,
      procedure(RowA: Integer)
      var
        ColB: Integer;
        PtrA, PtrBT: PSingle;
      begin
        PtrA := @PSingleArray(PABatch)^[RowA * ACols];
        for ColB := 0 to BCols - 1 do
        begin
          PtrBT := @PBT^[ColB * BRows];
          if UseAVX512 then
            PSingleArray(POutBatch)^[RowA * BCols + ColB] := TKernels.DotProductAVX512(PtrA, PtrBT, ACols)
          else if UseSSE then
            PSingleArray(POutBatch)^[RowA * BCols + ColB] := TKernels.DotProductSSE(PtrA, PtrBT, ACols)
          else
            PSingleArray(POutBatch)^[RowA * BCols + ColB] := TKernels.DotProduct(PtrA, PtrBT, ACols);
        end;
      end, 64);  // MatMul threshold: 64 rows
  end;
  
  // Free temporary memory
  Arena.Restore(SavePoint);
end;

class procedure TOps.Add(Arena: TArena; const A, B: TTensor; var OutT: TTensor);
var
  PA, PB, POut: PSingleArray;
  OutCount: Integer;
begin
  OutCount := OutT.ElementCount;
  if OutCount = 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  // Fast path: same shape, no broadcasting needed
  if A.SameShape(B) then
  begin
    if CPUFeatures.HasAVX512 then
      TKernels.VectorAddAVX512(PSingle(PA), PSingle(PB), PSingle(POut), OutCount)
    else if CPUFeatures.HasSSE then
      TKernels.VectorAddSSE(PSingle(PA), PSingle(PB), PSingle(POut), OutCount)
    else
      TKernels.VectorAdd(PSingle(PA), PSingle(PB), PSingle(POut), OutCount);
  end
  else
  begin
    // Broadcasting path: compute indices for each output element
    // Parallelize for better performance on large tensors
    // Capture OutT.Shape and OutT.Strides into local variables for anonymous method
    var LocalOutShape := OutT.Shape;
    var LocalAStrides := A.Strides;
    var LocalBStrides := B.Strides;
    var LocalAShape := A.Shape;
    var LocalBShape := B.Shape;
    TMLParallel.ForEach(0, OutCount - 1,
      procedure(i: Integer)
      var
        LocalIdxA, LocalIdxB: Integer;
      begin
        LocalIdxA := BroadcastIndex(i, LocalOutShape, LocalAShape, LocalAStrides);
        LocalIdxB := BroadcastIndex(i, LocalOutShape, LocalBShape, LocalBStrides);
        POut^[i] := PA^[LocalIdxA] + PB^[LocalIdxB];
      end, 5000);  // Threshold: 5000 elements (light work per element, avoid overhead)
  end;
end;

class procedure TOps.Mul(Arena: TArena; const A, B: TTensor; var OutT: TTensor);
var
  PA, PB, POut: PSingleArray;
  OutCount: Integer;
begin
  OutCount := OutT.ElementCount;
  if OutCount = 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  // Fast path: same shape, no broadcasting needed
  if A.SameShape(B) then
  begin
    if CPUFeatures.HasAVX512 then
      TKernels.VectorMulAVX512(PSingle(PA), PSingle(PB), PSingle(POut), OutCount)
    else if CPUFeatures.HasSSE then
      TKernels.VectorMulSSE(PSingle(PA), PSingle(PB), PSingle(POut), OutCount)
    else
      TKernels.VectorMul(PSingle(PA), PSingle(PB), PSingle(POut), OutCount);
  end
  else
  begin
    // Broadcasting path: compute indices for each output element
    // Parallelize for better performance on large tensors
    // Capture OutT.Shape and strides into local variables for anonymous method
    var LocalOutShape := OutT.Shape;
    var LocalAStrides := A.Strides;
    var LocalBStrides := B.Strides;
    var LocalAShape := A.Shape;
    var LocalBShape := B.Shape;
    TMLParallel.ForEach(0, OutCount - 1,
      procedure(i: Integer)
      var
        LocalIdxA, LocalIdxB: Integer;
      begin
        LocalIdxA := BroadcastIndex(i, LocalOutShape, LocalAShape, LocalAStrides);
        LocalIdxB := BroadcastIndex(i, LocalOutShape, LocalBShape, LocalBStrides);
        POut^[i] := PA^[LocalIdxA] * PB^[LocalIdxB];
      end, 5000);  // Threshold: 5000 elements (light work per element, avoid overhead)
  end;
end;

class procedure TOps.ReLU(Arena: TArena; const A: TTensor; var OutT: TTensor);
var
  PA, POut: PSingleArray;
  i, Count: Integer;
begin
  Count := A.ElementCount;
  if Count = 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  for i := 0 to Count - 1 do
    if PA^[i] > 0 then
      POut^[i] := PA^[i]
    else
      POut^[i] := 0;
end;

class procedure TOps.LeakyReLU(Arena: TArena; const A: TTensor; var OutT: TTensor; Alpha: Single);
var
  PA, POut: PSingleArray;
  i, Count: Integer;
begin
  Count := A.ElementCount;
  if Count = 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  for i := 0 to Count - 1 do
    if PA^[i] > 0 then
      POut^[i] := PA^[i]
    else
      POut^[i] := Alpha * PA^[i];
end;

class procedure TOps.Sigmoid(Arena: TArena; const A: TTensor; var OutT: TTensor);
var
  PA, POut: PSingleArray;
  i, Count: Integer;
begin
  Count := A.ElementCount;
  if Count = 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  for i := 0 to Count - 1 do
    POut^[i] := 1.0 / (1.0 + Exp(-PA^[i]));
end;

class procedure TOps.Tanh(Arena: TArena; const A: TTensor; var OutT: TTensor);
var
  PA, POut: PSingleArray;
  i, Count: Integer;
begin
  Count := A.ElementCount;
  if Count = 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  for i := 0 to Count - 1 do
    POut^[i] := System.Math.Tanh(PA^[i]);
end;

class procedure TOps.Softmax(Arena: TArena; const A: TTensor; var OutT: TTensor);
var
  PA, POut: PSingleArray;
  i, Count: Integer;
  MaxVal, Sum, ExpVal: Single;
begin
  Count := A.ElementCount;
  if Count = 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  MaxVal := PA^[0];
  for i := 1 to Count - 1 do
    if PA^[i] > MaxVal then MaxVal := PA^[i];
  
  Sum := 0;
  for i := 0 to Count - 1 do
  begin
    ExpVal := Exp(PA^[i] - MaxVal);
    POut^[i] := ExpVal;
    Sum := Sum + ExpVal;
  end;
  
  if Sum > 1e-10 then
    for i := 0 to Count - 1 do
      POut^[i] := POut^[i] / Sum
  else
    for i := 0 to Count - 1 do
      POut^[i] := 1.0 / Count;
end;

class function TOps.MSE(Arena: TArena; const Pred, Target: TTensor): Single;
var
  PPred, PTarget: PSingleArray;
  i, Count: Integer;
  Diff, Sum: Single;
begin
  if not Pred.SameShape(Target) then
    raise Exception.Create('MSE: Shape mismatch');
    
  Count := Pred.ElementCount;
  if Count = 0 then Exit(0);
  
  PPred := PSingleArray(Pred.RawData(Arena));
  PTarget := PSingleArray(Target.RawData(Arena));
  
  Sum := 0;
  for i := 0 to Count - 1 do
  begin
    Diff := PPred^[i] - PTarget^[i];
    Sum := Sum + Diff * Diff;
  end;
  
  Result := Sum / Count;
end;

class function TOps.CrossEntropy(Arena: TArena; const Pred, Target: TTensor): Single;
var
  PPred, PTarget: PSingleArray;
  i, Count: Integer;
  Sum, P: Single;
begin
  if not Pred.SameShape(Target) then
    raise Exception.Create('CrossEntropy: Shape mismatch');
    
  Count := Pred.ElementCount;
  if Count = 0 then Exit(0);
  
  PPred := PSingleArray(Pred.RawData(Arena));
  PTarget := PSingleArray(Target.RawData(Arena));
  
  Sum := 0;
  for i := 0 to Count - 1 do
  begin
    P := PPred^[i];
    if P < 1e-7 then P := 1e-7;
    if P > 1 - 1e-7 then P := 1 - 1e-7;
    Sum := Sum - (PTarget^[i] * Ln(P));
  end;
  
  Result := Sum / Count;
end;

class function TOps.SoftmaxCrossEntropy(Arena: TArena; const Logits, Target: TTensor;
  var SoftmaxOut: TTensor): Single;
var
  PLogits, PTarget, PSoft: PSingleArray;
  i, Count: Integer;
  MaxVal, Sum, ExpVal, Loss, P: Single;
begin
  if not Logits.SameShape(Target) then
    raise Exception.Create('SoftmaxCrossEntropy: Shape mismatch');
    
  Count := Logits.ElementCount;
  if Count = 0 then Exit(0);
  
  PLogits := PSingleArray(Logits.RawData(Arena));
  PTarget := PSingleArray(Target.RawData(Arena));
  PSoft := PSingleArray(SoftmaxOut.RawData(Arena));
  
  // 1. Compute Softmax (numerically stable)
  MaxVal := PLogits^[0];
  for i := 1 to Count - 1 do
    if PLogits^[i] > MaxVal then MaxVal := PLogits^[i];
  
  Sum := 0;
  for i := 0 to Count - 1 do
  begin
    ExpVal := Exp(PLogits^[i] - MaxVal);
    PSoft^[i] := ExpVal;
    Sum := Sum + ExpVal;
  end;
  
  if Sum > 1e-10 then
    for i := 0 to Count - 1 do
      PSoft^[i] := PSoft^[i] / Sum;
  
  // 2. Compute CrossEntropy loss
  Loss := 0;
  for i := 0 to Count - 1 do
  begin
    P := PSoft^[i];
    if P < 1e-7 then P := 1e-7;
    Loss := Loss - PTarget^[i] * Ln(P);
  end;
  
  Result := Loss / Count;
end;

// ============================================================================
// BACKWARD OPERATIONS
// ============================================================================

class procedure TOps.MatMulBackward(Arena: TArena; const A, B, OutGrad: TTensor;
  var AGrad, BGrad: TTensor);
var
  PA, PB, POutGrad, PAGrad, PBGrad: PSingleArray;
  ARows, ACols, BRows, BCols: Integer;
  i, j, k, BatchIdx, BatchSize: Integer;
  Sum: Single;
  ANdim, BNdim: Integer;
  MatrixSizeA, MatrixSizeB, MatrixSizeOut: Integer;
  PABatch, PBBatch, POutGradBatch, PAGradBatch, PBGradBatch: PSingle;
begin
  ANdim := A.NDim;
  BNdim := B.NDim;
  
  if ANdim < 2 then Exit;
  if BNdim < 2 then Exit;
  
  // Get matrix dimensions (last 2 dims)
  ARows := A.Shape[ANdim - 2];
  ACols := A.Shape[ANdim - 1];
  BRows := B.Shape[BNdim - 2];
  BCols := B.Shape[BNdim - 1];
  
  if (ARows = 0) or (ACols = 0) or (BRows = 0) or (BCols = 0) then Exit;
  if OutGrad.GradPtr < 0 then Exit;
  
  // Compute batch size
  BatchSize := 1;
  for i := 0 to ANdim - 3 do
    BatchSize := BatchSize * A.Shape[i];
  
  MatrixSizeA := ARows * ACols;
  MatrixSizeB := BRows * BCols;
  MatrixSizeOut := ARows * BCols;
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  
  // Process each batch
  for BatchIdx := 0 to BatchSize - 1 do
  begin
    PABatch := @PA^[BatchIdx * MatrixSizeA];
    PBBatch := @PB^[BatchIdx * MatrixSizeB];
    POutGradBatch := @POutGrad^[BatchIdx * MatrixSizeOut];
    
    // dA = OutGrad * B^T
    if AGrad.RequiresGrad and (AGrad.GradPtr >= 0) then
    begin
      PAGrad := PSingleArray(AGrad.RawGrad(Arena));
      PAGradBatch := @PAGrad^[BatchIdx * MatrixSizeA];
      
      for i := 0 to ARows - 1 do
        for j := 0 to ACols - 1 do
        begin
          Sum := 0;
          for k := 0 to BCols - 1 do
            Sum := Sum + PSingleArray(POutGradBatch)^[i * BCols + k] * PSingleArray(PBBatch)^[j * BCols + k];
          PSingleArray(PAGradBatch)^[i * ACols + j] := PSingleArray(PAGradBatch)^[i * ACols + j] + Sum;
        end;
    end;
    
    // dB = A^T * OutGrad
    if BGrad.RequiresGrad and (BGrad.GradPtr >= 0) then
    begin
      PBGrad := PSingleArray(BGrad.RawGrad(Arena));
      PBGradBatch := @PBGrad^[BatchIdx * MatrixSizeB];
      
      for i := 0 to BRows - 1 do
        for j := 0 to BCols - 1 do
        begin
          Sum := 0;
          for k := 0 to ARows - 1 do
            Sum := Sum + PSingleArray(PABatch)^[k * ACols + i] * PSingleArray(POutGradBatch)^[k * BCols + j];
          PSingleArray(PBGradBatch)^[i * BCols + j] := PSingleArray(PBGradBatch)^[i * BCols + j] + Sum;
        end;
    end;
  end;
end;

class procedure TOps.AddBackward(Arena: TArena; const OutGrad: TTensor;
  var AGrad, BGrad: TTensor);
var
  POutGrad, PAGrad, PBGrad: PSingleArray;
  i, OutCount, IdxA, IdxB: Integer;
begin
  OutCount := OutGrad.ElementCount;
  if OutCount = 0 then Exit;
  if OutGrad.GradPtr < 0 then Exit;
  
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  
  // Backward for A: accumulate gradients, handling broadcasting
  if AGrad.GradPtr >= 0 then
  begin
    PAGrad := PSingleArray(AGrad.RawGrad(Arena));
    
    // If same shape, simple copy; else sum along broadcast dims
    if AGrad.SameShape(OutGrad) then
    begin
      for i := 0 to OutCount - 1 do
        PAGrad^[i] := PAGrad^[i] + POutGrad^[i];
    end
    else
    begin
      // Broadcasting: sum gradients to correct shape
      for i := 0 to OutCount - 1 do
      begin
        IdxA := BroadcastIndex(i, OutGrad.Shape, AGrad.Shape, AGrad.Strides);
        PAGrad^[IdxA] := PAGrad^[IdxA] + POutGrad^[i];
      end;
    end;
  end;
  
  // Backward for B: accumulate gradients, handling broadcasting
  if BGrad.GradPtr >= 0 then
  begin
    PBGrad := PSingleArray(BGrad.RawGrad(Arena));
    
    if BGrad.SameShape(OutGrad) then
    begin
      for i := 0 to OutCount - 1 do
        PBGrad^[i] := PBGrad^[i] + POutGrad^[i];
    end
    else
    begin
      for i := 0 to OutCount - 1 do
      begin
        IdxB := BroadcastIndex(i, OutGrad.Shape, BGrad.Shape, BGrad.Strides);
        PBGrad^[IdxB] := PBGrad^[IdxB] + POutGrad^[i];
      end;
    end;
  end;
end;

class procedure TOps.MulBackward(Arena: TArena; const A, B, OutGrad: TTensor;
  var AGrad, BGrad: TTensor);
var
  PA, PB, POutGrad, PAGrad, PBGrad: PSingleArray;
  i, OutCount, IdxA, IdxB: Integer;
begin
  OutCount := OutGrad.ElementCount;
  if OutCount = 0 then Exit;
  if OutGrad.GradPtr < 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  
  // dA = OutGrad * B, with broadcast handling
  if AGrad.GradPtr >= 0 then
  begin
    PAGrad := PSingleArray(AGrad.RawGrad(Arena));
    
    if AGrad.SameShape(OutGrad) and B.SameShape(OutGrad) then
    begin
      // Fast path: no broadcasting
      for i := 0 to OutCount - 1 do
        PAGrad^[i] := PAGrad^[i] + POutGrad^[i] * PB^[i];
    end
    else
    begin
      // Broadcasting path
      for i := 0 to OutCount - 1 do
      begin
        IdxA := BroadcastIndex(i, OutGrad.Shape, AGrad.Shape, AGrad.Strides);
        IdxB := BroadcastIndex(i, OutGrad.Shape, B.Shape, B.Strides);
        PAGrad^[IdxA] := PAGrad^[IdxA] + POutGrad^[i] * PB^[IdxB];
      end;
    end;
  end;
  
  // dB = OutGrad * A, with broadcast handling
  if BGrad.GradPtr >= 0 then
  begin
    PBGrad := PSingleArray(BGrad.RawGrad(Arena));
    
    if BGrad.SameShape(OutGrad) and A.SameShape(OutGrad) then
    begin
      // Fast path: no broadcasting
      for i := 0 to OutCount - 1 do
        PBGrad^[i] := PBGrad^[i] + POutGrad^[i] * PA^[i];
    end
    else
    begin
      // Broadcasting path
      for i := 0 to OutCount - 1 do
      begin
        IdxA := BroadcastIndex(i, OutGrad.Shape, A.Shape, A.Strides);
        IdxB := BroadcastIndex(i, OutGrad.Shape, BGrad.Shape, BGrad.Strides);
        PBGrad^[IdxB] := PBGrad^[IdxB] + POutGrad^[i] * PA^[IdxA];
      end;
    end;
  end;
end;

class procedure TOps.ReLUBackward(Arena: TArena; const A, OutGrad: TTensor;
  var AGrad: TTensor);
var
  PA, POutGrad, PAGrad: PSingleArray;
  i, Count: Integer;
begin
  Count := A.ElementCount;
  if Count = 0 then Exit;
  if (OutGrad.GradPtr < 0) or (AGrad.GradPtr < 0) then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PAGrad := PSingleArray(AGrad.RawGrad(Arena));
  
  for i := 0 to Count - 1 do
    if PA^[i] > 0 then
      PAGrad^[i] := PAGrad^[i] + POutGrad^[i];
end;

class procedure TOps.LeakyReLUBackward(Arena: TArena; const A, OutGrad: TTensor;
  var AGrad: TTensor; Alpha: Single);
var
  PA, POutGrad, PAGrad: PSingleArray;
  i, Count: Integer;
begin
  Count := A.ElementCount;
  if Count = 0 then Exit;
  if (OutGrad.GradPtr < 0) or (AGrad.GradPtr < 0) then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PAGrad := PSingleArray(AGrad.RawGrad(Arena));
  
  for i := 0 to Count - 1 do
    if PA^[i] > 0 then
      PAGrad^[i] := PAGrad^[i] + POutGrad^[i]
    else
      PAGrad^[i] := PAGrad^[i] + Alpha * POutGrad^[i];
end;

class procedure TOps.SigmoidBackward(Arena: TArena; const Out, OutGrad: TTensor;
  var AGrad: TTensor);
var
  POut, POutGrad, PAGrad: PSingleArray;
  i, Count: Integer;
begin
  Count := Out.ElementCount;
  if Count = 0 then Exit;
  if (OutGrad.GradPtr < 0) or (AGrad.GradPtr < 0) then Exit;
  
  POut := PSingleArray(Out.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PAGrad := PSingleArray(AGrad.RawGrad(Arena));
  
  for i := 0 to Count - 1 do
    PAGrad^[i] := PAGrad^[i] + POutGrad^[i] * POut^[i] * (1.0 - POut^[i]);
end;

class procedure TOps.TanhBackward(Arena: TArena; const Out, OutGrad: TTensor;
  var AGrad: TTensor);
var
  POut, POutGrad, PAGrad: PSingleArray;
  i, Count: Integer;
begin
  Count := Out.ElementCount;
  if Count = 0 then Exit;
  if (OutGrad.GradPtr < 0) or (AGrad.GradPtr < 0) then Exit;
  
  POut := PSingleArray(Out.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PAGrad := PSingleArray(AGrad.RawGrad(Arena));
  
  for i := 0 to Count - 1 do
    PAGrad^[i] := PAGrad^[i] + POutGrad^[i] * (1.0 - POut^[i] * POut^[i]);
end;

class procedure TOps.SoftmaxBackward(Arena: TArena; const Out, OutGrad: TTensor;
  var AGrad: TTensor);
var
  POut, POutGrad, PAGrad: PSingleArray;
  i, k, Count: Integer;
  Sum, Jacobian: Single;
begin
  // Full Softmax backward with proper Jacobian
  // Jacobian[i,k] = p[i] * (delta_ik - p[k])
  // grad[k] = sum_i ( OutGrad[i] * Jacobian[i,k] )
  //         = sum_i ( OutGrad[i] * p[i] * (delta_ik - p[k]) )
  //         = OutGrad[k] * p[k] - p[k] * sum_i(OutGrad[i] * p[i])
  Count := Out.ElementCount;
  if Count = 0 then Exit;
  if (OutGrad.GradPtr < 0) or (AGrad.GradPtr < 0) then Exit;
  
  POut := PSingleArray(Out.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PAGrad := PSingleArray(AGrad.RawGrad(Arena));
  
  // Compute sum_i(OutGrad[i] * p[i])
  Sum := 0;
  for i := 0 to Count - 1 do
    Sum := Sum + POutGrad^[i] * POut^[i];
  
  // grad[k] = p[k] * (OutGrad[k] - Sum)
  for k := 0 to Count - 1 do
    PAGrad^[k] := PAGrad^[k] + POut^[k] * (POutGrad^[k] - Sum);
end;

class procedure TOps.SoftmaxCrossEntropyBackward(Arena: TArena; 
  const SoftmaxOut, Target, OutGrad: TTensor; var AGrad: TTensor);
var
  PSoft, PTarget, POutGrad, PAGrad: PSingleArray;
  i, Count: Integer;
  LossGrad: Single;
begin
  // Combined Softmax + CrossEntropy gradient:
  // Gradient = Softmax(x) - Target
  // Numerically stable, no Jacobian matrix required
  Count := SoftmaxOut.ElementCount;
  if Count = 0 then Exit;
  if (OutGrad.GradPtr < 0) or (AGrad.GradPtr < 0) then Exit;
  
  PSoft := PSingleArray(SoftmaxOut.RawData(Arena));
  PTarget := PSingleArray(Target.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PAGrad := PSingleArray(AGrad.RawGrad(Arena));
  
  // Scale by upstream gradient (usually 1.0 from loss)
  LossGrad := POutGrad^[0];
  
  for i := 0 to Count - 1 do
    PAGrad^[i] := PAGrad^[i] + (PSoft^[i] - PTarget^[i]) * LossGrad / Count;
end;

// ============================================================================
// CONVOLUTION OPERATIONS
// ============================================================================

class procedure TOps.Conv2D(Arena: TArena; const Input, Weight: TTensor;
  Padding, Stride: Integer; var OutT: TTensor);
var
  PInput, PWeight, POut: PSingleArray;
  Batch, InCh, InH, InW: Integer;
  OutCh, KernelH, KernelW: Integer;
  OutH, OutW: Integer;
  b, oc, oh, ow, ic, kh, kw: Integer;
  InRow, InCol: Integer;
  ColSize, ColRows, ColCols: Integer;
  PCol: PSingleArray;
  ColIdx, InputIdx, WeightIdx, OutIdx: Integer;
  SavePoint: TMemPtr;
  UseAVX512, UseSSE: Boolean;
  Sum: Single;
  PtrW, PtrCol: PSingle;
begin
  // Validate shapes: Input must be [Batch, InChannels, InH, InW]
  if Input.NDim <> 4 then
    raise Exception.Create('Conv2D: Input must be 4D tensor [Batch, InChannels, InH, InW]');
  
  // Weight must be [OutChannels, InChannels, KernelH, KernelW]
  if Weight.NDim <> 4 then
    raise Exception.Create('Conv2D: Weight must be 4D tensor [OutChannels, InChannels, KernelH, KernelW]');
  
  Batch := Input.Shape[0];
  InCh := Input.Shape[1];
  InH := Input.Shape[2];
  InW := Input.Shape[3];
  
  OutCh := Weight.Shape[0];
  if Weight.Shape[1] <> InCh then
    raise Exception.CreateFmt('Conv2D: Input channels mismatch: input has %d, weight expects %d',
      [InCh, Weight.Shape[1]]);
  KernelH := Weight.Shape[2];
  KernelW := Weight.Shape[3];
  
  // Compute output dimensions
  OutH := (InH + 2 * Padding - KernelH) div Stride + 1;
  OutW := (InW + 2 * Padding - KernelW) div Stride + 1;
  
  if (OutH <= 0) or (OutW <= 0) then
    raise Exception.CreateFmt('Conv2D: Invalid output dimensions: OutH=%d, OutW=%d', [OutH, OutW]);
  
  // Validate output tensor shape
  if (OutT.NDim <> 4) or (OutT.Shape[0] <> Batch) or (OutT.Shape[1] <> OutCh) or
     (OutT.Shape[2] <> OutH) or (OutT.Shape[3] <> OutW) then
    raise Exception.CreateFmt('Conv2D: Output shape mismatch: expected [%d, %d, %d, %d], got [%d, %d, %d, %d]',
      [Batch, OutCh, OutH, OutW, OutT.Shape[0], OutT.Shape[1], OutT.Shape[2], OutT.Shape[3]]);
  
  PInput := PSingleArray(Input.RawData(Arena));
  PWeight := PSingleArray(Weight.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  // =====================================================================
  // IM2COL + GEMM APPROACH
  // =====================================================================
  // Instead of 6 nested loops, we:
  // 1. Extract input patches into a column matrix (im2col)
  // 2. Perform matrix multiplication: Weight @ Col = Output
  //
  // Col matrix shape: [InCh * KernelH * KernelW, OutH * OutW]
  // Weight reshaped:  [OutCh, InCh * KernelH * KernelW]
  // Output reshaped:  [OutCh, OutH * OutW]
  //
  // This converts convolution to GEMM, which can use SSE/AVX-512!
  // =====================================================================
  
  ColRows := InCh * KernelH * KernelW;  // Rows in col matrix
  ColCols := OutH * OutW;                // Cols in col matrix
  ColSize := ColRows * ColCols;
  
  // Save arena state for temporary allocation
  SavePoint := Arena.GetSavePoint;
  
  // Allocate im2col buffer (temporary)
  PCol := PSingleArray(Arena.GetPtr(Arena.Alloc(ColSize)));
  
  // Get CPU features for SIMD dispatch
  UseAVX512 := CPUFeatures.HasAVX512;
  UseSSE := CPUFeatures.HasSSE;
  
  // Process each batch
  for b := 0 to Batch - 1 do
  begin
    // Step 1: im2col - extract patches into column matrix
    for oh := 0 to OutH - 1 do
      for ow := 0 to OutW - 1 do
      begin
        ColIdx := oh * OutW + ow;  // Column index in col matrix
        
        for ic := 0 to InCh - 1 do
          for kh := 0 to KernelH - 1 do
            for kw := 0 to KernelW - 1 do
            begin
              InRow := oh * Stride + kh - Padding;
              InCol := ow * Stride + kw - Padding;
              
              // Row index in col matrix
              InputIdx := (ic * KernelH + kh) * KernelW + kw;
              
              // Store in ROW-major order: Col[row][col] = Col[InputIdx * ColCols + ColIdx]
              // For GEMM, we need Weight[oc, :] dot Col[:, colIdx]
              // So store Col as [ColRows, ColCols] row-major, then access column is strided
              // BETTER: Store as [ColCols, ColRows] so Col[colIdx, :] is contiguous
              // ColT[colIdx * ColRows + InputIdx] gives contiguous row for dot product
              if (InRow >= 0) and (InRow < InH) and (InCol >= 0) and (InCol < InW) then
                PCol^[ColIdx * ColRows + InputIdx] := PInput^[b * (InCh * InH * InW) + ic * (InH * InW) + InRow * InW + InCol]
              else
                PCol^[ColIdx * ColRows + InputIdx] := 0;  // Padding
            end;
      end;
    
    // Step 2: GEMM - Weight @ Col^T = Output
    // Weight: [OutCh, ColRows], Col^T: [ColCols, ColRows] stored as [ColCols * ColRows]
    // For each output position, compute dot product of weight row and col row
    for oc := 0 to OutCh - 1 do
    begin
      PtrW := @PWeight^[oc * ColRows];  // Weight row pointer (contiguous)
      
      for ColIdx := 0 to ColCols - 1 do
      begin
        // Now Col[ColIdx, :] is contiguous at PCol^[ColIdx * ColRows]
        // Weight row is contiguous at PtrW
        // Perfect for SIMD dot product!
        if UseAVX512 then
          Sum := TKernels.DotProductAVX512(PtrW, @PCol^[ColIdx * ColRows], ColRows)
        else if UseSSE then
          Sum := TKernels.DotProductSSE(PtrW, @PCol^[ColIdx * ColRows], ColRows)
        else
          Sum := TKernels.DotProduct(PtrW, @PCol^[ColIdx * ColRows], ColRows);
        
        // Store output
        OutIdx := b * (OutCh * OutH * OutW) + oc * (OutH * OutW) + ColIdx;
        POut^[OutIdx] := Sum;
      end;
    end;
  end;
  
  // Free temporary memory
  Arena.Restore(SavePoint);
end;

class procedure TOps.Conv2DBackward(Arena: TArena; const Input, Weight, OutGrad: TTensor;
  Padding, Stride: Integer; var InputGrad, WeightGrad: TTensor);
var
  PInput, PWeight, POutGrad: PSingleArray;
  PInputGrad, PWeightGrad: PSingleArray;
  Batch, InCh, InH, InW: Integer;
  OutCh, KernelH, KernelW: Integer;
  OutH, OutW: Integer;
  b, oc, oh, ow, ic, kh, kw: Integer;
  InRow, InCol: Integer;
  InputIdx, WeightIdx, OutIdx: Integer;
  InputGradIdx, WeightGradIdx: Integer;
  InputStride, WeightStride, OutStride: TArray<Integer>;
  InputGradStride, WeightGradStride: TArray<Integer>;
begin
  // Validate shapes
  if Input.NDim <> 4 then Exit;
  if Weight.NDim <> 4 then Exit;
  if OutGrad.NDim <> 4 then Exit;
  if OutGrad.GradPtr < 0 then Exit;  // Need gradient allocated to proceed
  
  Batch := Input.Shape[0];
  InCh := Input.Shape[1];
  InH := Input.Shape[2];
  InW := Input.Shape[3];
  
  OutCh := Weight.Shape[0];
  KernelH := Weight.Shape[2];
  KernelW := Weight.Shape[3];
  
  OutH := OutGrad.Shape[2];
  OutW := OutGrad.Shape[3];
  
  PInput := PSingleArray(Input.RawData(Arena));
  PWeight := PSingleArray(Weight.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));  // Gradient OF the output
  
  InputStride := Input.Strides;
  WeightStride := Weight.Strides;
  OutStride := OutGrad.Strides;
  
  // Input gradient: full convolution of output gradient with rotated/flipped kernel
  if InputGrad.GradPtr >= 0 then
  begin
    PInputGrad := PSingleArray(InputGrad.RawGrad(Arena));
    InputGradStride := InputGrad.Strides;
    
    // Initialize to zero
    for InputGradIdx := 0 to InputGrad.ElementCount - 1 do
      PInputGrad^[InputGradIdx] := 0;
    
    // For each input position, accumulate contributions from all output positions
    for b := 0 to Batch - 1 do
      for ic := 0 to InCh - 1 do
        for InRow := 0 to InH - 1 do
          for InCol := 0 to InW - 1 do
          begin
            InputGradIdx := b * InputGradStride[0] + ic * InputGradStride[1] + 
                           InRow * InputGradStride[2] + InCol * InputGradStride[3];
            
            // Sum contributions from all output channels and positions
            for oc := 0 to OutCh - 1 do
              for kh := 0 to KernelH - 1 do
                for kw := 0 to KernelW - 1 do
                begin
                  // Find corresponding output position
                  oh := (InRow + Padding - kh) div Stride;
                  ow := (InCol + Padding - kw) div Stride;
                  
                  // Check if this output position is valid
                  if (oh >= 0) and (oh < OutH) and (ow >= 0) and (ow < OutW) and
                     ((InRow + Padding - kh) mod Stride = 0) and
                     ((InCol + Padding - kw) mod Stride = 0) then
                  begin
                    OutIdx := b * OutStride[0] + oc * OutStride[1] + 
                              oh * OutStride[2] + ow * OutStride[3];
                    
                    WeightIdx := oc * WeightStride[0] + ic * WeightStride[1] + 
                                 kh * WeightStride[2] + kw * WeightStride[3];
                    
                    PInputGrad^[InputGradIdx] := PInputGrad^[InputGradIdx] + 
                                                 POutGrad^[OutIdx] * PWeight^[WeightIdx];
                  end;
                end;
          end;
  end;
  
  // Weight gradient: convolution of input with output gradient
  if WeightGrad.GradPtr >= 0 then
  begin
    PWeightGrad := PSingleArray(WeightGrad.RawGrad(Arena));
    WeightGradStride := WeightGrad.Strides;
    
    // Initialize to zero
    for WeightGradIdx := 0 to WeightGrad.ElementCount - 1 do
      PWeightGrad^[WeightGradIdx] := 0;
    
    // For each kernel position, accumulate contributions
    for oc := 0 to OutCh - 1 do
      for ic := 0 to InCh - 1 do
        for kh := 0 to KernelH - 1 do
          for kw := 0 to KernelW - 1 do
          begin
            WeightGradIdx := oc * WeightGradStride[0] + ic * WeightGradStride[1] + 
                             kh * WeightGradStride[2] + kw * WeightGradStride[3];
            
            // Sum over all batches and output positions
            for b := 0 to Batch - 1 do
              for oh := 0 to OutH - 1 do
                for ow := 0 to OutW - 1 do
                begin
                  // Compute corresponding input position
                  InRow := oh * Stride + kh - Padding;
                  InCol := ow * Stride + kw - Padding;
                  
                  if (InRow >= 0) and (InRow < InH) and (InCol >= 0) and (InCol < InW) then
                  begin
                    InputIdx := b * InputStride[0] + ic * InputStride[1] + 
                                InRow * InputStride[2] + InCol * InputStride[3];
                    
                    OutIdx := b * OutStride[0] + oc * OutStride[1] + 
                              oh * OutStride[2] + ow * OutStride[3];
                    
                    PWeightGrad^[WeightGradIdx] := PWeightGrad^[WeightGradIdx] + 
                                                    PInput^[InputIdx] * POutGrad^[OutIdx];
                  end;
                end;
          end;
  end;
end;

// ============================================================================
// MAXPOOL2D OPERATIONS
// ============================================================================

class procedure TOps.MaxPool2D(Arena: TArena; const Input: TTensor;
  PoolSize, Stride: Integer; var OutT: TTensor; var MaxIndices: TArray<Integer>);
var
  PInput, POut: PSingleArray;
  Batch, Channels, InH, InW: Integer;
  OutH, OutW: Integer;
  b, c, oh, ow, ph, pw: Integer;
  InRow, InCol: Integer;
  MaxVal, Val: Single;
  MaxIdx, CurIdx, OutIdx: Integer;
begin
  // Input shape: [Batch, Channels, InH, InW]
  if Input.NDim <> 4 then Exit;
  
  Batch := Input.Shape[0];
  Channels := Input.Shape[1];
  InH := Input.Shape[2];
  InW := Input.Shape[3];
  
  OutH := (InH - PoolSize) div Stride + 1;
  OutW := (InW - PoolSize) div Stride + 1;
  
  PInput := PSingleArray(Input.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  // Allocate max indices for backward pass
  SetLength(MaxIndices, Batch * Channels * OutH * OutW);
  
  for b := 0 to Batch - 1 do
    for c := 0 to Channels - 1 do
      for oh := 0 to OutH - 1 do
        for ow := 0 to OutW - 1 do
        begin
          MaxVal := -1e30;
          MaxIdx := 0;
          
          // Find max in pooling window
          for ph := 0 to PoolSize - 1 do
            for pw := 0 to PoolSize - 1 do
            begin
              InRow := oh * Stride + ph;
              InCol := ow * Stride + pw;
              
              if (InRow < InH) and (InCol < InW) then
              begin
                CurIdx := b * (Channels * InH * InW) + c * (InH * InW) + InRow * InW + InCol;
                Val := PInput^[CurIdx];
                if Val > MaxVal then
                begin
                  MaxVal := Val;
                  MaxIdx := CurIdx;
                end;
              end;
            end;
          
          OutIdx := b * (Channels * OutH * OutW) + c * (OutH * OutW) + oh * OutW + ow;
          POut^[OutIdx] := MaxVal;
          MaxIndices[OutIdx] := MaxIdx;
        end;
end;

class procedure TOps.MaxPool2DBackward(Arena: TArena; const OutGrad: TTensor;
  const MaxIndices: TArray<Integer>; var InputGrad: TTensor);
var
  POutGrad, PInputGrad: PSingleArray;
  i, Count: Integer;
begin
  if InputGrad.GradPtr < 0 then Exit;
  
  Count := OutGrad.ElementCount;
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PInputGrad := PSingleArray(InputGrad.RawGrad(Arena));
  
  // Route gradient to max element positions
  for i := 0 to Count - 1 do
    PInputGrad^[MaxIndices[i]] := PInputGrad^[MaxIndices[i]] + POutGrad^[i];
end;

// ============================================================================
// DROPOUT OPERATIONS
// ============================================================================

class procedure TOps.Dropout(Arena: TArena; const Input: TTensor;
  DropRate: Single; IsTraining: Boolean; var OutT: TTensor; var Mask: TArray<Byte>);
var
  PInput, POut: PSingleArray;
  i, Count: Integer;
  Scale: Single;
begin
  Count := Input.ElementCount;
  PInput := PSingleArray(Input.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  if not IsTraining then
  begin
    // During inference, just copy (no dropout)
    for i := 0 to Count - 1 do
      POut^[i] := PInput^[i];
    Exit;
  end;
  
  // During training, randomly zero elements and scale
  SetLength(Mask, Count);
  Scale := 1.0 / (1.0 - DropRate);  // Inverted dropout scaling
  
  for i := 0 to Count - 1 do
  begin
    if Random < DropRate then
    begin
      Mask[i] := 0;  // Dropped
      POut^[i] := 0;
    end
    else
    begin
      Mask[i] := 1;  // Kept
      POut^[i] := PInput^[i] * Scale;
    end;
  end;
end;

class procedure TOps.DropoutBackward(Arena: TArena; const OutGrad: TTensor;
  const Mask: TArray<Byte>; DropRate: Single; var AGrad: TTensor);
var
  POutGrad, PAGrad: PSingleArray;
  i, Count: Integer;
  Scale: Single;
begin
  if AGrad.GradPtr < 0 then Exit;
  
  Count := OutGrad.ElementCount;
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PAGrad := PSingleArray(AGrad.RawGrad(Arena));
  Scale := 1.0 / (1.0 - DropRate);
  
  for i := 0 to Count - 1 do
    if Mask[i] = 1 then
      PAGrad^[i] := PAGrad^[i] + POutGrad^[i] * Scale;
    // Dropped elements get 0 gradient (no accumulation needed)
end;

// ============================================================================
// CPU FEATURE DETECTION
// ============================================================================

{$IFDEF CPUX64}
// Detect AVX-512F support using CPUID instruction
function DetectAVX512Support: Boolean; assembler;
asm
  .NOFRAME
  PUSH RBX              // RBX is callee-saved, must preserve
  MOV EAX, 7            // CPUID function 7: Extended Features
  XOR ECX, ECX          // Sub-leaf 0
  CPUID
  BT EBX, 16            // Test bit 16 of EBX (AVX-512F)
  SETC AL               // Set AL to 1 if bit was set
  MOVZX EAX, AL         // Zero-extend to full register for return value
  POP RBX               // Restore RBX
end;
{$ELSE}
function DetectAVX512Support: Boolean;
begin
  Result := False;  // 32-bit doesn't support AVX-512
end;
{$ENDIF}

function DetectCPUFeatures: TCPUFeatures;
{$IFDEF MSWINDOWS}
const
  PF_XMMI_INSTRUCTIONS_AVAILABLE = 6;  // SSE support
  PF_AVX_INSTRUCTIONS_AVAILABLE = 12;  // AVX support
var
  SSEPresent, AVXPresent: LongBool;
begin
  // Use Windows API to detect SSE and AVX
  SSEPresent := IsProcessorFeaturePresent(PF_XMMI_INSTRUCTIONS_AVAILABLE);
  AVXPresent := IsProcessorFeaturePresent(PF_AVX_INSTRUCTIONS_AVAILABLE);
  Result.HasSSE := Boolean(SSEPresent);
  Result.HasAVX := Boolean(AVXPresent);
  // Use CPUID to detect AVX-512 (Windows API doesn't expose this)
  Result.HasAVX512 := DetectAVX512Support;
end;
{$ELSE}
begin
  // Non-Windows: assume basic features only
  Result.HasSSE := False;
  Result.HasAVX := False;
  Result.HasAVX512 := False;
end;
{$ENDIF}

// ============================================================================
// AVX-512 KERNELS
// ============================================================================

{$IFDEF CPUX64}
class function TKernels.DotProductAVX512(A, B: PSingle; Count: Integer): Single; assembler;
asm
  // RCX = A, RDX = B, R8D = Count (Windows x64 calling convention)
  .NOFRAME
  
  VXORPS ZMM0, ZMM0, ZMM0  // Clear accumulator (512-bit)
  
  TEST R8D, R8D
  JLE @Done
  
  // Main AVX-512 loop (16 floats at a time)
  MOV R9D, R8D
  SHR R9D, 4           // Count div 16
  JZ @Tail
  
@Loop16:
  VMOVUPS ZMM1, [RCX]  // Load 16 floats from A
  VMOVUPS ZMM2, [RDX]  // Load 16 floats from B
  VMULPS ZMM1, ZMM1, ZMM2  // Multiply packed
  VADDPS ZMM0, ZMM0, ZMM1  // Accumulate
  
  ADD RCX, 64          // 16 floats * 4 bytes = 64 bytes
  ADD RDX, 64
  DEC R9D
  JNZ @Loop16
  
  // Horizontal reduction: sum all 16 floats in ZMM0
  // Use VHADDPS for horizontal addition
  VEXTRACTF32X8 YMM1, ZMM0, 0  // Extract lower 8 floats
  VEXTRACTF32X8 YMM2, ZMM0, 1   // Extract upper 8 floats
  VADDPS YMM0, YMM1, YMM2       // Add the two halves
  
  // Reduce 8 floats to 1
  VHADDPS YMM0, YMM0, YMM0      // [a+b, c+d, e+f, g+h, ...]
  VHADDPS YMM0, YMM0, YMM0      // [sum1, sum2, sum3, sum4, ...]
  VPERM2F128 YMM1, YMM0, YMM0, 1
  VADDPS YMM0, YMM0, YMM1
  VEXTRACTPS EAX, XMM0, 0        // Extract result to EAX (will be in XMM0)
  
@Tail:
  // Handle remaining 0-15 elements
  AND R8D, 15
  JZ @Done
  
@Loop1:
  VMOVSS XMM1, [RCX]
  VMULSS XMM1, XMM1, [RDX]
  VADDSS XMM0, XMM0, XMM1
  ADD RCX, 4
  ADD RDX, 4
  DEC R8D
  JNZ @Loop1
  
@Done:
  // Result is in XMM0 (return value for Single)
end;

class procedure TKernels.VectorAddAVX512(A, B, Output: PSingle; Count: Integer);
asm
  // RCX = A, RDX = B, R8 = Output, R9D = Count (Windows x64)
  .NOFRAME
  
  TEST R9D, R9D
  JLE @Done
  
  MOV R10D, R9D
  SHR R10D, 4          // Count div 16
  JZ @Tail
  
@Loop16:
  VMOVUPS ZMM0, [RCX]
  VMOVUPS ZMM1, [RDX]
  VADDPS ZMM0, ZMM0, ZMM1
  VMOVUPS [R8], ZMM0
  
  ADD RCX, 64
  ADD RDX, 64
  ADD R8, 64
  DEC R10D
  JNZ @Loop16
  
@Tail:
  AND R9D, 15
  JZ @Done
  
@Loop1:
  VMOVSS XMM0, [RCX]
  VADDSS XMM0, XMM0, [RDX]
  VMOVSS [R8], XMM0
  ADD RCX, 4
  ADD RDX, 4
  ADD R8, 4
  DEC R9D
  JNZ @Loop1
  
@Done:
end;

class procedure TKernels.VectorMulAVX512(A, B, Output: PSingle; Count: Integer); assembler;
asm
  // RCX = A, RDX = B, R8 = Output, R9D = Count (Windows x64)
  .NOFRAME
  
  TEST R9D, R9D
  JLE @Done
  
  MOV R10D, R9D
  SHR R10D, 4
  JZ @Tail
  
@Loop16:
  VMOVUPS ZMM0, [RCX]
  VMOVUPS ZMM1, [RDX]
  VMULPS ZMM0, ZMM0, ZMM1
  VMOVUPS [R8], ZMM0
  
  ADD RCX, 64
  ADD RDX, 64
  ADD R8, 64
  DEC R10D
  JNZ @Loop16
  
@Tail:
  AND R9D, 15
  JZ @Done
  
@Loop1:
  VMOVSS XMM0, [RCX]
  VMULSS XMM0, XMM0, [RDX]
  VMOVSS [R8], XMM0
  ADD RCX, 4
  ADD RDX, 4
  ADD R8, 4
  DEC R9D
  JNZ @Loop1
  
@Done:
end;

// ============================================================================
// SSE KERNELS (Fallback for CPUs without AVX-512)
// ============================================================================

class function TKernels.DotProductSSE(A, B: PSingle; Count: Integer): Single; assembler;
asm
  // RCX = A, RDX = B, R8D = Count (Windows x64 calling convention)
  .NOFRAME
  
  XORPS XMM0, XMM0      // Clear accumulator (128-bit)
  
  TEST R8D, R8D
  JLE @Done
  
  // Main SSE loop (4 floats at a time)
  MOV R9D, R8D
  SHR R9D, 2            // Count div 4
  JZ @Tail
  
@Loop4:
  MOVUPS XMM1, [RCX]    // Load 4 floats from A
  MOVUPS XMM2, [RDX]    // Load 4 floats from B
  MULPS XMM1, XMM2      // Multiply packed
  ADDPS XMM0, XMM1      // Accumulate
  
  ADD RCX, 16           // 4 floats * 4 bytes = 16 bytes
  ADD RDX, 16
  DEC R9D
  JNZ @Loop4
  
  // Horizontal sum: reduce 4 floats to 1
  HADDPS XMM0, XMM0     // [a+b, c+d, a+b, c+d]
  HADDPS XMM0, XMM0     // [sum, sum, sum, sum]
  
@Tail:
  // Handle remaining 0-3 elements
  AND R8D, 3
  JZ @Done
  
@Loop1:
  MOVSS XMM1, [RCX]
  MULSS XMM1, [RDX]
  ADDSS XMM0, XMM1
  ADD RCX, 4
  ADD RDX, 4
  DEC R8D
  JNZ @Loop1
  
@Done:
  // Result is in XMM0 (return value for Single)
end;

class procedure TKernels.VectorAddSSE(A, B, Output: PSingle; Count: Integer); assembler;
asm
  // RCX = A, RDX = B, R8 = Output, R9D = Count (Windows x64)
  .NOFRAME
  
  TEST R9D, R9D
  JLE @Done
  
  MOV R10D, R9D
  SHR R10D, 2           // Count div 4
  JZ @Tail
  
@Loop4:
  MOVUPS XMM0, [RCX]
  MOVUPS XMM1, [RDX]
  ADDPS XMM0, XMM1
  MOVUPS [R8], XMM0
  
  ADD RCX, 16
  ADD RDX, 16
  ADD R8, 16
  DEC R10D
  JNZ @Loop4
  
@Tail:
  AND R9D, 3
  JZ @Done
  
@Loop1:
  MOVSS XMM0, [RCX]
  ADDSS XMM0, [RDX]
  MOVSS [R8], XMM0
  ADD RCX, 4
  ADD RDX, 4
  ADD R8, 4
  DEC R9D
  JNZ @Loop1
  
@Done:
end;

class procedure TKernels.VectorMulSSE(A, B, Output: PSingle; Count: Integer); assembler;
asm
  // RCX = A, RDX = B, R8 = Output, R9D = Count (Windows x64)
  .NOFRAME
  
  TEST R9D, R9D
  JLE @Done
  
  MOV R10D, R9D
  SHR R10D, 2           // Count div 4
  JZ @Tail
  
@Loop4:
  MOVUPS XMM0, [RCX]
  MOVUPS XMM1, [RDX]
  MULPS XMM0, XMM1
  MOVUPS [R8], XMM0
  
  ADD RCX, 16
  ADD RDX, 16
  ADD R8, 16
  DEC R10D
  JNZ @Loop4
  
@Tail:
  AND R9D, 3
  JZ @Done
  
@Loop1:
  MOVSS XMM0, [RCX]
  MULSS XMM0, [RDX]
  MOVSS [R8], XMM0
  ADD RCX, 4
  ADD RDX, 4
  ADD R8, 4
  DEC R9D
  JNZ @Loop1
  
@Done:
end;
{$ENDIF}

// ============================================================================
// BATCH NORMALIZATION
// ============================================================================

class procedure TOps.BatchNorm(Arena: TArena; const Input, Gamma, Beta: TTensor;
  Epsilon: Single; IsTraining: Boolean; var RunningMean, RunningVar: TArray<Single>;
  Momentum: Single; var OutT: TTensor);
var
  PInput, PGamma, PBeta, POut: PSingleArray;
  Batch, Channels, SpatialSize: Integer;
  i, c, s: Integer;
  Mean, Var_, Std: Single;
  Sum, SumSq: Single;
  Count: Integer;
  InputIdx, OutIdx: Integer;
  InputStrides, OutStrides: TArray<Integer>;
  d, DimSize, DimIdx, SpatialIdx: Integer;
begin
  // BatchNorm expects input shape [Batch, Channels, ...] (at least 2D)
  if Input.NDim < 2 then
    raise Exception.Create('BatchNorm: Input must have at least 2 dimensions');
  
  Batch := Input.Shape[0];
  Channels := Input.Shape[1];
  
  // Compute spatial size (product of remaining dimensions)
  SpatialSize := 1;
  for i := 2 to Input.NDim - 1 do
    SpatialSize := SpatialSize * Input.Shape[i];
  
  // Validate gamma and beta shapes
  if (Gamma.NDim <> 1) or (Gamma.Shape[0] <> Channels) then
    raise Exception.CreateFmt('BatchNorm: Gamma shape mismatch: expected [%d], got %d dims',
      [Channels, Gamma.NDim]);
  if (Beta.NDim <> 1) or (Beta.Shape[0] <> Channels) then
    raise Exception.CreateFmt('BatchNorm: Beta shape mismatch: expected [%d], got %d dims',
      [Channels, Beta.NDim]);
  
  PInput := PSingleArray(Input.RawData(Arena));
  PGamma := PSingleArray(Gamma.RawData(Arena));
  PBeta := PSingleArray(Beta.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  InputStrides := Input.Strides;
  OutStrides := OutT.Strides;
  
  // Initialize running statistics if needed
  if Length(RunningMean) <> Channels then
  begin
    SetLength(RunningMean, Channels);
    SetLength(RunningVar, Channels);
    for c := 0 to Channels - 1 do
    begin
      RunningMean[c] := 0;
      RunningVar[c] := 1;  // Initialize variance to 1
    end;
  end;
  
  // Process each channel
  for c := 0 to Channels - 1 do
  begin
    if IsTraining then
    begin
      // Training mode: compute batch statistics
      Sum := 0;
      SumSq := 0;
      Count := Batch * SpatialSize;
      
      for i := 0 to Batch - 1 do
        for s := 0 to SpatialSize - 1 do
        begin
          InputIdx := i * InputStrides[0] + c * InputStrides[1];
          // Add spatial dimension offsets
          SpatialIdx := s;
          for d := 2 to Input.NDim - 1 do
          begin
            DimSize := Input.Shape[d];
            DimIdx := SpatialIdx mod DimSize;
            SpatialIdx := SpatialIdx div DimSize;
            InputIdx := InputIdx + DimIdx * InputStrides[d];
          end;
          
          var Val := PInput^[InputIdx];
          Sum := Sum + Val;
          SumSq := SumSq + Val * Val;
        end;
      
      Mean := Sum / Count;
      Var_ := (SumSq / Count) - (Mean * Mean);
      if Var_ < 0 then Var_ := 0;  // Numerical stability
      
      // Update running statistics
      RunningMean[c] := Momentum * RunningMean[c] + (1 - Momentum) * Mean;
      RunningVar[c] := Momentum * RunningVar[c] + (1 - Momentum) * Var_;
    end
    else
    begin
      // Inference mode: use running statistics
      Mean := RunningMean[c];
      Var_ := RunningVar[c];
    end;
    
    // Compute standard deviation
    Std := Sqrt(Var_ + Epsilon);
    
    // Normalize and scale
    for i := 0 to Batch - 1 do
      for s := 0 to SpatialSize - 1 do
      begin
        InputIdx := i * InputStrides[0] + c * InputStrides[1];
        // Add spatial dimension offsets
        SpatialIdx := s;
        for d := 2 to Input.NDim - 1 do
        begin
          DimSize := Input.Shape[d];
          DimIdx := SpatialIdx mod DimSize;
          SpatialIdx := SpatialIdx div DimSize;
          InputIdx := InputIdx + DimIdx * InputStrides[d];
        end;
        
        OutIdx := i * OutStrides[0] + c * OutStrides[1];
        // Add spatial dimension offsets for output
        SpatialIdx := s;
        for d := 2 to OutT.NDim - 1 do
        begin
          DimSize := OutT.Shape[d];
          DimIdx := SpatialIdx mod DimSize;
          SpatialIdx := SpatialIdx div DimSize;
          OutIdx := OutIdx + DimIdx * OutStrides[d];
        end;
        
        // BatchNorm: output = gamma * (input - mean) / std + beta
        POut^[OutIdx] := PGamma^[c] * ((PInput^[InputIdx] - Mean) / Std) + PBeta^[c];
      end;
  end;
end;

class procedure TOps.BatchNormBackward(Arena: TArena; const Input, OutGrad, Gamma: TTensor;
  Epsilon: Single; var InputGrad, GammaGrad, BetaGrad: TTensor);
var
  PInput, POutGrad, PGamma, PInputGrad, PGammaGrad, PBetaGrad: PSingleArray;
  Batch, Channels, SpatialSize: Integer;
  i, c, s: Integer;
  Mean, Var_, Std: Single;
  Sum, SumSq, SumGrad, SumGradNorm: Single;
  Count: Integer;
  InputIdx, OutGradIdx, InputGradIdx: Integer;
  InputStrides, OutGradStrides, InputGradStrides: TArray<Integer>;
  Normalized: Single;
  d, DimSize, DimIdx, SpatialIdx: Integer;
begin
  // Compute batch statistics (same as forward pass)
  Batch := Input.Shape[0];
  Channels := Input.Shape[1];
  SpatialSize := 1;
  for i := 2 to Input.NDim - 1 do
    SpatialSize := SpatialSize * Input.Shape[i];
  
  PInput := PSingleArray(Input.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PGamma := PSingleArray(Gamma.RawData(Arena));
  PInputGrad := PSingleArray(InputGrad.RawGrad(Arena));
  PGammaGrad := PSingleArray(GammaGrad.RawGrad(Arena));
  PBetaGrad := PSingleArray(BetaGrad.RawGrad(Arena));
  
  InputStrides := Input.Strides;
  OutGradStrides := OutGrad.Strides;
  InputGradStrides := InputGrad.Strides;
  
  Count := Batch * SpatialSize;
  
  // Process each channel
  for c := 0 to Channels - 1 do
  begin
    // Compute mean and variance for this channel
    Sum := 0;
    SumSq := 0;
    for i := 0 to Batch - 1 do
      for s := 0 to SpatialSize - 1 do
      begin
        InputIdx := i * InputStrides[0] + c * InputStrides[1];
        SpatialIdx := s;
        for d := 2 to Input.NDim - 1 do
        begin
          DimSize := Input.Shape[d];
          DimIdx := SpatialIdx mod DimSize;
          SpatialIdx := SpatialIdx div DimSize;
          InputIdx := InputIdx + DimIdx * InputStrides[d];
        end;
        var Val := PInput^[InputIdx];
        Sum := Sum + Val;
        SumSq := SumSq + Val * Val;
      end;
    
    Mean := Sum / Count;
    Var_ := (SumSq / Count) - (Mean * Mean);
    if Var_ < 0 then Var_ := 0;
    Std := Sqrt(Var_ + Epsilon);
    
    // Compute gradients
    SumGrad := 0;
    SumGradNorm := 0;
    
    for i := 0 to Batch - 1 do
      for s := 0 to SpatialSize - 1 do
      begin
        InputIdx := i * InputStrides[0] + c * InputStrides[1];
        SpatialIdx := s;
        for d := 2 to Input.NDim - 1 do
        begin
          DimSize := Input.Shape[d];
          DimIdx := SpatialIdx mod DimSize;
          SpatialIdx := SpatialIdx div DimSize;
          InputIdx := InputIdx + DimIdx * InputStrides[d];
        end;
        
        OutGradIdx := i * OutGradStrides[0] + c * OutGradStrides[1];
        SpatialIdx := s;
        for d := 2 to OutGrad.NDim - 1 do
        begin
          DimSize := OutGrad.Shape[d];
          DimIdx := SpatialIdx mod DimSize;
          SpatialIdx := SpatialIdx div DimSize;
          OutGradIdx := OutGradIdx + DimIdx * OutGradStrides[d];
        end;
        
        Normalized := (PInput^[InputIdx] - Mean) / Std;
        var Grad := POutGrad^[OutGradIdx];
        
        SumGrad := SumGrad + Grad;
        SumGradNorm := SumGradNorm + Grad * Normalized;
      end;
    
    // Beta gradient: sum of output gradients
    PBetaGrad^[c] := PBetaGrad^[c] + SumGrad;
    
    // Gamma gradient: sum of (output_grad * normalized_input)
    PGammaGrad^[c] := PGammaGrad^[c] + SumGradNorm;
    
    // Input gradient: more complex formula
    for i := 0 to Batch - 1 do
      for s := 0 to SpatialSize - 1 do
      begin
        InputIdx := i * InputStrides[0] + c * InputStrides[1];
        SpatialIdx := s;
        for d := 2 to Input.NDim - 1 do
        begin
          DimSize := Input.Shape[d];
          DimIdx := SpatialIdx mod DimSize;
          SpatialIdx := SpatialIdx div DimSize;
          InputIdx := InputIdx + DimIdx * InputStrides[d];
        end;
        
        OutGradIdx := i * OutGradStrides[0] + c * OutGradStrides[1];
        SpatialIdx := s;
        for d := 2 to OutGrad.NDim - 1 do
        begin
          DimSize := OutGrad.Shape[d];
          DimIdx := SpatialIdx mod DimSize;
          SpatialIdx := SpatialIdx div DimSize;
          OutGradIdx := OutGradIdx + DimIdx * OutGradStrides[d];
        end;
        
        InputGradIdx := i * InputGradStrides[0] + c * InputGradStrides[1];
        SpatialIdx := s;
        for d := 2 to InputGrad.NDim - 1 do
        begin
          DimSize := InputGrad.Shape[d];
          DimIdx := SpatialIdx mod DimSize;
          SpatialIdx := SpatialIdx div DimSize;
          InputGradIdx := InputGradIdx + DimIdx * InputGradStrides[d];
        end;
        
        Normalized := (PInput^[InputIdx] - Mean) / Std;
        var Grad := POutGrad^[OutGradIdx];
        
        // BatchNorm backward: dL/dx = (gamma / std) * (dL/dy - mean(dL/dy) - normalized * mean(dL/dy * normalized))
        var InputGradVal := (PGamma^[c] / Std) * (Grad - (SumGrad / Count) - Normalized * (SumGradNorm / Count));
        PInputGrad^[InputGradIdx] := PInputGrad^[InputGradIdx] + InputGradVal;
      end;
  end;
end;

// ============================================================================
// FUSED OPERATIONS
// ============================================================================

class procedure TOps.MatMulAdd(Arena: TArena; const A, B, Bias: TTensor; var OutT: TTensor);
var
  PA, PB, PBias, POut: PSingleArray;
  ARows, ACols, BRows, BCols: Integer;
  ANdim, BNdim, OutNdim: Integer;
  BatchSize, MatrixSizeA, MatrixSizeB, MatrixSizeOut: Integer;
  BatchIdx, RowA, ColB: Integer;
  SavePoint: TMemPtr;
  TempPtr: TMemPtr;
  PBT, PABatch, PBBatch, POutBatch: PSingleArray;
  PtrA, PtrBT: PSingle;
  UseAVX512: Boolean;
  i: Integer;
  BiasIdx: Integer;
  LocalOutShape, LocalBiasShape, LocalBiasStrides: TArray<Integer>;
begin
  // Fused MatMul + Add: Out = (A @ B) + Bias
  // Do MatMul directly into OutT, then add bias in-place (true fusion)
  
  // First, do MatMul into OutT
  ANdim := A.NDim;
  BNdim := B.NDim;
  OutNdim := OutT.NDim;
  
  if (ANdim < 2) or (BNdim < 2) then
    raise Exception.Create('MatMulAdd: Tensors must have at least 2 dimensions');
  
  ARows := A.Shape[ANdim - 2];
  ACols := A.Shape[ANdim - 1];
  BRows := B.Shape[BNdim - 2];
  BCols := B.Shape[BNdim - 1];
  
  if ACols <> BRows then
    raise Exception.CreateFmt('MatMulAdd: Inner dimension mismatch: %d != %d', [ACols, BRows]);
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  PBias := PSingleArray(Bias.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  // Compute batch size
  BatchSize := 1;
  for i := 0 to ANdim - 3 do
    BatchSize := BatchSize * A.Shape[i];
  
  MatrixSizeA := ARows * ACols;
  MatrixSizeB := BRows * BCols;
  MatrixSizeOut := ARows * BCols;
  
  SavePoint := Arena.GetSavePoint;
  TempPtr := Arena.Alloc(BRows * BCols);
  PBT := PSingleArray(Arena.GetPtr(TempPtr));
  UseAVX512 := CPUFeatures.HasAVX512;
  
  // Process each batch: MatMul then add bias in-place
  for BatchIdx := 0 to BatchSize - 1 do
  begin
    PABatch := @PA^[BatchIdx * MatrixSizeA];
    PBBatch := @PB^[BatchIdx * MatrixSizeB];
    POutBatch := @POut^[BatchIdx * MatrixSizeOut];
    
    // Transpose B for cache-friendly access
    TKernels.Transpose(PSingleArray(PBBatch), PBT, BRows, BCols);
    
    // MatMul for this batch
    TMLParallel.ForEach(0, ARows - 1,
      procedure(RowA: Integer)
      var
        ColB: Integer;
        PtrA, PtrBT: PSingle;
      begin
        PtrA := @PSingleArray(PABatch)^[RowA * ACols];
        for ColB := 0 to BCols - 1 do
        begin
          PtrBT := @PBT^[ColB * BRows];
          if UseAVX512 then
            PSingleArray(POutBatch)^[RowA * BCols + ColB] := TKernels.DotProductAVX512(PtrA, PtrBT, ACols)
          else
            PSingleArray(POutBatch)^[RowA * BCols + ColB] := TKernels.DotProduct(PtrA, PtrBT, ACols);
        end;
      end, 64);
    
    // Add bias in-place (fused - no temp allocation)
    // Bias broadcasting: Bias can be [BCols] or match output shape
    LocalOutShape := OutT.Shape;
    LocalBiasShape := Bias.Shape;
    LocalBiasStrides := Bias.Strides;
    
    if Bias.SameShape(OutT) then
    begin
      // Same shape: direct addition
      if UseAVX512 then
        TKernels.VectorAddAVX512(PSingle(POutBatch), PSingle(@PBias^[BatchIdx * MatrixSizeOut]), 
          PSingle(POutBatch), MatrixSizeOut)
      else
        TKernels.VectorAdd(PSingle(POutBatch), PSingle(@PBias^[BatchIdx * MatrixSizeOut]), 
          PSingle(POutBatch), MatrixSizeOut);
    end
    else
    begin
      // Broadcasting: add bias to each row
      for RowA := 0 to ARows - 1 do
      begin
        for ColB := 0 to BCols - 1 do
        begin
          BiasIdx := BroadcastIndex(RowA * BCols + ColB, LocalOutShape, LocalBiasShape, LocalBiasStrides);
          POutBatch^[RowA * BCols + ColB] := POutBatch^[RowA * BCols + ColB] + PBias^[BiasIdx];
        end;
      end;
    end;
  end;
  
  Arena.Restore(SavePoint);
end;

class procedure TOps.MatMulAddBackward(Arena: TArena; const A, B, Bias, OutGrad: TTensor;
  var AGrad, BGrad, BiasGrad: TTensor);
begin
  // Backward for MatMulAdd: gradients flow to MatMul and Add
  // First, Add backward (gradient flows to both MatMul output and Bias)
  var MatMulOutGrad := OutGrad;  // Gradient from Add backward
  AddBackward(Arena, OutGrad, MatMulOutGrad, BiasGrad);
  // Then MatMul backward
  MatMulBackward(Arena, A, B, MatMulOutGrad, AGrad, BGrad);
end;

class procedure TOps.AddReLU(Arena: TArena; const A, B: TTensor; var OutT: TTensor);
var
  PA, PB, POut: PSingleArray;
  OutCount: Integer;
  LocalOutShape, LocalAShape, LocalBShape, LocalAStrides, LocalBStrides: TArray<Integer>;
  i: Integer;
begin
  // Fused Add + ReLU: Out = ReLU(A + B)
  // Do Add directly into OutT, then apply ReLU in-place (true fusion, no temp allocation)
  
  OutCount := OutT.ElementCount;
  if OutCount = 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  // Fast path: same shape, no broadcasting needed
  if A.SameShape(B) then
  begin
    // Add with SIMD, then ReLU in-place
    if CPUFeatures.HasAVX512 then
      TKernels.VectorAddAVX512(PSingle(PA), PSingle(PB), PSingle(POut), OutCount)
    else
      TKernels.VectorAdd(PSingle(PA), PSingle(PB), PSingle(POut), OutCount);
    
    // Apply ReLU in-place (fused)
    for i := 0 to OutCount - 1 do
      if POut^[i] <= 0 then
        POut^[i] := 0;
  end
  else
  begin
    // Broadcasting path: Add then ReLU in-place
    LocalOutShape := OutT.Shape;
    LocalAShape := A.Shape;
    LocalBShape := B.Shape;
    LocalAStrides := A.Strides;
    LocalBStrides := B.Strides;
    
    TMLParallel.ForEach(0, OutCount - 1,
      procedure(i: Integer)
      var
        LocalIdxA, LocalIdxB: Integer;
        Sum: Single;
      begin
        LocalIdxA := BroadcastIndex(i, LocalOutShape, LocalAShape, LocalAStrides);
        LocalIdxB := BroadcastIndex(i, LocalOutShape, LocalBShape, LocalBStrides);
        Sum := PA^[LocalIdxA] + PB^[LocalIdxB];
        // Apply ReLU immediately (fused)
        if Sum > 0 then
          POut^[i] := Sum
        else
          POut^[i] := 0;
      end, 5000);
  end;
end;

class procedure TOps.AddReLUBackward(Arena: TArena; const A, B, OutGrad: TTensor;
  var AGrad, BGrad: TTensor);
var
  TempAdd: TTensor;
  i: Integer;
begin
  // Backward for AddReLU: ReLU backward then Add backward
  // Create temporary tensor for Add output (needed for ReLU backward)
  TempAdd.DataPtr := -1;  // Not allocated, just shape info
  TempAdd.GradPtr := -1;
  TempAdd.RequiresGrad := False;
  SetLength(TempAdd.Shape, Length(OutGrad.Shape));
  for i := 0 to High(OutGrad.Shape) do
    TempAdd.Shape[i] := OutGrad.Shape[i];
  TempAdd.Strides := OutGrad.Strides;
  
  // ReLU backward (gradient flows through ReLU)
  var ReLUGrad := OutGrad;
  ReLUBackward(Arena, TempAdd, OutGrad, ReLUGrad);
  // Add backward
  AddBackward(Arena, ReLUGrad, AGrad, BGrad);
end;

class procedure TOps.MatMulReLU(Arena: TArena; const A, B: TTensor; var OutT: TTensor);
var
  PA, PB, POut: PSingleArray;
  ARows, ACols, BRows, BCols: Integer;
  ANdim, BNdim: Integer;
  BatchSize, MatrixSizeA, MatrixSizeB, MatrixSizeOut: Integer;
  BatchIdx, RowA, ColB: Integer;
  SavePoint: TMemPtr;
  TempPtr: TMemPtr;
  PBT, PABatch, PBBatch, POutBatch: PSingleArray;
  PtrA, PtrBT: PSingle;
  UseAVX512: Boolean;
  i: Integer;
begin
  // Fused MatMul + ReLU: Out = ReLU(A @ B)
  // Do MatMul directly into OutT, then apply ReLU in-place (true fusion, no temp allocation)
  
  ANdim := A.NDim;
  BNdim := B.NDim;
  
  if (ANdim < 2) or (BNdim < 2) then
    raise Exception.Create('MatMulReLU: Tensors must have at least 2 dimensions');
  
  ARows := A.Shape[ANdim - 2];
  ACols := A.Shape[ANdim - 1];
  BRows := B.Shape[BNdim - 2];
  BCols := B.Shape[BNdim - 1];
  
  if ACols <> BRows then
    raise Exception.CreateFmt('MatMulReLU: Inner dimension mismatch: %d != %d', [ACols, BRows]);
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  POut := PSingleArray(OutT.RawData(Arena));
  
  // Compute batch size
  BatchSize := 1;
  for i := 0 to ANdim - 3 do
    BatchSize := BatchSize * A.Shape[i];
  
  MatrixSizeA := ARows * ACols;
  MatrixSizeB := BRows * BCols;
  MatrixSizeOut := ARows * BCols;
  
  SavePoint := Arena.GetSavePoint;
  TempPtr := Arena.Alloc(BRows * BCols);
  PBT := PSingleArray(Arena.GetPtr(TempPtr));
  UseAVX512 := CPUFeatures.HasAVX512;
  
  // Process each batch: MatMul then ReLU in-place
  for BatchIdx := 0 to BatchSize - 1 do
  begin
    PABatch := @PA^[BatchIdx * MatrixSizeA];
    PBBatch := @PB^[BatchIdx * MatrixSizeB];
    POutBatch := @POut^[BatchIdx * MatrixSizeOut];
    
    // Transpose B for cache-friendly access
    TKernels.Transpose(PSingleArray(PBBatch), PBT, BRows, BCols);
    
    // MatMul for this batch, then apply ReLU immediately (fused)
    TMLParallel.ForEach(0, ARows - 1,
      procedure(RowA: Integer)
      var
        ColB: Integer;
        PtrA, PtrBT: PSingle;
        Val: Single;
      begin
        PtrA := @PSingleArray(PABatch)^[RowA * ACols];
        for ColB := 0 to BCols - 1 do
        begin
          PtrBT := @PBT^[ColB * BRows];
          if UseAVX512 then
            Val := TKernels.DotProductAVX512(PtrA, PtrBT, ACols)
          else
            Val := TKernels.DotProduct(PtrA, PtrBT, ACols);
          // Apply ReLU immediately (fused - no separate pass)
          if Val > 0 then
            PSingleArray(POutBatch)^[RowA * BCols + ColB] := Val
          else
            PSingleArray(POutBatch)^[RowA * BCols + ColB] := 0;
        end;
      end, 64);
  end;
  
  Arena.Restore(SavePoint);
end;

class procedure TOps.MatMulReLUBackward(Arena: TArena; const A, B, OutGrad: TTensor;
  var AGrad, BGrad: TTensor);
var
  TempMatMul: TTensor;
  i: Integer;
begin
  // Backward for MatMulReLU: ReLU backward then MatMul backward
  TempMatMul.DataPtr := -1;  // Not allocated, just shape info
  TempMatMul.GradPtr := -1;
  TempMatMul.RequiresGrad := False;
  SetLength(TempMatMul.Shape, Length(OutGrad.Shape));
  for i := 0 to High(OutGrad.Shape) do
    TempMatMul.Shape[i] := OutGrad.Shape[i];
  TempMatMul.Strides := OutGrad.Strides;
  
  // ReLU backward
  var ReLUGrad := OutGrad;
  ReLUBackward(Arena, TempMatMul, OutGrad, ReLUGrad);
  // MatMul backward
  MatMulBackward(Arena, A, B, ReLUGrad, AGrad, BGrad);
end;

// ============================================================================
// CPU FEATURE DETECTION IMPLEMENTATION
// ============================================================================

// ============================================================================
// INITIALIZATION
// ============================================================================

initialization
  CPUFeatures := DetectCPUFeatures;

end.
