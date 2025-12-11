unit ML.Ops;

{$R-} // Disable range checking for performance

interface

uses
  System.Math,
  System.SysUtils,
  System.Classes,
  System.Threading,
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
  end;

  // ==========================================================================
  // PARALLEL EXECUTION - Uses RTL Thread Pool
  // ==========================================================================
  TMLParallel = class
  public
    class procedure ForEach(AStart, AEnd: Integer; const AProc: TProc<Integer>); static;
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
  end;

implementation

// ============================================================================
// MATH KERNELS IMPLEMENTATION - Pure ASM for x64, Pascal fallback for x86
// ============================================================================

{$IFDEF CPUX64}
class function TKernels.DotProduct(A, B: PSingle; Count: Integer): Single;
asm
  // RCX = A, RDX = B, R8D = Count (Windows x64 calling convention)
  .NOFRAME
  
  XORPS   XMM0, XMM0       // Clear accumulator
  
  TEST    R8D, R8D
  JLE     @Done
  
  // Main SIMD loop (4 floats at a time)
  MOV     R9D, R8D
  SHR     R9D, 2           // Count div 4
  JZ      @Tail
  
@Loop4:
  MOVUPS  XMM1, [RCX]      // Load 4 floats from A
  MOVUPS  XMM2, [RDX]      // Load 4 floats from B
  MULPS   XMM1, XMM2       // Multiply packed
  ADDPS   XMM0, XMM1       // Accumulate
  
  ADD     RCX, 16
  ADD     RDX, 16
  DEC     R9D
  JNZ     @Loop4
  
  // Horizontal add: sum all 4 floats in XMM0
  HADDPS  XMM0, XMM0       // [a+b, c+d, a+b, c+d]
  HADDPS  XMM0, XMM0       // [sum, sum, sum, sum]
  
@Tail:
  // Handle remaining 0-3 elements
  AND     R8D, 3
  JZ      @Done
  
@Loop1:
  MOVSS   XMM1, [RCX]
  MULSS   XMM1, [RDX]
  ADDSS   XMM0, XMM1
  ADD     RCX, 4
  ADD     RDX, 4
  DEC     R8D
  JNZ     @Loop1
  
@Done:
  // Result is already in XMM0 (return value for Single)
end;
{$ELSE}
class function TKernels.DotProduct(A, B: PSingle; Count: Integer): Single;
var
  i: Integer;
  Sum: Single;
begin
  Sum := 0;
  for i := 0 to Count - 1 do
    Sum := Sum + PSingleArray(A)^[i] * PSingleArray(B)^[i];
  Result := Sum;
end;
{$ENDIF}

{$IFDEF CPUX64}
class procedure TKernels.VectorAdd(A, B, Output: PSingle; Count: Integer);
asm
  // RCX = A, RDX = B, R8 = Output, R9D = Count (Windows x64)
  .NOFRAME
  
  TEST    R9D, R9D
  JLE     @Done
  
  MOV     R10D, R9D
  SHR     R10D, 2          // Count div 4
  JZ      @Tail
  
@Loop4:
  MOVUPS  XMM0, [RCX]
  MOVUPS  XMM1, [RDX]
  ADDPS   XMM0, XMM1
  MOVUPS  [R8], XMM0
  
  ADD     RCX, 16
  ADD     RDX, 16
  ADD     R8, 16
  DEC     R10D
  JNZ     @Loop4
  
@Tail:
  AND     R9D, 3
  JZ      @Done
  
@Loop1:
  MOVSS   XMM0, [RCX]
  ADDSS   XMM0, [RDX]
  MOVSS   [R8], XMM0
  ADD     RCX, 4
  ADD     RDX, 4
  ADD     R8, 4
  DEC     R9D
  JNZ     @Loop1
  
@Done:
end;
{$ELSE}
class procedure TKernels.VectorAdd(A, B, Output: PSingle; Count: Integer);
var
  i: Integer;
begin
  for i := 0 to Count - 1 do
    PSingleArray(Output)^[i] := PSingleArray(A)^[i] + PSingleArray(B)^[i];
end;
{$ENDIF}

{$IFDEF CPUX64}
class procedure TKernels.VectorMul(A, B, Output: PSingle; Count: Integer);
asm
  // RCX = A, RDX = B, R8 = Output, R9D = Count (Windows x64)
  .NOFRAME
  
  TEST    R9D, R9D
  JLE     @Done
  
  MOV     R10D, R9D
  SHR     R10D, 2
  JZ      @Tail
  
@Loop4:
  MOVUPS  XMM0, [RCX]
  MOVUPS  XMM1, [RDX]
  MULPS   XMM0, XMM1
  MOVUPS  [R8], XMM0
  
  ADD     RCX, 16
  ADD     RDX, 16
  ADD     R8, 16
  DEC     R10D
  JNZ     @Loop4
  
@Tail:
  AND     R9D, 3
  JZ      @Done
  
@Loop1:
  MOVSS   XMM0, [RCX]
  MULSS   XMM0, [RDX]
  MOVSS   [R8], XMM0
  ADD     RCX, 4
  ADD     RDX, 4
  ADD     R8, 4
  DEC     R9D
  JNZ     @Loop1
  
@Done:
end;
{$ELSE}
class procedure TKernels.VectorMul(A, B, Output: PSingle; Count: Integer);
var
  i: Integer;
begin
  for i := 0 to Count - 1 do
    PSingleArray(Output)^[i] := PSingleArray(A)^[i] * PSingleArray(B)^[i];
end;
{$ENDIF}

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

class procedure TMLParallel.ForEach(AStart, AEnd: Integer; const AProc: TProc<Integer>);
var
  i: Integer;
begin
  if AStart > AEnd then Exit;
  
  // TParallel.For has significant overhead (~1-5ms per call)
  // Only use for genuinely large workloads
  // For ML: matrix rows > 256 where each row has significant work
  if (AEnd - AStart) < 256 then
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
  ARows, ACols, BRows, BCols: Integer;
  SavePoint: Integer;
  TempPtr: TMemPtr;
  ANdim, BNdim: Integer;
  i: Integer;
begin
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
  
  // For now, handle 2D case (simple)
  // TODO: Add batched 3D+ support with proper indexing
  if (ANdim = 2) and (BNdim = 2) then
  begin
    PA := PSingleArray(A.RawData(Arena));
    PB := PSingleArray(B.RawData(Arena));
    POut := PSingleArray(OutT.RawData(Arena));
    
    // Save arena state for cleanup
    SavePoint := Arena.GetSavePoint;
    
    // Transpose B for cache-friendly sequential access
    TempPtr := Arena.Alloc(BRows * BCols);
    PBT := PSingleArray(Arena.GetPtr(TempPtr));
    TKernels.Transpose(PB, PBT, BRows, BCols);
    
    // Parallel MatMul: each row of A dot-products with all columns of B
    TMLParallel.ForEach(0, ARows - 1,
      procedure(RowA: Integer)
      var
        ColB: Integer;
        PtrA, PtrBT: PSingle;
      begin
        PtrA := @PA^[RowA * ACols];
        for ColB := 0 to BCols - 1 do
        begin
          PtrBT := @PBT^[ColB * BRows];
          POut^[RowA * BCols + ColB] := TKernels.DotProduct(PtrA, PtrBT, ACols);
        end;
      end);
      
    // Free temporary memory
    Arena.Restore(SavePoint);
  end
  else
    raise Exception.Create('MatMul: Batched N-D tensors (3D+) not yet implemented');
end;

class procedure TOps.Add(Arena: TArena; const A, B: TTensor; var OutT: TTensor);
var
  Count: Integer;
begin
  if not A.SameShape(B) then
    raise Exception.Create('Add: Shape mismatch');
    
  Count := A.ElementCount;
  if Count = 0 then Exit;
  TKernels.VectorAdd(A.RawData(Arena), B.RawData(Arena), OutT.RawData(Arena), Count);
end;

class procedure TOps.Mul(Arena: TArena; const A, B: TTensor; var OutT: TTensor);
var
  Count: Integer;
begin
  if not A.SameShape(B) then
    raise Exception.Create('Mul: Shape mismatch');
    
  Count := A.ElementCount;
  if Count = 0 then Exit;
  TKernels.VectorMul(A.RawData(Arena), B.RawData(Arena), OutT.RawData(Arena), Count);
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
  i, j, k: Integer;
  Sum: Single;
  ANdim, BNdim: Integer;
begin
  ANdim := A.NDim;
  BNdim := B.NDim;
  
  // For now, only support 2D backward
  if (ANdim <> 2) or (BNdim <> 2) or (OutGrad.NDim <> 2) then
    raise Exception.Create('MatMulBackward: Batched N-D tensors (3D+) not yet implemented');
  
  ARows := A.Shape[0];
  ACols := A.Shape[1];
  BRows := B.Shape[0];
  BCols := B.Shape[1];
  
  if (ARows = 0) or (ACols = 0) or (BRows = 0) or (BCols = 0) then Exit;
  if OutGrad.GradPtr < 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  
  // dA = OutGrad * B^T
  if AGrad.RequiresGrad and (AGrad.GradPtr >= 0) then
  begin
    PAGrad := PSingleArray(AGrad.RawGrad(Arena));
    for i := 0 to ARows - 1 do
      for j := 0 to ACols - 1 do
      begin
        Sum := 0;
        for k := 0 to BCols - 1 do
          Sum := Sum + POutGrad^[i * BCols + k] * PB^[j * BCols + k];
        PAGrad^[i * ACols + j] := PAGrad^[i * ACols + j] + Sum;
      end;
  end;
  
  // dB = A^T * OutGrad
  if BGrad.RequiresGrad and (BGrad.GradPtr >= 0) then
  begin
    PBGrad := PSingleArray(BGrad.RawGrad(Arena));
    for i := 0 to BRows - 1 do
      for j := 0 to BCols - 1 do
      begin
        Sum := 0;
        for k := 0 to ARows - 1 do
          Sum := Sum + PA^[k * ACols + i] * POutGrad^[k * BCols + j];
        PBGrad^[i * BCols + j] := PBGrad^[i * BCols + j] + Sum;
      end;
  end;
end;

class procedure TOps.AddBackward(Arena: TArena; const OutGrad: TTensor;
  var AGrad, BGrad: TTensor);
var
  POutGrad, PAGrad, PBGrad: PSingleArray;
  i, Count: Integer;
begin
  Count := OutGrad.ElementCount;
  if Count = 0 then Exit;
  if (OutGrad.GradPtr < 0) or (AGrad.GradPtr < 0) or (BGrad.GradPtr < 0) then Exit;
  
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PAGrad := PSingleArray(AGrad.RawGrad(Arena));
  PBGrad := PSingleArray(BGrad.RawGrad(Arena));
  
  for i := 0 to Count - 1 do
  begin
    PAGrad^[i] := PAGrad^[i] + POutGrad^[i];
    PBGrad^[i] := PBGrad^[i] + POutGrad^[i];
  end;
end;

class procedure TOps.MulBackward(Arena: TArena; const A, B, OutGrad: TTensor;
  var AGrad, BGrad: TTensor);
var
  PA, PB, POutGrad, PAGrad, PBGrad: PSingleArray;
  i, Count: Integer;
begin
  Count := A.ElementCount;
  if Count = 0 then Exit;
  if OutGrad.GradPtr < 0 then Exit;
  
  PA := PSingleArray(A.RawData(Arena));
  PB := PSingleArray(B.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  
  if AGrad.GradPtr >= 0 then
  begin
    PAGrad := PSingleArray(AGrad.RawGrad(Arena));
    for i := 0 to Count - 1 do
      PAGrad^[i] := PAGrad^[i] + POutGrad^[i] * PB^[i];
  end;
  
  if BGrad.GradPtr >= 0 then
  begin
    PBGrad := PSingleArray(BGrad.RawGrad(Arena));
    for i := 0 to Count - 1 do
      PBGrad^[i] := PBGrad^[i] + POutGrad^[i] * PA^[i];
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
  i, Count: Integer;
begin
  // NOTE: This is a simplified approximation. Full Softmax backward requires
  // the Jacobian matrix. For accurate gradients, use SoftmaxCrossEntropyBackward
  // which combines Softmax + CrossEntropy with the simple gradient: Pred - Target
  Count := Out.ElementCount;
  if Count = 0 then Exit;
  if (OutGrad.GradPtr < 0) or (AGrad.GradPtr < 0) then Exit;
  
  POut := PSingleArray(Out.RawData(Arena));
  POutGrad := PSingleArray(OutGrad.RawGrad(Arena));
  PAGrad := PSingleArray(AGrad.RawGrad(Arena));
  
  // Approximation using diagonal of Jacobian
  for i := 0 to Count - 1 do
    PAGrad^[i] := PAGrad^[i] + POutGrad^[i] * POut^[i] * (1.0 - POut^[i]);
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

end.
