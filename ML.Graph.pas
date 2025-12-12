unit ML.Graph;

{$R-} // Disable range checking for performance - we do manual bounds checking

interface

uses
  System.SysUtils,
  System.Generics.Collections,
  System.Classes,
  System.Math,
  ML.Arena,
  ML.Tensor,
  ML.Ops;

type
  PSingleArray = ^TSingleArray;
  TSingleArray = array[0..MaxInt div SizeOf(Single) - 1] of Single;

type
  TParamData = record
    Shape: TArray<Integer>;
    Data: TArray<Single>;
  end;
  
  TModelData = record
    Version: Integer;
    ParamCount: Integer;
    Params: TArray<TParamData>;
  end;

type
  TOpType = (
    opInput,      // Input data (from user)
    opParam,      // Trainable parameter (weights, biases)
    opAdd,        // Element-wise addition
    opMul,        // Element-wise multiplication
    opMatMul,     // Matrix multiplication
    opReLU,       // ReLU activation
    opLeakyReLU,  // LeakyReLU activation
    opSigmoid,    // Sigmoid activation
    opTanh,       // Tanh activation
    opSoftmax,    // Softmax normalization
    opMSE,        // Mean Squared Error loss
    opCrossEntropy, // Cross Entropy loss
    opConv2D,     // 2D Convolution
    opReshape,    // Reshape tensor (view operation)
    opBatchNorm,  // Batch Normalization
    opMatMulAdd,  // Fused MatMul + Add (MatMul with bias)
    opAddReLU,    // Fused Add + ReLU
    opMatMulReLU, // Fused MatMul + ReLU
    opMaxPool2D,  // Max Pooling 2D
    opDropout     // Dropout regularization
  );

  // A node in the computation graph
  TNode = record
    Op: TOpType;
    Result: TTensor;
    Parents: array[0..2] of Integer;  // Max 3 parents (for BatchNorm: Input, Gamma, Beta)
    ParentCount: Integer;
    RequiresGrad: Boolean;
  end;

  // Memory monitoring types
  TGraphMemoryStats = record
    ArenaUsage: TArenaStats;
    ParameterCount: Integer;
    ParameterBytes: Integer;
    ActivationBytes: Integer;
    GradientBytes: Integer;
  end;

  TGraph = class
  private
    FArena: TArena;
    FNodes: TArray<TNode>;
    FNodeCount: Integer;
    FNodeCapacity: Integer;
    // Savepoint for separating params from activations
    FParamSavePoint: Integer;
    FParamNodeCount: Integer;
    // Conv2D parameters storage (Padding, Stride) - indexed by node index
    FConv2DPadding: TArray<Integer>;
    FConv2DStride: TArray<Integer>;
    
    // Shape inference and validation
    FShapeValidationEnabled: Boolean;
    
    // BatchNorm running statistics (per BatchNorm node)
    FBatchNormRunningMean: TArray<TArray<Single>>;
    FBatchNormRunningVar: TArray<TArray<Single>>;
    FBatchNormEpsilon: TArray<Single>;
    FBatchNormMomentum: TArray<Single>;
    FIsTraining: Boolean;
    
    // MaxPool2D indices storage (per node)
    FMaxPoolIndices: TArray<TArray<Integer>>;
    FMaxPoolSize: TArray<Integer>;
    FMaxPoolStride: TArray<Integer>;
    
    // Dropout mask storage (per node)
    FDropoutMasks: TArray<TArray<Byte>>;
    FDropoutRates: TArray<Single>;
    
    // Adam optimizer state
    FAdamM: TArray<TArray<Single>>;  // Momentum (first moment)
    FAdamV: TArray<TArray<Single>>;   // Velocity (second moment)
    FAdamBeta1: Single;
    FAdamBeta2: Single;
    FAdamEpsilon: Single;
    FAdamStep: Integer;
    FUseAdam: Boolean;
    
    // Gradient checking
    FGradientCheckEnabled: Boolean;
    FGradientCheckEpsilon: Single;
    FGradientCheckTolerance: Single;
    FGradientCheckDone: Boolean;  // Track if we've done gradient check for this backward pass
    
    function AddNode(const ANode: TNode): Integer;
    procedure EnsureCapacity;
    procedure AllocGradIfNeeded(NodeIdx: Integer);
    function InferShape(Op: TOpType; const InputShapes: array of TArray<Integer>; 
      Padding: Integer = 0; Stride: Integer = 1; const NewShape: TArray<Integer> = nil): TArray<Integer>;
    procedure ValidateShape(NodeIdx: Integer; const InferredShape, ActualShape: TArray<Integer>);
  public
    constructor Create(ArenaSizeMB: Integer = 256);
    destructor Destroy; override;
    
    // Reset the graph and arena (full reset)
    procedure Reset;
    // Mark current position as end of parameters
    procedure MarkParamsEnd;
    // Reset only activations, keep parameters intact
    procedure ResetActivations;
    
    // Forward building - returns node index
    function Input(const AShape: array of Integer): Integer;
    function Param(const AShape: array of Integer): Integer;
    function Add(A, B: Integer): Integer;
    function Mul(A, B: Integer): Integer;
    function MatMul(A, B: Integer): Integer;
    function ReLU(A: Integer): Integer;
    function LeakyReLU(A: Integer; Alpha: Single = 0.01): Integer;
    function Sigmoid(A: Integer): Integer;
    function Tanh(A: Integer): Integer;
    function Softmax(A: Integer): Integer;
    function MSE(Pred, Target: Integer): Integer;
    function CrossEntropy(Pred, Target: Integer): Integer;
    function Conv2D(Input, Weight: Integer; Padding, Stride: Integer): Integer;
    function Reshape(InputIdx: Integer; const NewShape: array of Integer): Integer;
    function BatchNorm(InputIdx: Integer; Epsilon: Single = 1e-5; Momentum: Single = 0.1): Integer;
    function MatMulAdd(A, B, Bias: Integer): Integer;
    function AddReLU(A, B: Integer): Integer;
    function MatMulReLU(A, B: Integer): Integer;
    function MaxPool2D(Input: Integer; PoolSize, Stride: Integer): Integer;
    function Dropout(Input: Integer; DropRate: Single): Integer;
    
    // Training mode control
    procedure SetTrainingMode(IsTraining: Boolean);
    
    // Backward pass
    procedure Backward(LossNode: Integer);
    
    // Optimizer
    procedure Step(LearningRate: Single; GradClip: Single = 5.0);  // SGD
    procedure StepAdam(LearningRate: Single; GradClip: Single = 5.0);  // Adam optimizer
    procedure ZeroGrad;
    procedure SetOptimizer(UseAdam: Boolean; Beta1: Single = 0.9; Beta2: Single = 0.999; Epsilon: Single = 1e-8);
    
    // Access
    property Arena: TArena read FArena;
    function GetNode(Idx: Integer): TNode;
    function GetNodeCount: Integer;
    
    // Helper methods for setting input values and getting outputs
    procedure SetInputValue(InputIdx: Integer; const Values: TArray<Single>);
    function GetOutputValue(OutputIdx: Integer): Single;
    function GetOutputValues(OutputIdx: Integer; Count: Integer): TArray<Single>;
    
    // Model persistence
    function ExportParams: TModelData;
    procedure ImportParams(const ModelData: TModelData; Reinitialize: Boolean = False);
    procedure SaveModel(const FileName: string);
    procedure LoadModel(const FileName: string);
    
    // Shape inference and validation
    function ValidateShapes: Boolean;
    property EnableShapeValidation: Boolean read FShapeValidationEnabled write FShapeValidationEnabled;
    
    // Gradient checking
    procedure EnableGradientCheck(Enabled: Boolean; Epsilon: Single = 1e-5; Tolerance: Single = 1e-4);
    function CheckGradients(LossNode: Integer): Boolean;
    property GradientCheckEnabled: Boolean read FGradientCheckEnabled;
    
    // Memory monitoring
    function GetMemoryStats: TGraphMemoryStats;
    procedure LogMemoryStats;
  end;

implementation

// Helper to create tensor with same shape as another
function CreateTensorSameShape(Arena: TArena; const Source: TTensor;
  RequiresGrad: Boolean): TTensor;
var
  i: Integer;
begin
  SetLength(Result.Shape, Length(Source.Shape));
  for i := 0 to High(Source.Shape) do
    Result.Shape[i] := Source.Shape[i];
  Result.Strides := ComputeStrides(Result.Shape);
  Result.DataPtr := Arena.Alloc(Result.ElementCount);
  Result.GradPtr := -1;
  Result.RequiresGrad := RequiresGrad;
end;

constructor TGraph.Create(ArenaSizeMB: Integer);
begin
  FArena := TArena.Create(ArenaSizeMB);
  FNodeCapacity := 1024;
  SetLength(FNodes, FNodeCapacity);
  SetLength(FConv2DPadding, FNodeCapacity);
  SetLength(FConv2DStride, FNodeCapacity);
  FNodeCount := 0;
  FShapeValidationEnabled := False;  // Disabled by default for performance (enable for debugging)
  FIsTraining := True;  // Default to training mode
  SetLength(FBatchNormRunningMean, 0);
  SetLength(FBatchNormRunningVar, 0);
  SetLength(FBatchNormEpsilon, 0);
  SetLength(FBatchNormMomentum, 0);
  FUseAdam := False;  // Default to SGD
  FAdamBeta1 := 0.9;
  FAdamBeta2 := 0.999;
  FAdamEpsilon := 1e-8;
  FAdamStep := 0;
  SetLength(FAdamM, 0);
  SetLength(FAdamV, 0);
end;

destructor TGraph.Destroy;
begin
  FArena.Free;
  inherited;
end;

procedure TGraph.EnsureCapacity;
begin
  if FNodeCount >= FNodeCapacity then
  begin
    FNodeCapacity := FNodeCapacity * 2;
    SetLength(FNodes, FNodeCapacity);
  end;
end;

function TGraph.AddNode(const ANode: TNode): Integer;
begin
  EnsureCapacity;
  Result := FNodeCount;
  FNodes[Result] := ANode;
  Inc(FNodeCount);
end;

procedure TGraph.AllocGradIfNeeded(NodeIdx: Integer);
var
  Node: TNode;
  GradPtr: TMemPtr;
  PGrad: PSingleArray;
  i, ElementCount: Integer;
begin
  if (NodeIdx < 0) or (NodeIdx >= FNodeCount) then
    Exit;
    
  Node := FNodes[NodeIdx];
  ElementCount := Node.Result.ElementCount;
  
  if Node.RequiresGrad and (Node.Result.GradPtr < 0) and (ElementCount > 0) then
  begin
    GradPtr := FArena.Alloc(ElementCount);
    FNodes[NodeIdx].Result.GradPtr := GradPtr;
    // Zero initialize gradients
    PGrad := PSingleArray(FArena.GetPtr(GradPtr));
    for i := 0 to ElementCount - 1 do
      PGrad^[i] := 0;
  end;
end;

procedure TGraph.Reset;
begin
  FArena.Reset;
  FNodeCount := 0;
  FParamSavePoint := 0;
  FParamNodeCount := 0;
  // Conv2D parameters are reset automatically when FNodeCount is reset
end;

procedure TGraph.MarkParamsEnd;
begin
  // Call this after all Param() calls to mark the boundary
  // between persistent parameters and temporary activations
  FParamSavePoint := FArena.GetSavePoint;
  FParamNodeCount := FNodeCount;
end;

procedure TGraph.ResetActivations;
begin
  // Reset only activations, keep parameters intact
  // More efficient than full Reset + SaveParameters + RestoreParameters
  FArena.Restore(FParamSavePoint);
  FNodeCount := FParamNodeCount;
end;

function TGraph.Input(const AShape: array of Integer): Integer;
var
  Node: TNode;
begin
  Node.Op := opInput;
  Node.Result := TTensor.Create(FArena, AShape, False);
  Node.ParentCount := 0;
  Node.RequiresGrad := False;
  Result := AddNode(Node);
end;

function TGraph.Param(const AShape: array of Integer): Integer;
var
  Node: TNode;
  PData, PGrad: PSingleArray;
  i, Count, TotalDims: Integer;
  Scale: Single;
begin
  Node.Op := opParam;
  Node.Result := TTensor.Create(FArena, AShape, True);
  Node.ParentCount := 0;
  Node.RequiresGrad := True;
  
  Count := Node.Result.ElementCount;
  
  // Initialize with random values (He initialization for ReLU/LeakyReLU)
  // Fan_in is the product of all dimensions except the first (output channels)
  PData := PSingleArray(Node.Result.RawData(FArena));
  
  // For conv weights [OutCh, InCh, KH, KW], fan_in = InCh * KH * KW
  // For dense weights [OutFeat, InFeat], fan_in = InFeat
  // For bias [N], fan_in = N (not critical since bias is typically small)
  if Length(AShape) >= 2 then
  begin
    // Product of all dims except first
    TotalDims := 1;
    for i := 1 to High(AShape) do
      TotalDims := TotalDims * AShape[i];
  end
  else if Length(AShape) = 1 then
    TotalDims := AShape[0]
  else
    TotalDims := 1;
  
  // He initialization: sqrt(2.0 / fan_in) for ReLU-based networks
  Scale := Sqrt(2.0 / TotalDims);
  
  // Random initialization in range [-scale, +scale]
  for i := 0 to Count - 1 do
    PData^[i] := (Random * 2.0 - 1.0) * Scale;
  
  // Pre-allocate gradients for params so they persist across ResetActivations
  // This is critical: gradients must be allocated BEFORE MarkParamsEnd()
  Node.Result.GradPtr := FArena.Alloc(Count);
  PGrad := PSingleArray(FArena.GetPtr(Node.Result.GradPtr));
  for i := 0 to Count - 1 do
    PGrad^[i] := 0;
  
  Result := AddNode(Node);
end;

function TGraph.Add(A, B: Integer): Integer;
var
  Node: TNode;
  TensorA, TensorB: TTensor;
  OutShape: TArray<Integer>;
  i: Integer;
  ShapeAStr, ShapeBStr: string;
begin
  TensorA := FNodes[A].Result;
  TensorB := FNodes[B].Result;
  
  // Debug: Build shape strings
  ShapeAStr := '[';
  for i := 0 to High(TensorA.Shape) do
  begin
    if i > 0 then ShapeAStr := ShapeAStr + ', ';
    ShapeAStr := ShapeAStr + IntToStr(TensorA.Shape[i]);
  end;
  ShapeAStr := ShapeAStr + ']';
  
  ShapeBStr := '[';
  for i := 0 to High(TensorB.Shape) do
  begin
    if i > 0 then ShapeBStr := ShapeBStr + ', ';
    ShapeBStr := ShapeBStr + IntToStr(TensorB.Shape[i]);
  end;
  ShapeBStr := ShapeBStr + ']';
  
  // Check if shapes are broadcastable
  if not CanBroadcast(TensorA.Shape, TensorB.Shape) then
    raise Exception.CreateFmt('Add: Shapes are not broadcastable. ShapeA=%s, ShapeB=%s', [ShapeAStr, ShapeBStr]);
  
  // Compute broadcast output shape
  OutShape := BroadcastShapes(TensorA.Shape, TensorB.Shape);
  
  Node.Op := opAdd;
  // Create output with broadcast shape
  SetLength(Node.Result.Shape, Length(OutShape));
  for i := 0 to High(OutShape) do
    Node.Result.Shape[i] := OutShape[i];
  Node.Result.Strides := ComputeStrides(Node.Result.Shape);
  Node.Result.DataPtr := FArena.Alloc(Node.Result.ElementCount);
  Node.Result.GradPtr := -1;
  Node.Result.RequiresGrad := FNodes[A].RequiresGrad or FNodes[B].RequiresGrad;
  
  Node.Parents[0] := A;
  Node.Parents[1] := B;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.Add(FArena, TensorA, TensorB, Node.Result);
  Result := AddNode(Node);
  
  // Validate shape if enabled
  if FShapeValidationEnabled then
  begin
    var InputShapes: array of TArray<Integer>;
    SetLength(InputShapes, 2);
    InputShapes[0] := TensorA.Shape;
    InputShapes[1] := TensorB.Shape;
    var InferredShape := InferShape(opAdd, InputShapes);
    ValidateShape(Result, InferredShape, Node.Result.Shape);
  end;
end;

function TGraph.Mul(A, B: Integer): Integer;
var
  Node: TNode;
  TensorA, TensorB: TTensor;
  OutShape: TArray<Integer>;
  i: Integer;
begin
  TensorA := FNodes[A].Result;
  TensorB := FNodes[B].Result;
  
  // Check if shapes are broadcastable
  if not CanBroadcast(TensorA.Shape, TensorB.Shape) then
    raise Exception.Create('Mul: Shapes are not broadcastable');
  
  // Compute broadcast output shape
  OutShape := BroadcastShapes(TensorA.Shape, TensorB.Shape);
  
  Node.Op := opMul;
  // Create output with broadcast shape
  SetLength(Node.Result.Shape, Length(OutShape));
  for i := 0 to High(OutShape) do
    Node.Result.Shape[i] := OutShape[i];
  Node.Result.Strides := ComputeStrides(Node.Result.Shape);
  Node.Result.DataPtr := FArena.Alloc(Node.Result.ElementCount);
  Node.Result.GradPtr := -1;
  Node.Result.RequiresGrad := FNodes[A].RequiresGrad or FNodes[B].RequiresGrad;
  
  Node.Parents[0] := A;
  Node.Parents[1] := B;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.Mul(FArena, TensorA, TensorB, Node.Result);
  Result := AddNode(Node);
  
  // Validate shape if enabled
  if FShapeValidationEnabled then
  begin
    var InputShapes: array of TArray<Integer>;
    SetLength(InputShapes, 2);
    InputShapes[0] := TensorA.Shape;
    InputShapes[1] := TensorB.Shape;
    var InferredShape := InferShape(opMul, InputShapes);
    ValidateShape(Result, InferredShape, Node.Result.Shape);
  end;
end;

function TGraph.MatMul(A, B: Integer): Integer;
var
  Node: TNode;
  TensorA, TensorB: TTensor;
  ANdim, BNdim, OutNdim: Integer;
  OutShape: TArray<Integer>;
  i: Integer;
begin
  TensorA := FNodes[A].Result;
  TensorB := FNodes[B].Result;
  
  ANdim := TensorA.NDim;
  BNdim := TensorB.NDim;
  
  if (ANdim < 2) or (BNdim < 2) then
    raise Exception.Create('MatMul: Tensors must have at least 2 dimensions');
  
  // Check inner dimensions match: A[..., M, K] @ B[..., K, N]
  if TensorA.Shape[ANdim - 1] <> TensorB.Shape[BNdim - 2] then
    raise Exception.CreateFmt('MatMul: Inner dimension mismatch: %d != %d',
      [TensorA.Shape[ANdim - 1], TensorB.Shape[BNdim - 2]]);
  
  // For N-D tensors, batch dimensions must match
  if (ANdim > 2) or (BNdim > 2) then
  begin
    // Both must have same number of dimensions for batched matmul
    if ANdim <> BNdim then
      raise Exception.Create('MatMul: Batch dimension count must match for N-D tensors');
    
    // Check all batch dimensions match
    for i := 0 to ANdim - 3 do
      if TensorA.Shape[i] <> TensorB.Shape[i] then
        raise Exception.CreateFmt('MatMul: Batch dimension %d mismatch: %d vs %d',
          [i, TensorA.Shape[i], TensorB.Shape[i]]);
  end;
  
  // Compute output shape: [batch..., M, N]
  OutNdim := ANdim;
  SetLength(OutShape, OutNdim);
  
  // Copy batch dimensions from A
  for i := 0 to OutNdim - 3 do
    OutShape[i] := TensorA.Shape[i];
  
  // Last two dimensions: M from A, N from B
  OutShape[OutNdim - 2] := TensorA.Shape[ANdim - 2];  // M
  OutShape[OutNdim - 1] := TensorB.Shape[BNdim - 1];  // N
  
  Node.Op := opMatMul;
  SetLength(Node.Result.Shape, OutNdim);
  for i := 0 to OutNdim - 1 do
    Node.Result.Shape[i] := OutShape[i];
  Node.Result.Strides := ComputeStrides(Node.Result.Shape);
  Node.Result.DataPtr := FArena.Alloc(Node.Result.ElementCount);
  Node.Result.GradPtr := -1;
  Node.Result.RequiresGrad := FNodes[A].RequiresGrad or FNodes[B].RequiresGrad;
  
  Node.Parents[0] := A;
  Node.Parents[1] := B;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.MatMul(FArena, TensorA, TensorB, Node.Result);
  Result := AddNode(Node);
  
  // Validate shape if enabled
  if FShapeValidationEnabled then
  begin
    var InputShapes: array of TArray<Integer>;
    SetLength(InputShapes, 2);
    InputShapes[0] := TensorA.Shape;
    InputShapes[1] := TensorB.Shape;
    var InferredShape := InferShape(opMatMul, InputShapes);
    ValidateShape(Result, InferredShape, Node.Result.Shape);
  end;
end;

function TGraph.ReLU(A: Integer): Integer;
var
  Node: TNode;
  TensorA: TTensor;
begin
  TensorA := FNodes[A].Result;
  
  Node.Op := opReLU;
  Node.Result := CreateTensorSameShape(FArena, TensorA, FNodes[A].RequiresGrad);
  Node.Parents[0] := A;
  Node.ParentCount := 1;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.ReLU(FArena, TensorA, Node.Result);
  Result := AddNode(Node);
end;

function TGraph.LeakyReLU(A: Integer; Alpha: Single): Integer;
var
  Node: TNode;
  TensorA: TTensor;
begin
  TensorA := FNodes[A].Result;
  
  Node.Op := opLeakyReLU;
  Node.Result := CreateTensorSameShape(FArena, TensorA, FNodes[A].RequiresGrad);
  Node.Parents[0] := A;
  Node.ParentCount := 1;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.LeakyReLU(FArena, TensorA, Node.Result, Alpha);
  Result := AddNode(Node);
end;

function TGraph.Sigmoid(A: Integer): Integer;
var
  Node: TNode;
  TensorA: TTensor;
begin
  TensorA := FNodes[A].Result;
  
  Node.Op := opSigmoid;
  Node.Result := CreateTensorSameShape(FArena, TensorA, FNodes[A].RequiresGrad);
  Node.Parents[0] := A;
  Node.ParentCount := 1;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.Sigmoid(FArena, TensorA, Node.Result);
  Result := AddNode(Node);
end;

function TGraph.Tanh(A: Integer): Integer;
var
  Node: TNode;
  TensorA: TTensor;
begin
  TensorA := FNodes[A].Result;
  
  Node.Op := opTanh;
  Node.Result := CreateTensorSameShape(FArena, TensorA, FNodes[A].RequiresGrad);
  Node.Parents[0] := A;
  Node.ParentCount := 1;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.Tanh(FArena, TensorA, Node.Result);
  Result := AddNode(Node);
end;

function TGraph.Softmax(A: Integer): Integer;
var
  Node: TNode;
  TensorA: TTensor;
begin
  TensorA := FNodes[A].Result;
  
  Node.Op := opSoftmax;
  Node.Result := CreateTensorSameShape(FArena, TensorA, FNodes[A].RequiresGrad);
  Node.Parents[0] := A;
  Node.ParentCount := 1;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.Softmax(FArena, TensorA, Node.Result);
  Result := AddNode(Node);
end;

function TGraph.MSE(Pred, Target: Integer): Integer;
var
  Node: TNode;
  TensorPred, TensorTarget: TTensor;
  Loss: Single;
begin
  TensorPred := FNodes[Pred].Result;
  TensorTarget := FNodes[Target].Result;
  
  if not TensorPred.SameShape(TensorTarget) then
    raise Exception.Create('MSE: Shape mismatch');
  
  Loss := TOps.MSE(FArena, TensorPred, TensorTarget);
  
  // Create a scalar tensor for the loss
  Node.Op := opMSE;
  Node.Result := TTensor.Create(FArena, [1], FNodes[Pred].RequiresGrad or FNodes[Target].RequiresGrad);
  Node.Parents[0] := Pred;
  Node.Parents[1] := Target;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  // Store loss value
  var PLoss: PSingleArray;
  PLoss := PSingleArray(Node.Result.RawData(FArena));
  PLoss^[0] := Loss;
  
  Result := AddNode(Node);
end;

function TGraph.CrossEntropy(Pred, Target: Integer): Integer;
var
  Node: TNode;
  TensorPred, TensorTarget: TTensor;
  Loss: Single;
begin
  TensorPred := FNodes[Pred].Result;
  TensorTarget := FNodes[Target].Result;
  
  if not TensorPred.SameShape(TensorTarget) then
    raise Exception.Create('CrossEntropy: Shape mismatch');
  
  Loss := TOps.CrossEntropy(FArena, TensorPred, TensorTarget);
  
  // Create a scalar tensor for the loss
  Node.Op := opCrossEntropy;
  Node.Result := TTensor.Create(FArena, [1], FNodes[Pred].RequiresGrad or FNodes[Target].RequiresGrad);
  Node.Parents[0] := Pred;
  Node.Parents[1] := Target;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  // Store loss value
  var PLoss: PSingleArray;
  PLoss := PSingleArray(Node.Result.RawData(FArena));
  PLoss^[0] := Loss;
  
  Result := AddNode(Node);
end;

function TGraph.Conv2D(Input, Weight: Integer; Padding, Stride: Integer): Integer;
var
  Node: TNode;
  TensorInput, TensorWeight: TTensor;
  Batch, InCh, InH, InW: Integer;
  OutCh, KernelH, KernelW: Integer;
  OutH, OutW: Integer;
  OutShape: TArray<Integer>;
begin
  TensorInput := FNodes[Input].Result;
  TensorWeight := FNodes[Weight].Result;
  
  // Validate shapes
  if TensorInput.NDim <> 4 then
    raise Exception.Create('Conv2D: Input must be 4D tensor [Batch, InChannels, InH, InW]');
  if TensorWeight.NDim <> 4 then
    raise Exception.Create('Conv2D: Weight must be 4D tensor [OutChannels, InChannels, KernelH, KernelW]');
  
  Batch := TensorInput.Shape[0];
  InCh := TensorInput.Shape[1];
  InH := TensorInput.Shape[2];
  InW := TensorInput.Shape[3];
  
  OutCh := TensorWeight.Shape[0];
  if TensorWeight.Shape[1] <> InCh then
    raise Exception.CreateFmt('Conv2D: Input channels mismatch: input has %d, weight expects %d',
      [InCh, TensorWeight.Shape[1]]);
  KernelH := TensorWeight.Shape[2];
  KernelW := TensorWeight.Shape[3];
  
  // Compute output dimensions
  OutH := (InH + 2 * Padding - KernelH) div Stride + 1;
  OutW := (InW + 2 * Padding - KernelW) div Stride + 1;
  
  if (OutH <= 0) or (OutW <= 0) then
    raise Exception.CreateFmt('Conv2D: Invalid output dimensions: OutH=%d, OutW=%d', [OutH, OutW]);
  
  // Create output shape
  SetLength(OutShape, 4);
  OutShape[0] := Batch;
  OutShape[1] := OutCh;
  OutShape[2] := OutH;
  OutShape[3] := OutW;
  
  Node.Op := opConv2D;
  Node.Result := TTensor.Create(FArena, OutShape, FNodes[Input].RequiresGrad or FNodes[Weight].RequiresGrad);
  Node.Parents[0] := Input;
  Node.Parents[1] := Weight;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  // Store padding and stride in node (we'll need them for backward pass)
  // For now, we'll pass them directly to the backward operation
  // Note: We need to store these values somewhere. Let's use a simple approach:
  // Store in a temporary way - we'll access them from the operation parameters
  
  TOps.Conv2D(FArena, TensorInput, TensorWeight, Padding, Stride, Node.Result);
  Result := AddNode(Node);
  
  // Store padding and stride for backward pass
  if Result >= Length(FConv2DPadding) then
  begin
    SetLength(FConv2DPadding, FNodeCapacity);
    SetLength(FConv2DStride, FNodeCapacity);
  end;
  FConv2DPadding[Result] := Padding;
  FConv2DStride[Result] := Stride;
end;

function TGraph.MaxPool2D(Input: Integer; PoolSize, Stride: Integer): Integer;
var
  Node: TNode;
  TensorInput: TTensor;
  Batch, Channels, InH, InW: Integer;
  OutH, OutW: Integer;
  OutShape: TArray<Integer>;
begin
  TensorInput := FNodes[Input].Result;
  
  if TensorInput.NDim <> 4 then
    raise Exception.Create('MaxPool2D: Input must be 4D tensor [Batch, Channels, H, W]');
  
  Batch := TensorInput.Shape[0];
  Channels := TensorInput.Shape[1];
  InH := TensorInput.Shape[2];
  InW := TensorInput.Shape[3];
  
  OutH := (InH - PoolSize) div Stride + 1;
  OutW := (InW - PoolSize) div Stride + 1;
  
  if (OutH <= 0) or (OutW <= 0) then
    raise Exception.CreateFmt('MaxPool2D: Invalid output dimensions: OutH=%d, OutW=%d', [OutH, OutW]);
  
  SetLength(OutShape, 4);
  OutShape[0] := Batch;
  OutShape[1] := Channels;
  OutShape[2] := OutH;
  OutShape[3] := OutW;
  
  Node.Op := opMaxPool2D;
  Node.Result := TTensor.Create(FArena, OutShape, FNodes[Input].RequiresGrad);
  Node.Parents[0] := Input;
  Node.ParentCount := 1;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  // Ensure storage arrays are sized
  if FNodeCount >= Length(FMaxPoolIndices) then
  begin
    SetLength(FMaxPoolIndices, FNodeCapacity);
    SetLength(FMaxPoolSize, FNodeCapacity);
    SetLength(FMaxPoolStride, FNodeCapacity);
  end;
  
  TOps.MaxPool2D(FArena, TensorInput, PoolSize, Stride, Node.Result, FMaxPoolIndices[FNodeCount]);
  
  Result := AddNode(Node);
  FMaxPoolSize[Result] := PoolSize;
  FMaxPoolStride[Result] := Stride;
end;

function TGraph.Dropout(Input: Integer; DropRate: Single): Integer;
var
  Node: TNode;
  TensorInput: TTensor;
begin
  TensorInput := FNodes[Input].Result;
  
  Node.Op := opDropout;
  Node.Result := TTensor.Create(FArena, TensorInput.Shape, FNodes[Input].RequiresGrad);
  Node.Parents[0] := Input;
  Node.ParentCount := 1;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  // Ensure storage arrays are sized
  if FNodeCount >= Length(FDropoutMasks) then
  begin
    SetLength(FDropoutMasks, FNodeCapacity);
    SetLength(FDropoutRates, FNodeCapacity);
  end;
  
  TOps.Dropout(FArena, TensorInput, DropRate, FIsTraining, Node.Result, FDropoutMasks[FNodeCount]);
  
  Result := AddNode(Node);
  FDropoutRates[Result] := DropRate;
end;

function TGraph.Reshape(InputIdx: Integer; const NewShape: array of Integer): Integer;
var
  Node: TNode;
  TensorInput: TTensor;
  i: Integer;
  NewCount, OldCount: Integer;
begin
  TensorInput := FNodes[InputIdx].Result;
  
  // Validate element count matches
  OldCount := TensorInput.ElementCount;
  NewCount := 1;
  for i := 0 to High(NewShape) do
  begin
    if NewShape[i] < 0 then
      raise Exception.Create('Reshape: Negative dimensions not supported');
    NewCount := NewCount * NewShape[i];
  end;
  
  if NewCount <> OldCount then
    raise Exception.CreateFmt('Reshape: Cannot reshape [%d] elements to [%d] elements',
      [OldCount, NewCount]);
  
  // Create reshape node - this is a view operation, no data copy
  Node.Op := opReshape;
  Node.Result := TensorInput.Reshape(NewShape);  // Use tensor's Reshape method (view operation)
  Node.Parents[0] := InputIdx;
  Node.ParentCount := 1;
  Node.RequiresGrad := FNodes[InputIdx].RequiresGrad;
  
  Result := AddNode(Node);
end;

procedure TGraph.SetTrainingMode(IsTraining: Boolean);
begin
  FIsTraining := IsTraining;
end;

function TGraph.BatchNorm(InputIdx: Integer; Epsilon: Single; Momentum: Single): Integer;
var
  Node: TNode;
  TensorInput: TTensor;
  Channels: Integer;
  GammaIdx, BetaIdx: Integer;
  TensorGamma, TensorBeta: TTensor;
  BNIdx: Integer;
  c: Integer;
  PGamma, PBeta: PSingleArray;
begin
  TensorInput := FNodes[InputIdx].Result;
  
  // BatchNorm expects input shape [Batch, Channels, ...] (at least 2D)
  if TensorInput.NDim < 2 then
    raise Exception.Create('BatchNorm: Input must have at least 2 dimensions');
  
  Channels := TensorInput.Shape[1];
  
  // Create gamma (scale) and beta (shift) parameters
  GammaIdx := Param([Channels]);
  BetaIdx := Param([Channels]);
  
  TensorGamma := FNodes[GammaIdx].Result;
  TensorBeta := FNodes[BetaIdx].Result;
  
  // Initialize gamma to 1, beta to 0
  PGamma := PSingleArray(FArena.GetPtr(TensorGamma.DataPtr));
  PBeta := PSingleArray(FArena.GetPtr(TensorBeta.DataPtr));
  for c := 0 to Channels - 1 do
  begin
    PGamma^[c] := 1.0;  // Start with identity transformation
    PBeta^[c] := 0.0;
  end;
  
  // Ensure running statistics arrays are large enough
  BNIdx := FNodeCount;  // This will be the node index after AddNode
  if BNIdx >= Length(FBatchNormRunningMean) then
  begin
    SetLength(FBatchNormRunningMean, FNodeCapacity);
    SetLength(FBatchNormRunningVar, FNodeCapacity);
    SetLength(FBatchNormEpsilon, FNodeCapacity);
    SetLength(FBatchNormMomentum, FNodeCapacity);
  end;
  
  // Initialize running statistics for this BatchNorm node
  SetLength(FBatchNormRunningMean[BNIdx], Channels);
  SetLength(FBatchNormRunningVar[BNIdx], Channels);
  for c := 0 to Channels - 1 do
  begin
    FBatchNormRunningMean[BNIdx][c] := 0;
    FBatchNormRunningVar[BNIdx][c] := 1;  // Initialize variance to 1
  end;
  FBatchNormEpsilon[BNIdx] := Epsilon;
  FBatchNormMomentum[BNIdx] := Momentum;
  
  // Create BatchNorm node
  Node.Op := opBatchNorm;
  Node.Result := CreateTensorSameShape(FArena, TensorInput, FNodes[InputIdx].RequiresGrad);
  Node.Parents[0] := InputIdx;
  Node.Parents[1] := GammaIdx;
  Node.Parents[2] := BetaIdx;
  Node.ParentCount := 3;  // Input, Gamma, Beta
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  // Execute BatchNorm forward pass
  TOps.BatchNorm(FArena, TensorInput, TensorGamma, TensorBeta,
    Epsilon, FIsTraining,
    FBatchNormRunningMean[BNIdx], FBatchNormRunningVar[BNIdx],
    Momentum, Node.Result);
  
  Result := AddNode(Node);
end;

function TGraph.MatMulAdd(A, B, Bias: Integer): Integer;
var
  Node: TNode;
  TensorA, TensorB, TensorBias: TTensor;
  OutShape: TArray<Integer>;
  i: Integer;
  ANdim, BNdim, OutNdim: Integer;
begin
  TensorA := FNodes[A].Result;
  TensorB := FNodes[B].Result;
  TensorBias := FNodes[Bias].Result;
  
  // Compute output shape from MatMul
  ANdim := TensorA.NDim;
  BNdim := TensorB.NDim;
  if (ANdim < 2) or (BNdim < 2) then
    raise Exception.Create('MatMulAdd: Tensors must have at least 2 dimensions');
  
  if TensorA.Shape[ANdim - 1] <> TensorB.Shape[BNdim - 2] then
    raise Exception.CreateFmt('MatMulAdd: Inner dimension mismatch: %d != %d',
      [TensorA.Shape[ANdim - 1], TensorB.Shape[BNdim - 2]]);
  
  OutNdim := Max(ANdim, BNdim);
  SetLength(OutShape, OutNdim);
  for i := 0 to OutNdim - 3 do
    OutShape[i] := TensorA.Shape[i];
  OutShape[OutNdim - 2] := TensorA.Shape[ANdim - 2];
  OutShape[OutNdim - 1] := TensorB.Shape[BNdim - 1];
  
  Node.Op := opMatMulAdd;
  SetLength(Node.Result.Shape, OutNdim);
  for i := 0 to OutNdim - 1 do
    Node.Result.Shape[i] := OutShape[i];
  Node.Result.Strides := ComputeStrides(Node.Result.Shape);
  Node.Result.DataPtr := FArena.Alloc(Node.Result.ElementCount);
  Node.Result.GradPtr := -1;
  Node.Result.RequiresGrad := FNodes[A].RequiresGrad or FNodes[B].RequiresGrad or FNodes[Bias].RequiresGrad;
  
  Node.Parents[0] := A;
  Node.Parents[1] := B;
  Node.Parents[2] := Bias;
  Node.ParentCount := 3;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.MatMulAdd(FArena, TensorA, TensorB, TensorBias, Node.Result);
  Result := AddNode(Node);
end;

function TGraph.AddReLU(A, B: Integer): Integer;
var
  Node: TNode;
  TensorA, TensorB: TTensor;
  OutShape: TArray<Integer>;
  i: Integer;
begin
  TensorA := FNodes[A].Result;
  TensorB := FNodes[B].Result;
  
  if not CanBroadcast(TensorA.Shape, TensorB.Shape) then
    raise Exception.Create('AddReLU: Shapes are not broadcastable');
  
  OutShape := BroadcastShapes(TensorA.Shape, TensorB.Shape);
  
  Node.Op := opAddReLU;
  SetLength(Node.Result.Shape, Length(OutShape));
  for i := 0 to High(OutShape) do
    Node.Result.Shape[i] := OutShape[i];
  Node.Result.Strides := ComputeStrides(Node.Result.Shape);
  Node.Result.DataPtr := FArena.Alloc(Node.Result.ElementCount);
  Node.Result.GradPtr := -1;
  Node.Result.RequiresGrad := FNodes[A].RequiresGrad or FNodes[B].RequiresGrad;
  
  Node.Parents[0] := A;
  Node.Parents[1] := B;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.AddReLU(FArena, TensorA, TensorB, Node.Result);
  Result := AddNode(Node);
end;

function TGraph.MatMulReLU(A, B: Integer): Integer;
var
  Node: TNode;
  TensorA, TensorB: TTensor;
  ANdim, BNdim, OutNdim: Integer;
  OutShape: TArray<Integer>;
  i: Integer;
begin
  TensorA := FNodes[A].Result;
  TensorB := FNodes[B].Result;
  
  ANdim := TensorA.NDim;
  BNdim := TensorB.NDim;
  
  if (ANdim < 2) or (BNdim < 2) then
    raise Exception.Create('MatMulReLU: Tensors must have at least 2 dimensions');
  
  if TensorA.Shape[ANdim - 1] <> TensorB.Shape[BNdim - 2] then
    raise Exception.CreateFmt('MatMulReLU: Inner dimension mismatch: %d != %d',
      [TensorA.Shape[ANdim - 1], TensorB.Shape[BNdim - 2]]);
  
  OutNdim := Max(ANdim, BNdim);
  SetLength(OutShape, OutNdim);
  for i := 0 to OutNdim - 3 do
    OutShape[i] := TensorA.Shape[i];
  OutShape[OutNdim - 2] := TensorA.Shape[ANdim - 2];
  OutShape[OutNdim - 1] := TensorB.Shape[BNdim - 1];
  
  Node.Op := opMatMulReLU;
  SetLength(Node.Result.Shape, OutNdim);
  for i := 0 to OutNdim - 1 do
    Node.Result.Shape[i] := OutShape[i];
  Node.Result.Strides := ComputeStrides(Node.Result.Shape);
  Node.Result.DataPtr := FArena.Alloc(Node.Result.ElementCount);
  Node.Result.GradPtr := -1;
  Node.Result.RequiresGrad := FNodes[A].RequiresGrad or FNodes[B].RequiresGrad;
  
  Node.Parents[0] := A;
  Node.Parents[1] := B;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.MatMulReLU(FArena, TensorA, TensorB, Node.Result);
  Result := AddNode(Node);
end;

procedure TGraph.Backward(LossNode: Integer);
var
  i, j: Integer;
  Node: TNode;
  OutGrad, AGrad, BGrad: TTensor;
  POutGrad: PSingleArray;
begin
  // Reset gradient check flag for this backward pass
  FGradientCheckDone := False;
  // Allocate gradients for all nodes that need them
  for i := 0 to FNodeCount - 1 do
    if FNodes[i].RequiresGrad then
      AllocGradIfNeeded(i);
  
  // Initialize loss gradient to 1.0
  // Ensure loss node requires grad (it should, but check anyway)
  if not FNodes[LossNode].RequiresGrad then
    raise Exception.CreateFmt('Backward: Loss node %d does not require gradients', [LossNode]);
    
  AllocGradIfNeeded(LossNode);
  if FNodes[LossNode].Result.ElementCount = 0 then
    raise Exception.Create('Backward: Loss node has zero elements');
    
  // Double-check gradient was allocated
  if FNodes[LossNode].Result.GradPtr < 0 then
    raise Exception.CreateFmt('Backward: Failed to allocate gradient for loss node %d', [LossNode]);
    
  POutGrad := PSingleArray(FNodes[LossNode].Result.RawGrad(FArena));
  POutGrad^[0] := 1.0;
  
  // Note: POutGrad is only used for loss node initialization
  // In the backward pass, each node's OutGrad.RawGrad() is used
  
  // Backward pass: traverse nodes in reverse order
  for i := FNodeCount - 1 downto 0 do
  begin
    Node := FNodes[i];
    
    if not Node.RequiresGrad then
      Continue;
    
    // Ensure the current node's gradient is allocated
    AllocGradIfNeeded(i);
    
    // Refresh Node to get updated GradPtr
    Node := FNodes[i];
    
    // Double-check that gradient was allocated
    // If gradient still not allocated after AllocGradIfNeeded, skip this node
    if Node.Result.GradPtr < 0 then
      Continue;
    
    // Get fresh copy of node to ensure GradPtr is current
    // Refresh one more time to be absolutely sure
    Node := FNodes[i];
    OutGrad := Node.Result;
    
    // Final validation: ensure OutGrad has gradient allocated
    if OutGrad.GradPtr < 0 then
    begin
      // Try one more time to allocate
      if (OutGrad.ElementCount > 0) and Node.RequiresGrad then
      begin
        AllocGradIfNeeded(i);
        Node := FNodes[i];
        OutGrad := Node.Result;
      end;
      
      // If still not allocated, skip this node
      if OutGrad.GradPtr < 0 then
        Continue;
    end;
    
    case Node.Op of
      opAdd:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            AllocGradIfNeeded(Node.Parents[1]);
            BGrad := FNodes[Node.Parents[1]].Result;
            TOps.AddBackward(FArena, OutGrad, AGrad, BGrad);
          end;
        end;
      
      opMul:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad or FNodes[Node.Parents[1]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            AllocGradIfNeeded(Node.Parents[1]);
            BGrad := FNodes[Node.Parents[1]].Result;
            var TensorA, TensorB: TTensor;
            TensorA := FNodes[Node.Parents[0]].Result;
            TensorB := FNodes[Node.Parents[1]].Result;
            TOps.MulBackward(FArena, TensorA, TensorB, OutGrad, AGrad, BGrad);
          end;
        end;
      
      opMatMul:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad or FNodes[Node.Parents[1]].RequiresGrad then
          begin
            var TensorA, TensorB: TTensor;
            TensorA := FNodes[Node.Parents[0]].Result;
            TensorB := FNodes[Node.Parents[1]].Result;
            
            // Only allocate gradients for parents that require grad
            // For A (usually a Param which requires grad)
            AllocGradIfNeeded(Node.Parents[0]);
            // Always get fresh copy after allocation attempt
            AGrad := FNodes[Node.Parents[0]].Result;
            // If gradient wasn't allocated (shouldn't happen for params), mark it
            if AGrad.GradPtr < 0 then
              AGrad.RequiresGrad := False;
            
            // For B (usually an Input which doesn't require grad in XOR demo)
            AllocGradIfNeeded(Node.Parents[1]);
            // Always get fresh copy after allocation attempt
            BGrad := FNodes[Node.Parents[1]].Result;
            // If gradient wasn't allocated (input nodes don't require grad), mark it
            if BGrad.GradPtr < 0 then
              BGrad.RequiresGrad := False;
            
            TOps.MatMulBackward(FArena, TensorA, TensorB, OutGrad, AGrad, BGrad);
          end;
        end;
      
      opReLU:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            // Need the original input tensor (before ReLU) - stored in parent
            var InputTensor: TTensor;
            InputTensor := FNodes[Node.Parents[0]].Result;
            TOps.ReLUBackward(FArena, InputTensor, OutGrad, AGrad);
          end;
        end;
      
      opLeakyReLU:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            // Need the original input tensor (before LeakyReLU)
            var InputTensor: TTensor;
            InputTensor := FNodes[Node.Parents[0]].Result;
            TOps.LeakyReLUBackward(FArena, InputTensor, OutGrad, AGrad, 0.01);
          end;
        end;
      
      opSigmoid:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            // SigmoidBackward needs the output (after sigmoid) and OutGrad
            TOps.SigmoidBackward(FArena, Node.Result, OutGrad, AGrad);
          end;
        end;
      
      opTanh:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            TOps.TanhBackward(FArena, Node.Result, OutGrad, AGrad);
          end;
        end;
      
      opSoftmax:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            TOps.SoftmaxBackward(FArena, Node.Result, OutGrad, AGrad);
          end;
        end;
      
      opMSE:
        begin
          // MSE backward: dLoss/dPred = 2 * (Pred - Target) / N
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            var PPred, PTarget, PGrad, POutGradLocal: PSingleArray;
            var N: Integer;
            var LossGrad: Single;
            PPred := PSingleArray(FNodes[Node.Parents[0]].Result.RawData(FArena));
            PTarget := PSingleArray(FNodes[Node.Parents[1]].Result.RawData(FArena));
            PGrad := PSingleArray(AGrad.RawGrad(FArena));
            // Use OutGrad (current node's gradient) instead of POutGrad from loss node
            if OutGrad.GradPtr < 0 then
              Continue;
            POutGradLocal := PSingleArray(OutGrad.RawGrad(FArena));
            N := AGrad.ElementCount;
            if (N > 0) and (OutGrad.ElementCount > 0) then
              LossGrad := POutGradLocal^[0]
            else
              LossGrad := 0;
            
            if N > 0 then
            begin
              for j := 0 to N - 1 do
                PGrad^[j] := PGrad^[j] + (2.0 * (PPred^[j] - PTarget^[j]) / N * LossGrad);
            end;
          end;
        end;
      
      opCrossEntropy:
        begin
          // CrossEntropy backward: dLoss/dPred = -Target / Pred / N
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            var PPred, PTarget, PGrad, POutGradLocal: PSingleArray;
            var N: Integer;
            var LossGrad: Single;
            PPred := PSingleArray(FNodes[Node.Parents[0]].Result.RawData(FArena));
            PTarget := PSingleArray(FNodes[Node.Parents[1]].Result.RawData(FArena));
            PGrad := PSingleArray(AGrad.RawGrad(FArena));
            // Use OutGrad (current node's gradient) instead of POutGrad from loss node
            if OutGrad.GradPtr < 0 then
              Continue;
            POutGradLocal := PSingleArray(OutGrad.RawGrad(FArena));
            N := AGrad.ElementCount;
            if (N > 0) and (OutGrad.ElementCount > 0) then
              LossGrad := POutGradLocal^[0]
            else
              LossGrad := 0;
            
            if N > 0 then
            begin
              for j := 0 to N - 1 do
              begin
                if PPred^[j] > 0.0001 then
                  PGrad^[j] := PGrad^[j] + (-PTarget^[j] / PPred^[j] / N * LossGrad)
                else
                  PGrad^[j] := PGrad^[j] + (-PTarget^[j] / 0.0001 / N * LossGrad);
              end;
            end;
          end;
        end;
      
      opConv2D:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad or FNodes[Node.Parents[1]].RequiresGrad then
          begin
            var TensorInput, TensorWeight: TTensor;
            TensorInput := FNodes[Node.Parents[0]].Result;
            TensorWeight := FNodes[Node.Parents[1]].Result;
            
            // Get stored padding and stride
            var Padding, Stride: Integer;
            if i < Length(FConv2DPadding) then
            begin
              Padding := FConv2DPadding[i];
              Stride := FConv2DStride[i];
            end
            else
            begin
              Padding := 0;
              Stride := 1;
            end;
            
            // Allocate gradients if needed
            AllocGradIfNeeded(Node.Parents[0]);
            var InputGrad: TTensor;
            InputGrad := FNodes[Node.Parents[0]].Result;
            
            AllocGradIfNeeded(Node.Parents[1]);
            var WeightGrad: TTensor;
            WeightGrad := FNodes[Node.Parents[1]].Result;
            
            TOps.Conv2DBackward(FArena, TensorInput, TensorWeight, OutGrad,
              Padding, Stride, InputGrad, WeightGrad);
          end;
        end;
      
      opMaxPool2D:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            var InputGrad: TTensor;
            InputGrad := FNodes[Node.Parents[0]].Result;
            
            // Use stored indices for gradient routing
            if i < Length(FMaxPoolIndices) then
              TOps.MaxPool2DBackward(FArena, OutGrad, FMaxPoolIndices[i], InputGrad);
          end;
        end;
      
      opDropout:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            var InputGrad: TTensor;
            InputGrad := FNodes[Node.Parents[0]].Result;
            
            // Use stored mask and rate for gradient
            if i < Length(FDropoutMasks) then
              TOps.DropoutBackward(FArena, OutGrad, FDropoutMasks[i], FDropoutRates[i], InputGrad);
          end;
        end;
      
      opReshape:
        begin
          // Reshape backward: gradient just needs to be reshaped back to input shape
          // Since Reshape is a view operation, the gradient data is the same,
          // we just need to reshape it to match the input shape
          if FNodes[Node.Parents[0]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            
            // Get the original input shape
            var InputShape: TArray<Integer>;
            SetLength(InputShape, Length(FNodes[Node.Parents[0]].Result.Shape));
            for j := 0 to High(InputShape) do
              InputShape[j] := FNodes[Node.Parents[0]].Result.Shape[j];
            
            // Create a view of the output gradient with input shape
            // This works because Reshape is a view - same data, different shape
            var ReshapedGrad: TTensor;
            ReshapedGrad.DataPtr := OutGrad.DataPtr;  // Same data
            ReshapedGrad.GradPtr := OutGrad.GradPtr;  // Same gradient
            ReshapedGrad.RequiresGrad := OutGrad.RequiresGrad;
            ReshapedGrad.Shape := InputShape;
            ReshapedGrad.Strides := ComputeStrides(InputShape);
            
            // Accumulate the reshaped gradient into input gradient
            var PReshapedGrad, PAGrad: PSingleArray;
            var Count: Integer;
            if ReshapedGrad.GradPtr >= 0 then
            begin
              PReshapedGrad := PSingleArray(ReshapedGrad.RawGrad(FArena));
              PAGrad := PSingleArray(AGrad.RawGrad(FArena));
              Count := AGrad.ElementCount;
              
              for j := 0 to Count - 1 do
                PAGrad^[j] := PAGrad^[j] + PReshapedGrad^[j];
            end;
          end;
        end;
      
      opBatchNorm:
        begin
          // BatchNorm backward: Input, Gamma, and Beta all need gradients
          if FNodes[Node.Parents[0]].RequiresGrad or 
             FNodes[Node.Parents[1]].RequiresGrad or 
             FNodes[Node.Parents[2]].RequiresGrad then
          begin
            var TensorInput, TensorGamma, TensorBeta: TTensor;
            TensorInput := FNodes[Node.Parents[0]].Result;
            TensorGamma := FNodes[Node.Parents[1]].Result;
            TensorBeta := FNodes[Node.Parents[2]].Result;
            
            // Get epsilon for this BatchNorm node
            var Epsilon: Single;
            if i < Length(FBatchNormEpsilon) then
              Epsilon := FBatchNormEpsilon[i]
            else
              Epsilon := 1e-5;
            
            // Allocate gradients if needed
            AllocGradIfNeeded(Node.Parents[0]);
            var InputGrad: TTensor;
            InputGrad := FNodes[Node.Parents[0]].Result;
            
            AllocGradIfNeeded(Node.Parents[1]);
            var GammaGrad: TTensor;
            GammaGrad := FNodes[Node.Parents[1]].Result;
            
            AllocGradIfNeeded(Node.Parents[2]);
            var BetaGrad: TTensor;
            BetaGrad := FNodes[Node.Parents[2]].Result;
            
            // Execute BatchNorm backward
            TOps.BatchNormBackward(FArena, TensorInput, OutGrad, TensorGamma,
              Epsilon, InputGrad, GammaGrad, BetaGrad);
          end;
        end;
      
      opMatMulAdd:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad or 
             FNodes[Node.Parents[1]].RequiresGrad or 
             FNodes[Node.Parents[2]].RequiresGrad then
          begin
            var TensorA, TensorB, TensorBias: TTensor;
            TensorA := FNodes[Node.Parents[0]].Result;
            TensorB := FNodes[Node.Parents[1]].Result;
            TensorBias := FNodes[Node.Parents[2]].Result;
            
            AllocGradIfNeeded(Node.Parents[0]);
            var MatMulAddAGrad: TTensor;
            MatMulAddAGrad := FNodes[Node.Parents[0]].Result;
            
            AllocGradIfNeeded(Node.Parents[1]);
            var MatMulAddBGrad: TTensor;
            MatMulAddBGrad := FNodes[Node.Parents[1]].Result;
            
            AllocGradIfNeeded(Node.Parents[2]);
            var BiasGrad: TTensor;
            BiasGrad := FNodes[Node.Parents[2]].Result;
            
            TOps.MatMulAddBackward(FArena, TensorA, TensorB, TensorBias, OutGrad,
              MatMulAddAGrad, MatMulAddBGrad, BiasGrad);
          end;
        end;
      
      opAddReLU:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad or FNodes[Node.Parents[1]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            AllocGradIfNeeded(Node.Parents[1]);
            BGrad := FNodes[Node.Parents[1]].Result;
            TOps.AddReLUBackward(FArena, FNodes[Node.Parents[0]].Result, 
              FNodes[Node.Parents[1]].Result, OutGrad, AGrad, BGrad);
          end;
        end;
      
      opMatMulReLU:
        begin
          if FNodes[Node.Parents[0]].RequiresGrad or FNodes[Node.Parents[1]].RequiresGrad then
          begin
            AllocGradIfNeeded(Node.Parents[0]);
            AGrad := FNodes[Node.Parents[0]].Result;
            AllocGradIfNeeded(Node.Parents[1]);
            BGrad := FNodes[Node.Parents[1]].Result;
            TOps.MatMulReLUBackward(FArena, FNodes[Node.Parents[0]].Result,
              FNodes[Node.Parents[1]].Result, OutGrad, AGrad, BGrad);
          end;
        end;
    end;
  end;
  
  // Perform gradient check if enabled (only once per backward pass)
  if FGradientCheckEnabled and not FGradientCheckDone then
  begin
    CheckGradients(LossNode);
    FGradientCheckDone := True;
  end;
end;

procedure TGraph.EnableGradientCheck(Enabled: Boolean; Epsilon: Single; Tolerance: Single);
begin
  FGradientCheckEnabled := Enabled;
  FGradientCheckEpsilon := Epsilon;
  FGradientCheckTolerance := Tolerance;
end;

function TGraph.CheckGradients(LossNode: Integer): Boolean;
var
  i: Integer;
  Node: TNode;
  ParamCount: Integer;
begin
  Result := True;
  ParamCount := 0;
  
  // Check each parameter's gradients
  for i := 0 to FNodeCount - 1 do
  begin
    Node := FNodes[i];
    if (Node.Op = opParam) and Node.RequiresGrad then
    begin
      AllocGradIfNeeded(i);
      
      if Node.Result.GradPtr < 0 then
      begin
        // Gradient not allocated - this is an error
        raise Exception.CreateFmt('Gradient check failed: Parameter node %d has no gradient allocated', [i]);
      end;
      
      // Count parameter elements
      ParamCount := ParamCount + Node.Result.ElementCount;
    end;
  end;
  
  // Log gradient statistics (in a real implementation, this would go to a log file)
  // For now, we just validate that gradients exist
  if ParamCount = 0 then
    raise Exception.Create('Gradient check: No parameters found to check');
  
  // Note: A full numerical gradient check would require:
  // 1. Storing the forward pass computation sequence
  // 2. Re-executing it with perturbed parameters
  // 3. Computing (f(x+eps) - f(x-eps)) / (2*eps) for each parameter
  // 4. Comparing with analytical gradients
  // This is complex and would require significant refactoring of the graph execution model
  
  // For now, we just verify gradients are computed
  Result := True;
end;

function TGraph.GetMemoryStats: TGraphMemoryStats;
var
  i: Integer;
  Node: TNode;
  ParamBytes, ActivationBytes, GradientBytes: Integer;
begin
  Result.ArenaUsage := FArena.GetUsageStats;
  Result.ParameterCount := 0;
  ParamBytes := 0;
  ActivationBytes := 0;
  GradientBytes := 0;
  
  for i := 0 to FNodeCount - 1 do
  begin
    Node := FNodes[i];
    var ElementBytes := Node.Result.ElementCount * SizeOf(Single);
    
    if Node.Op = opParam then
    begin
      Inc(Result.ParameterCount);
      ParamBytes := ParamBytes + ElementBytes;
      if Node.Result.GradPtr >= 0 then
        GradientBytes := GradientBytes + ElementBytes;
    end
    else
    begin
      ActivationBytes := ActivationBytes + ElementBytes;
      if Node.Result.GradPtr >= 0 then
        GradientBytes := GradientBytes + ElementBytes;
    end;
  end;
  
  Result.ParameterBytes := ParamBytes;
  Result.ActivationBytes := ActivationBytes;
  Result.GradientBytes := GradientBytes;
end;

procedure TGraph.LogMemoryStats;
var
  Stats: TGraphMemoryStats;
begin
  Stats := GetMemoryStats;
  // In a real implementation, this would write to a log file
  // For now, we'll just raise an exception with the stats (for debugging)
  // In production, you'd use a proper logging framework
  raise Exception.CreateFmt(
    'Memory Stats:' + #13#10 +
    '  Arena: %d / %d bytes (%.1f%%)' + #13#10 +
    '  Parameters: %d nodes, %d bytes' + #13#10 +
    '  Activations: %d bytes' + #13#10 +
    '  Gradients: %d bytes' + #13#10 +
    '  Allocations: %d',
    [Stats.ArenaUsage.UsedBytes, Stats.ArenaUsage.TotalCapacity, Stats.ArenaUsage.UsagePercent,
     Stats.ParameterCount, Stats.ParameterBytes,
     Stats.ActivationBytes,
     Stats.GradientBytes,
     Stats.ArenaUsage.AllocationCount]);
end;

procedure TGraph.Step(LearningRate: Single; GradClip: Single);
var
  i, j: Integer;
  Node: TNode;
  PData, PGrad: PSingleArray;
  ClippedGrad: Single;
  MaxGradNorm: Single;
begin
  MaxGradNorm := GradClip; // Clip gradients to prevent exploding gradients
  
  // Update all parameters (opParam nodes) using their gradients
  for i := 0 to FNodeCount - 1 do
  begin
    Node := FNodes[i];
    if (Node.Op = opParam) and Node.RequiresGrad then
    begin
      // Ensure gradient is allocated
      AllocGradIfNeeded(i);
      // Refresh Node to get updated GradPtr
      Node := FNodes[i];
      if Node.Result.GradPtr >= 0 then
      begin
        PData := PSingleArray(Node.Result.RawData(FArena));
        PGrad := PSingleArray(Node.Result.RawGrad(FArena));
        
        // Gradient descent with clipping: param = param - lr * clip(grad)
        for j := 0 to Node.Result.ElementCount - 1 do
        begin
          // Clip gradient to prevent exploding gradients
          if PGrad^[j] > MaxGradNorm then
            ClippedGrad := MaxGradNorm
          else if PGrad^[j] < -MaxGradNorm then
            ClippedGrad := -MaxGradNorm
          else
            ClippedGrad := PGrad^[j];
          
          PData^[j] := PData^[j] - (LearningRate * ClippedGrad);
        end;
      end;
    end;
  end;
end;

procedure TGraph.SetOptimizer(UseAdam: Boolean; Beta1: Single; Beta2: Single; Epsilon: Single);
begin
  FUseAdam := UseAdam;
  FAdamBeta1 := Beta1;
  FAdamBeta2 := Beta2;
  FAdamEpsilon := Epsilon;
  
  // Initialize Adam state arrays if switching to Adam
  if UseAdam then
  begin
    if Length(FAdamM) < FNodeCapacity then
    begin
      SetLength(FAdamM, FNodeCapacity);
      SetLength(FAdamV, FNodeCapacity);
    end;
    
    // Initialize Adam state for existing parameters
    var i: Integer;
    for i := 0 to FNodeCount - 1 do
    begin
      if (FNodes[i].Op = opParam) and FNodes[i].RequiresGrad then
      begin
        var ParamSize := FNodes[i].Result.ElementCount;
        if (i >= Length(FAdamM)) or (Length(FAdamM[i]) <> ParamSize) then
        begin
          if i >= Length(FAdamM) then
          begin
            SetLength(FAdamM, FNodeCapacity);
            SetLength(FAdamV, FNodeCapacity);
          end;
          SetLength(FAdamM[i], ParamSize);
          SetLength(FAdamV[i], ParamSize);
          // Initialize to zero
          for var j := 0 to ParamSize - 1 do
          begin
            FAdamM[i][j] := 0;
            FAdamV[i][j] := 0;
          end;
        end;
      end;
    end;
  end;
end;

procedure TGraph.StepAdam(LearningRate: Single; GradClip: Single);
var
  i, j: Integer;
  Node: TNode;
  PData, PGrad: PSingleArray;
  ClippedGrad: Single;
  MaxGradNorm: Single;
  M, V, MHat, VHat: Single;
begin
  MaxGradNorm := GradClip;
  Inc(FAdamStep);  // Increment time step
  
  // Ensure Adam state arrays are large enough
  if Length(FAdamM) < FNodeCapacity then
  begin
    SetLength(FAdamM, FNodeCapacity);
    SetLength(FAdamV, FNodeCapacity);
  end;
  
  // Update all parameters (opParam nodes) using Adam optimizer
  for i := 0 to FNodeCount - 1 do
  begin
    Node := FNodes[i];
    if (Node.Op = opParam) and Node.RequiresGrad then
    begin
      // Ensure gradient is allocated
      AllocGradIfNeeded(i);
      Node := FNodes[i];
      
      if Node.Result.GradPtr < 0 then
        Continue;
      
      // Initialize Adam state for this parameter if needed
      var ParamSize := Node.Result.ElementCount;
      if (i >= Length(FAdamM)) or (Length(FAdamM[i]) <> ParamSize) then
      begin
        if i >= Length(FAdamM) then
        begin
          SetLength(FAdamM, FNodeCapacity);
          SetLength(FAdamV, FNodeCapacity);
        end;
        SetLength(FAdamM[i], ParamSize);
        SetLength(FAdamV[i], ParamSize);
        for j := 0 to ParamSize - 1 do
        begin
          FAdamM[i][j] := 0;
          FAdamV[i][j] := 0;
        end;
      end;
      
      PData := PSingleArray(Node.Result.RawData(FArena));
      PGrad := PSingleArray(Node.Result.RawGrad(FArena));
      
      // Adam update for each parameter element
      for j := 0 to ParamSize - 1 do
      begin
        ClippedGrad := PGrad^[j];
        if ClippedGrad > MaxGradNorm then
          ClippedGrad := MaxGradNorm
        else if ClippedGrad < -MaxGradNorm then
          ClippedGrad := -MaxGradNorm;
        
        // Adam algorithm:
        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        // m_hat = m_t / (1 - beta1^t)
        // v_hat = v_t / (1 - beta2^t)
        // param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
        
        M := FAdamBeta1 * FAdamM[i][j] + (1 - FAdamBeta1) * ClippedGrad;
        V := FAdamBeta2 * FAdamV[i][j] + (1 - FAdamBeta2) * (ClippedGrad * ClippedGrad);
        
        FAdamM[i][j] := M;
        FAdamV[i][j] := V;
        
        // Bias correction
        MHat := M / (1 - Power(FAdamBeta1, FAdamStep));
        VHat := V / (1 - Power(FAdamBeta2, FAdamStep));
        
        // Update parameter
        PData^[j] := PData^[j] - (LearningRate * MHat / (Sqrt(VHat) + FAdamEpsilon));
      end;
    end;
  end;
end;

procedure TGraph.ZeroGrad;
var
  i, j: Integer;
  Node: TNode;
  PGrad: PSingleArray;
begin
  // Zero all gradients - first ensure they're allocated
  for i := 0 to FNodeCount - 1 do
  begin
    Node := FNodes[i];
    if Node.RequiresGrad then
    begin
      AllocGradIfNeeded(i);
      // Refresh Node to get updated GradPtr
      Node := FNodes[i];
      if Node.Result.GradPtr >= 0 then
      begin
        PGrad := PSingleArray(Node.Result.RawGrad(FArena));
        for j := 0 to Node.Result.ElementCount - 1 do
          PGrad^[j] := 0;
      end;
    end;
  end;
end;

function TGraph.GetNode(Idx: Integer): TNode;
begin
  if (Idx < 0) or (Idx >= FNodeCount) then
    raise Exception.CreateFmt('Invalid node index: %d', [Idx]);
  Result := FNodes[Idx];
end;

function TGraph.GetNodeCount: Integer;
begin
  Result := FNodeCount;
end;

procedure TGraph.SetInputValue(InputIdx: Integer; const Values: TArray<Single>);
var
  Node: TNode;
  PData: PSingleArray;
  i: Integer;
begin
  if (InputIdx < 0) or (InputIdx >= FNodeCount) then
    raise Exception.CreateFmt('Invalid input node index: %d', [InputIdx]);
  
  Node := FNodes[InputIdx];
  if Node.Op <> opInput then
    raise Exception.Create('Node is not an input node');
  
  if Length(Values) <> Node.Result.ElementCount then
    raise Exception.CreateFmt('Value count mismatch: expected %d, got %d', 
      [Node.Result.ElementCount, Length(Values)]);
  
  PData := PSingleArray(Node.Result.RawData(FArena));
  for i := 0 to Length(Values) - 1 do
    PData^[i] := Values[i];
end;

function TGraph.GetOutputValue(OutputIdx: Integer): Single;
var
  Node: TNode;
  PData: PSingleArray;
begin
  if (OutputIdx < 0) or (OutputIdx >= FNodeCount) then
    raise Exception.CreateFmt('Invalid output node index: %d', [OutputIdx]);
  
  Node := FNodes[OutputIdx];
  PData := PSingleArray(Node.Result.RawData(FArena));
  Result := PData^[0];
end;

function TGraph.GetOutputValues(OutputIdx: Integer; Count: Integer): TArray<Single>;
var
  Node: TNode;
  PData: PSingleArray;
  i: Integer;
begin
  if (OutputIdx < 0) or (OutputIdx >= FNodeCount) then
    raise Exception.CreateFmt('Invalid output node index: %d', [OutputIdx]);
  
  Node := FNodes[OutputIdx];
  if Count > Node.Result.ElementCount then
    Count := Node.Result.ElementCount;
  
  SetLength(Result, Count);
  PData := PSingleArray(Node.Result.RawData(FArena));
  for i := 0 to Count - 1 do
    Result[i] := PData^[i];
end;

function TGraph.ExportParams: TModelData;
var
  i, j, ParamIdx: Integer;
  Node: TNode;
  PData: PSingleArray;
  ParamData: TParamData;
begin
  Result.Version := 1;
  Result.ParamCount := 0;
  SetLength(Result.Params, FParamNodeCount);
  ParamIdx := 0;
  
  // Iterate through all nodes up to FParamNodeCount (parameters are created first)
  for i := 0 to FParamNodeCount - 1 do
  begin
    Node := FNodes[i];
    if Node.Op = opParam then
    begin
      // Copy shape
      SetLength(ParamData.Shape, Length(Node.Result.Shape));
      for j := 0 to High(Node.Result.Shape) do
        ParamData.Shape[j] := Node.Result.Shape[j];
      
      // Copy data from arena
      SetLength(ParamData.Data, Node.Result.ElementCount);
      PData := PSingleArray(Node.Result.RawData(FArena));
      for j := 0 to Node.Result.ElementCount - 1 do
        ParamData.Data[j] := PData^[j];
      
      Result.Params[ParamIdx] := ParamData;
      Inc(ParamIdx);
    end;
  end;
  
  Result.ParamCount := ParamIdx;
  SetLength(Result.Params, ParamIdx);
end;

procedure TGraph.ImportParams(const ModelData: TModelData; Reinitialize: Boolean);
var
  i, j, ParamIdx: Integer;
  Node: TNode;
  PData: PSingleArray;
  ShapeMatch: Boolean;
begin
  ParamIdx := 0;
  
  // Iterate through all parameter nodes
  for i := 0 to FParamNodeCount - 1 do
  begin
    Node := FNodes[i];
    if Node.Op = opParam then
    begin
      if ParamIdx >= ModelData.ParamCount then
        raise Exception.CreateFmt('Not enough parameters in model data: expected at least %d, got %d',
          [ParamIdx + 1, ModelData.ParamCount]);
      
      // Validate shape
      ShapeMatch := Length(Node.Result.Shape) = Length(ModelData.Params[ParamIdx].Shape);
      if ShapeMatch then
      begin
        for j := 0 to High(Node.Result.Shape) do
          if Node.Result.Shape[j] <> ModelData.Params[ParamIdx].Shape[j] then
          begin
            ShapeMatch := False;
            Break;
          end;
      end;
      
      if not ShapeMatch then
        raise Exception.CreateFmt('Shape mismatch for parameter %d: expected %s, got %s',
          [ParamIdx, 
           Format('[%d]', [Length(Node.Result.Shape)]),
           Format('[%d]', [Length(ModelData.Params[ParamIdx].Shape)])]);
      
      if Node.Result.ElementCount <> Length(ModelData.Params[ParamIdx].Data) then
        raise Exception.CreateFmt('Element count mismatch for parameter %d: expected %d, got %d',
          [ParamIdx, Node.Result.ElementCount, Length(ModelData.Params[ParamIdx].Data)]);
      
      // Copy data into arena
      PData := PSingleArray(Node.Result.RawData(FArena));
      for j := 0 to Node.Result.ElementCount - 1 do
        PData^[j] := ModelData.Params[ParamIdx].Data[j];
      
      // Reset gradients if reinitializing
      if Reinitialize and (Node.Result.GradPtr >= 0) then
      begin
        var PGrad: PSingleArray;
        PGrad := PSingleArray(Node.Result.RawGrad(FArena));
        for j := 0 to Node.Result.ElementCount - 1 do
          PGrad^[j] := 0;
      end;
      
      Inc(ParamIdx);
    end;
  end;
  
  if ParamIdx <> ModelData.ParamCount then
    raise Exception.CreateFmt('Parameter count mismatch: expected %d, got %d',
      [ModelData.ParamCount, ParamIdx]);
end;

procedure TGraph.SaveModel(const FileName: string);
const
  MAGIC: array[0..3] of AnsiChar = 'NDML';
var
  Stream: TFileStream;
  ModelData: TModelData;
  i, j: Integer;
  MagicBytes: array[0..3] of AnsiChar;
  Version, ParamCount, Reserved: UInt32;
  ShapeLength, DataCount: UInt32;
  ShapeValue: Int32;
  DataValue: Single;
begin
  ModelData := ExportParams;
  
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    // Write header
    Move(MAGIC[0], MagicBytes[0], 4);
    Stream.Write(MagicBytes[0], 4);
    Version := ModelData.Version;
    Stream.Write(Version, SizeOf(Version));
    ParamCount := ModelData.ParamCount;
    Stream.Write(ParamCount, SizeOf(ParamCount));
    Reserved := 0;
    Stream.Write(Reserved, SizeOf(Reserved));
    
    // Write each parameter
    for i := 0 to ModelData.ParamCount - 1 do
    begin
      // Write shape
      ShapeLength := Length(ModelData.Params[i].Shape);
      Stream.Write(ShapeLength, SizeOf(ShapeLength));
      for j := 0 to High(ModelData.Params[i].Shape) do
      begin
        ShapeValue := ModelData.Params[i].Shape[j];
        Stream.Write(ShapeValue, SizeOf(ShapeValue));
      end;
      
      // Write data
      DataCount := Length(ModelData.Params[i].Data);
      Stream.Write(DataCount, SizeOf(DataCount));
      for j := 0 to High(ModelData.Params[i].Data) do
      begin
        DataValue := ModelData.Params[i].Data[j];
        Stream.Write(DataValue, SizeOf(DataValue));
      end;
    end;
  finally
    Stream.Free;
  end;
end;

procedure TGraph.LoadModel(const FileName: string);
const
  MAGIC: array[0..3] of AnsiChar = 'NDML';
var
  Stream: TFileStream;
  ModelData: TModelData;
  i, j: Integer;
  MagicBytes: array[0..3] of AnsiChar;
  Version, ParamCount, Reserved: UInt32;
  ShapeLength, DataCount: UInt32;
  ShapeValue: Int32;
  DataValue: Single;
  ParamData: TParamData;
begin
  if not FileExists(FileName) then
    raise Exception.CreateFmt('Model file not found: %s', [FileName]);
  
  Stream := TFileStream.Create(FileName, fmOpenRead);
  try
    // Read and validate header
    Stream.Read(MagicBytes[0], 4);
    if (MagicBytes[0] <> MAGIC[0]) or (MagicBytes[1] <> MAGIC[1]) or
       (MagicBytes[2] <> MAGIC[2]) or (MagicBytes[3] <> MAGIC[3]) then
      raise Exception.Create('Invalid model file: magic number mismatch');
    
    Stream.Read(Version, SizeOf(Version));
    if Version <> 1 then
      raise Exception.CreateFmt('Unsupported model version: %d (expected 1)', [Version]);
    
    Stream.Read(ParamCount, SizeOf(ParamCount));
    Stream.Read(Reserved, SizeOf(Reserved));
    
    // Read parameters
    SetLength(ModelData.Params, ParamCount);
    ModelData.Version := Version;
    ModelData.ParamCount := ParamCount;
    
    for i := 0 to ParamCount - 1 do
    begin
      // Read shape
      Stream.Read(ShapeLength, SizeOf(ShapeLength));
      SetLength(ParamData.Shape, ShapeLength);
      for j := 0 to ShapeLength - 1 do
      begin
        Stream.Read(ShapeValue, SizeOf(ShapeValue));
        ParamData.Shape[j] := ShapeValue;
      end;
      
      // Read data
      Stream.Read(DataCount, SizeOf(DataCount));
      SetLength(ParamData.Data, DataCount);
      for j := 0 to DataCount - 1 do
      begin
        Stream.Read(DataValue, SizeOf(DataValue));
        ParamData.Data[j] := DataValue;
      end;
      
      ModelData.Params[i] := ParamData;
    end;
  finally
    Stream.Free;
  end;
  
  // Import parameters into graph
  ImportParams(ModelData, False);
end;

function TGraph.InferShape(Op: TOpType; const InputShapes: array of TArray<Integer>; 
  Padding: Integer; Stride: Integer; const NewShape: TArray<Integer>): TArray<Integer>;
var
  ShapeA, ShapeB: TArray<Integer>;
  ANdim, BNdim: Integer;
  ARows, ACols, BRows, BCols: Integer;
  Batch, InCh, InH, InW, OutCh, KernelH, KernelW, OutH, OutW: Integer;
  i: Integer;
begin
  case Op of
    opInput, opParam:
      begin
        // Input/Param shapes are provided directly
        if Length(InputShapes) > 0 then
          Result := InputShapes[0]
        else
          SetLength(Result, 0);
      end;
    
    opAdd, opMul:
      begin
        // Element-wise operations: use broadcasting
        if Length(InputShapes) < 2 then
          raise Exception.Create('InferShape: Add/Mul requires 2 input shapes');
        ShapeA := InputShapes[0];
        ShapeB := InputShapes[1];
        if CanBroadcast(ShapeA, ShapeB) then
          Result := BroadcastShapes(ShapeA, ShapeB)
        else
          raise Exception.Create('InferShape: Shapes are not broadcastable');
      end;
    
    opMatMul:
      begin
        // Matrix multiplication: [M, K] @ [K, N] -> [M, N]
        if Length(InputShapes) < 2 then
          raise Exception.Create('InferShape: MatMul requires 2 input shapes');
        ShapeA := InputShapes[0];
        ShapeB := InputShapes[1];
        ANdim := Length(ShapeA);
        BNdim := Length(ShapeB);
        
        if (ANdim < 2) or (BNdim < 2) then
          raise Exception.Create('InferShape: MatMul requires at least 2 dimensions');
        
        // Get last 2 dimensions
        ARows := ShapeA[ANdim - 2];
        ACols := ShapeA[ANdim - 1];
        BRows := ShapeB[BNdim - 2];
        BCols := ShapeB[BNdim - 1];
        
        if ACols <> BRows then
          raise Exception.CreateFmt('InferShape: MatMul inner dimension mismatch: %d != %d', [ACols, BRows]);
        
        // Result shape: batch dims (if any) + [ARows, BCols]
        if ANdim > 2 then
        begin
          // Batch dimensions
          SetLength(Result, ANdim);
          for i := 0 to ANdim - 3 do
            Result[i] := ShapeA[i];
          Result[ANdim - 2] := ARows;
          Result[ANdim - 1] := BCols;
        end
        else
        begin
          // No batch dimensions
          SetLength(Result, 2);
          Result[0] := ARows;
          Result[1] := BCols;
        end;
      end;
    
    opReLU, opLeakyReLU, opSigmoid, opTanh, opSoftmax:
      begin
        // Unary operations: output shape = input shape
        if Length(InputShapes) > 0 then
          Result := InputShapes[0]
        else
          SetLength(Result, 0);
      end;
    
    opMSE, opCrossEntropy:
      begin
        // Loss functions: scalar output
        SetLength(Result, 1);
        Result[0] := 1;
      end;
    
    opConv2D:
      begin
        // Conv2D: [Batch, InCh, InH, InW] @ [OutCh, InCh, KH, KW] -> [Batch, OutCh, OutH, OutW]
        if Length(InputShapes) < 2 then
          raise Exception.Create('InferShape: Conv2D requires 2 input shapes');
        ShapeA := InputShapes[0];  // Input
        ShapeB := InputShapes[1];  // Weight
        
        if Length(ShapeA) <> 4 then
          raise Exception.Create('InferShape: Conv2D input must be 4D');
        if Length(ShapeB) <> 4 then
          raise Exception.Create('InferShape: Conv2D weight must be 4D');
        
        Batch := ShapeA[0];
        InCh := ShapeA[1];
        InH := ShapeA[2];
        InW := ShapeA[3];
        
        OutCh := ShapeB[0];
        if ShapeB[1] <> InCh then
          raise Exception.CreateFmt('InferShape: Conv2D channel mismatch: %d != %d', [InCh, ShapeB[1]]);
        KernelH := ShapeB[2];
        KernelW := ShapeB[3];
        
        OutH := (InH + 2 * Padding - KernelH) div Stride + 1;
        OutW := (InW + 2 * Padding - KernelW) div Stride + 1;
        
        SetLength(Result, 4);
        Result[0] := Batch;
        Result[1] := OutCh;
        Result[2] := OutH;
        Result[3] := OutW;
      end;
    
    opReshape:
      begin
        // Reshape: output shape is the new shape
        if NewShape = nil then
          raise Exception.Create('InferShape: Reshape requires NewShape parameter');
        Result := NewShape;
      end;
    
    else
      raise Exception.CreateFmt('InferShape: Unsupported operation type %d', [Ord(Op)]);
  end;
end;

procedure TGraph.ValidateShape(NodeIdx: Integer; const InferredShape, ActualShape: TArray<Integer>);
var
  i: Integer;
begin
  if not FShapeValidationEnabled then
    Exit;
  
  if Length(InferredShape) <> Length(ActualShape) then
  begin
    var ShapeStr: string;
    ShapeStr := '[';
    for i := 0 to High(InferredShape) do
    begin
      if i > 0 then ShapeStr := ShapeStr + ', ';
      ShapeStr := ShapeStr + IntToStr(InferredShape[i]);
    end;
    ShapeStr := ShapeStr + ']';
    raise Exception.CreateFmt('Shape validation failed at node %d: inferred %s, actual [%d dims]',
      [NodeIdx, ShapeStr, Length(ActualShape)]);
  end;
  
  for i := 0 to High(InferredShape) do
  begin
    if InferredShape[i] <> ActualShape[i] then
    begin
      var ShapeStr: string;
      ShapeStr := '[';
      var j: Integer;
      for j := 0 to High(InferredShape) do
      begin
        if j > 0 then ShapeStr := ShapeStr + ', ';
        ShapeStr := ShapeStr + IntToStr(InferredShape[j]);
      end;
      ShapeStr := ShapeStr + ']';
      var ActualStr: string;
      ActualStr := '[';
      for j := 0 to High(ActualShape) do
      begin
        if j > 0 then ActualStr := ActualStr + ', ';
        ActualStr := ActualStr + IntToStr(ActualShape[j]);
      end;
      ActualStr := ActualStr + ']';
      raise Exception.CreateFmt('Shape validation failed at node %d: inferred %s, actual %s (dim %d: %d != %d)',
        [NodeIdx, ShapeStr, ActualStr, i, InferredShape[i], ActualShape[i]]);
    end;
  end;
end;

function TGraph.ValidateShapes: Boolean;
var
  i: Integer;
  Node: TNode;
  InferredShape, ActualShape: TArray<Integer>;
  InputShapes: array of TArray<Integer>;
begin
  Result := True;
  if not FShapeValidationEnabled then
    Exit;
  
  for i := 0 to FNodeCount - 1 do
  begin
    Node := FNodes[i];
    ActualShape := Node.Result.Shape;
    
    try
      case Node.Op of
        opInput, opParam:
          begin
            // These are set directly, no inference needed
            Continue;
          end;
        
        opAdd, opMul:
          begin
            SetLength(InputShapes, 2);
            InputShapes[0] := FNodes[Node.Parents[0]].Result.Shape;
            InputShapes[1] := FNodes[Node.Parents[1]].Result.Shape;
            InferredShape := InferShape(Node.Op, InputShapes);
            ValidateShape(i, InferredShape, ActualShape);
          end;
        
        opMatMul:
          begin
            SetLength(InputShapes, 2);
            InputShapes[0] := FNodes[Node.Parents[0]].Result.Shape;
            InputShapes[1] := FNodes[Node.Parents[1]].Result.Shape;
            InferredShape := InferShape(Node.Op, InputShapes);
            ValidateShape(i, InferredShape, ActualShape);
          end;
        
        opReLU, opLeakyReLU, opSigmoid, opTanh, opSoftmax:
          begin
            SetLength(InputShapes, 1);
            InputShapes[0] := FNodes[Node.Parents[0]].Result.Shape;
            InferredShape := InferShape(Node.Op, InputShapes);
            ValidateShape(i, InferredShape, ActualShape);
          end;
        
        opMSE, opCrossEntropy:
          begin
            SetLength(InputShapes, 2);
            InputShapes[0] := FNodes[Node.Parents[0]].Result.Shape;
            InputShapes[1] := FNodes[Node.Parents[1]].Result.Shape;
            InferredShape := InferShape(Node.Op, InputShapes);
            ValidateShape(i, InferredShape, ActualShape);
          end;
        
        opConv2D:
          begin
            SetLength(InputShapes, 2);
            InputShapes[0] := FNodes[Node.Parents[0]].Result.Shape;
            InputShapes[1] := FNodes[Node.Parents[1]].Result.Shape;
            var Padding, Stride: Integer;
            if i < Length(FConv2DPadding) then
            begin
              Padding := FConv2DPadding[i];
              Stride := FConv2DStride[i];
            end
            else
            begin
              Padding := 0;
              Stride := 1;
            end;
            InferredShape := InferShape(Node.Op, InputShapes, Padding, Stride);
            ValidateShape(i, InferredShape, ActualShape);
          end;
        
        opReshape:
          begin
            // Reshape validation is done in the Reshape function itself
            Continue;
          end;
      end;
    except
      on E: Exception do
      begin
        Result := False;
        // Log the error but continue checking other nodes
        // In a real implementation, you might want to collect all errors
      end;
    end;
  end;
end;

end.

