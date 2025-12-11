unit ML.Graph;

{$R-} // Disable range checking for performance - we do manual bounds checking

interface

uses
  System.SysUtils,
  System.Generics.Collections,
  ML.Arena,
  ML.Tensor,
  ML.Ops;

type
  PSingleArray = ^TSingleArray;
  TSingleArray = array[0..MaxInt div SizeOf(Single) - 1] of Single;

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
    opCrossEntropy // Cross Entropy loss
  );

  // A node in the computation graph
  TNode = record
    Op: TOpType;
    Result: TTensor;
    Parents: array[0..1] of Integer;  // Max 2 parents for binary ops
    ParentCount: Integer;
    RequiresGrad: Boolean;
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
    
    function AddNode(const ANode: TNode): Integer;
    procedure EnsureCapacity;
    procedure AllocGradIfNeeded(NodeIdx: Integer);
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
    function Input(Rows, Cols: Integer): Integer;
    function Param(Rows, Cols: Integer): Integer;
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
    
    // Backward pass
    procedure Backward(LossNode: Integer);
    
    // Optimizer
    procedure Step(LearningRate: Single);
    procedure ZeroGrad;
    
    // Access
    property Arena: TArena read FArena;
    function GetNode(Idx: Integer): TNode;
    function GetNodeCount: Integer;
    
    // Helper methods for setting input values and getting outputs
    procedure SetInputValue(InputIdx: Integer; const Values: TArray<Single>);
    function GetOutputValue(OutputIdx: Integer): Single;
    function GetOutputValues(OutputIdx: Integer; Count: Integer): TArray<Single>;
  end;

implementation

constructor TGraph.Create(ArenaSizeMB: Integer);
begin
  FArena := TArena.Create(ArenaSizeMB);
  FNodeCapacity := 1024;
  SetLength(FNodes, FNodeCapacity);
  FNodeCount := 0;
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

function TGraph.Input(Rows, Cols: Integer): Integer;
var
  Node: TNode;
begin
  Node.Op := opInput;
  Node.Result := TTensor.CreateTensor(FArena, Rows, Cols, False);
  Node.ParentCount := 0;
  Node.RequiresGrad := False;
  Result := AddNode(Node);
end;

function TGraph.Param(Rows, Cols: Integer): Integer;
var
  Node: TNode;
  PData, PGrad: PSingleArray;
  i, Count: Integer;
  Scale: Single;
begin
  Node.Op := opParam;
  Node.Result := TTensor.CreateTensor(FArena, Rows, Cols, True);
  Node.ParentCount := 0;
  Node.RequiresGrad := True;
  
  Count := Node.Result.ElementCount;
  
  // Initialize with random values (Xavier initialization)
  PData := PSingleArray(Node.Result.RawData(FArena));
  Scale := Sqrt(2.0 / (Rows + Cols));
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
begin
  TensorA := FNodes[A].Result;
  TensorB := FNodes[B].Result;
  
  if not TensorA.SameShape(TensorB) then
    raise Exception.Create('Add: Shape mismatch');
  
  Node.Op := opAdd;
  Node.Result := TTensor.CreateTensor(FArena, TensorA.Rows, TensorA.Cols, 
    FNodes[A].RequiresGrad or FNodes[B].RequiresGrad);
  Node.Parents[0] := A;
  Node.Parents[1] := B;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.Add(FArena, TensorA, TensorB, Node.Result);
  Result := AddNode(Node);
end;

function TGraph.Mul(A, B: Integer): Integer;
var
  Node: TNode;
  TensorA, TensorB: TTensor;
begin
  TensorA := FNodes[A].Result;
  TensorB := FNodes[B].Result;
  
  if not TensorA.SameShape(TensorB) then
    raise Exception.Create('Mul: Shape mismatch');
  
  Node.Op := opMul;
  Node.Result := TTensor.CreateTensor(FArena, TensorA.Rows, TensorA.Cols, 
    FNodes[A].RequiresGrad or FNodes[B].RequiresGrad);
  Node.Parents[0] := A;
  Node.Parents[1] := B;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.Mul(FArena, TensorA, TensorB, Node.Result);
  Result := AddNode(Node);
end;

function TGraph.MatMul(A, B: Integer): Integer;
var
  Node: TNode;
  TensorA, TensorB: TTensor;
begin
  TensorA := FNodes[A].Result;
  TensorB := FNodes[B].Result;
  
  if TensorA.Cols <> TensorB.Rows then
    raise Exception.CreateFmt('MatMul: Dimension mismatch (%d x %d) * (%d x %d)', 
      [TensorA.Rows, TensorA.Cols, TensorB.Rows, TensorB.Cols]);
  
  Node.Op := opMatMul;
  Node.Result := TTensor.CreateTensor(FArena, TensorA.Rows, TensorB.Cols, 
    FNodes[A].RequiresGrad or FNodes[B].RequiresGrad);
  Node.Parents[0] := A;
  Node.Parents[1] := B;
  Node.ParentCount := 2;
  Node.RequiresGrad := Node.Result.RequiresGrad;
  
  TOps.MatMul(FArena, TensorA, TensorB, Node.Result);
  Result := AddNode(Node);
end;

function TGraph.ReLU(A: Integer): Integer;
var
  Node: TNode;
  TensorA: TTensor;
begin
  TensorA := FNodes[A].Result;
  
  Node.Op := opReLU;
  Node.Result := TTensor.CreateTensor(FArena, TensorA.Rows, TensorA.Cols, 
    FNodes[A].RequiresGrad);
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
  Node.Result := TTensor.CreateTensor(FArena, TensorA.Rows, TensorA.Cols, 
    FNodes[A].RequiresGrad);
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
  Node.Result := TTensor.CreateTensor(FArena, TensorA.Rows, TensorA.Cols, 
    FNodes[A].RequiresGrad);
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
  Node.Result := TTensor.CreateTensor(FArena, TensorA.Rows, TensorA.Cols, 
    FNodes[A].RequiresGrad);
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
  Node.Result := TTensor.CreateTensor(FArena, TensorA.Rows, TensorA.Cols, 
    FNodes[A].RequiresGrad);
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
  Node.Result := TTensor.CreateTensor(FArena, 1, 1, 
    FNodes[Pred].RequiresGrad or FNodes[Target].RequiresGrad);
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
  Node.Result := TTensor.CreateTensor(FArena, 1, 1, 
    FNodes[Pred].RequiresGrad or FNodes[Target].RequiresGrad);
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

procedure TGraph.Backward(LossNode: Integer);
var
  i, j: Integer;
  Node: TNode;
  OutGrad, AGrad, BGrad: TTensor;
  POutGrad: PSingleArray;
begin
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
    end;
  end;
end;

procedure TGraph.Step(LearningRate: Single);
var
  i, j: Integer;
  Node: TNode;
  PData, PGrad: PSingleArray;
begin
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
        
        // Gradient descent: param = param - lr * grad
        for j := 0 to Node.Result.ElementCount - 1 do
          PData^[j] := PData^[j] - (LearningRate * PGrad^[j]);
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

end.

