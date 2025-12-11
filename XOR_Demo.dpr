program XOR_Demo;

{$R-} // Disable range checking for ML operations

uses
  Winapi.Windows,
  Vcl.Forms,
  Vcl.Graphics,
  Vcl.Controls,
  Vcl.ExtCtrls,
  Vcl.StdCtrls,
  System.SysUtils,
  System.Generics.Collections,
  System.Classes,
  System.Types,
  System.Math,
  System.IOUtils,
  ML.Arena,
  ML.Tensor,
  ML.Ops,
  ML.Graph;

type
  PSingleArray = ^TSingleArray;
  TSingleArray = array[0..MaxInt div SizeOf(Single) - 1] of Single;

{ ============================================================================
  PART 1: THE AI ENGINE (High-Performance Arena-Based)
  ============================================================================ }

type
  // Network structure - stores node indices for the graph
  // ARCHITECTURE: Params allocated once, activations reset each forward pass
  TNetwork = class
  private
    FGraph: TGraph;
    // Parameter node indices (persistent)
    FW1Idx: Integer;  // Weights: 8x2
    FB1Idx: Integer;  // Bias: 8x1
    FW2Idx: Integer;  // Weights: 1x8
    FB2Idx: Integer;  // Bias: 1x1
    // Activation node indices (rebuilt each forward pass)
    FInputIdx: Integer;
    FH1Idx: Integer;  // Hidden layer output
    FOutputIdx: Integer;  // Final output
    FTargetIdx: Integer;
    FLossIdx: Integer;
    
    // Initialize parameters once
    procedure InitializeParams;
    // Build forward pass (resets activations, keeps params)
    procedure BuildForward;
  public
    constructor Create;
    destructor Destroy; override;
    
    // Forward pass with input values
    function Forward(const Inputs: TArray<Single>): Single;
    
    // Training step
    function TrainStep(const Inputs: TArray<Single>; Target: Single; LearningRate: Single): Single;
    
    // Reset parameters to random values
    procedure Reset;
    
    property Graph: TGraph read FGraph;
  end;

constructor TNetwork.Create;
begin
  FGraph := TGraph.Create(256); // 256MB arena
  InitializeParams;  // Allocate params ONCE
  BuildForward;      // Build initial forward pass
end;

destructor TNetwork.Destroy;
begin
  FGraph.Free;
  inherited;
end;

procedure TNetwork.InitializeParams;
begin
  // Full reset to start fresh
  FGraph.Reset;
  
  // Allocate parameters (weights and biases) - these persist
  FW1Idx := FGraph.Param(8, 2);  // 8x2 weight matrix
  FB1Idx := FGraph.Param(8, 1);  // 8x1 bias vector
  FW2Idx := FGraph.Param(1, 8);  // 1x8 weight matrix
  FB2Idx := FGraph.Param(1, 1);  // 1x1 bias scalar
  
  // Mark the boundary: everything before this is persistent
  FGraph.MarkParamsEnd;
end;

procedure TNetwork.BuildForward;
begin
  // Reset only activations, keep parameters intact
  // This is the key optimization - no more save/restore overhead
  FGraph.ResetActivations;
  
  // Build computation graph for forward pass
  // Layer 1: Input(2) -> Hidden(8)
  FInputIdx := FGraph.Input(2, 1);
  var H1PreIdx: Integer;
  H1PreIdx := FGraph.MatMul(FW1Idx, FInputIdx);  // W1 * x
  var H1BiasIdx: Integer;
  H1BiasIdx := FGraph.Add(H1PreIdx, FB1Idx);     // W1 * x + b1
  FH1Idx := FGraph.LeakyReLU(H1BiasIdx, 0.01);   // LeakyReLU(W1 * x + b1)
  
  // Layer 2: Hidden(8) -> Output(1)
  var OutPreIdx: Integer;
  OutPreIdx := FGraph.MatMul(FW2Idx, FH1Idx);    // W2 * h1
  var OutBiasIdx: Integer;
  OutBiasIdx := FGraph.Add(OutPreIdx, FB2Idx);   // W2 * h1 + b2
  FOutputIdx := FGraph.Sigmoid(OutBiasIdx);      // Sigmoid(W2 * h1 + b2)
  
  // Target input (for training)
  FTargetIdx := FGraph.Input(1, 1);
end;

function TNetwork.Forward(const Inputs: TArray<Single>): Single;
begin
  // Rebuild activations only (params persist)
  BuildForward;
  
  // Set input values
  FGraph.SetInputValue(FInputIdx, Inputs);
  
  // Forward pass is computed during BuildForward
  Result := FGraph.GetOutputValue(FOutputIdx);
end;

function TNetwork.TrainStep(const Inputs: TArray<Single>; Target: Single; LearningRate: Single): Single;
var
  LossValue: Single;
begin
  // Reset activations only - params stay in place
  BuildForward;
  
  // Set input and target values
  FGraph.SetInputValue(FInputIdx, Inputs);
  FGraph.SetInputValue(FTargetIdx, [Target]);
  
  // Compute loss
  FLossIdx := FGraph.MSE(FOutputIdx, FTargetIdx);
  LossValue := FGraph.GetOutputValue(FLossIdx);
  
  // Zero gradients for params (they persist and accumulate)
  FGraph.ZeroGrad;
  
  // Backward pass
  FGraph.Backward(FLossIdx);
  
  // Update parameters (they persist in the arena)
  FGraph.Step(LearningRate);
  
  Result := LossValue;
end;

procedure TNetwork.Reset;
begin
  // Full reset: reinitialize parameters with new random values
  InitializeParams;
  BuildForward;
end;

{ ============================================================================
  PART 2: THE VCL VISUALIZER
  ============================================================================ }

type
  TMainForm = class(TForm)
    PaintBox: TPaintBox;
    Timer: TTimer;
    LblEpoch: TLabel;
    LblLoss: TLabel;
    BtnSave: TButton;
    BtnLoad: TButton;
    constructor Create(AOwner: TComponent); override;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure TimerTimer(Sender: TObject);
    procedure PaintBoxPaint(Sender: TObject);
    procedure PaintBoxClick(Sender: TObject);
    procedure BtnSaveClick(Sender: TObject);
    procedure BtnLoadClick(Sender: TObject);
  private
    Network: TNetwork;
    X: array[0..3] of TArray<Single>;
    Targets: array[0..3] of Single;
    EpochCount: Integer;
    Buffer: TBitmap;
    procedure PrepareData;
    procedure TrainEpoch;
    procedure RenderBrain;
  end;

var
  MainForm: TMainForm;

constructor TMainForm.Create(AOwner: TComponent);
begin
  inherited CreateNew(AOwner);
  FormCreate(Self);
end;

procedure TMainForm.FormCreate(Sender: TObject);
begin
  ClientWidth := 500;
  ClientHeight := 600;
  Position := poScreenCenter;
  Caption := 'NeuralDelphi - XOR Demo';
  DoubleBuffered := True;

  PaintBox := TPaintBox.Create(Self);
  PaintBox.Parent := Self;
  PaintBox.Align := alTop;
  PaintBox.Height := 500;
  PaintBox.OnPaint := PaintBoxPaint;
  PaintBox.OnClick := PaintBoxClick;

  LblEpoch := TLabel.Create(Self);
  LblEpoch.Parent := Self;
  LblEpoch.Left := 10;
  LblEpoch.Top := 510;
  LblEpoch.Caption := 'Epoch: 0';

  LblLoss := TLabel.Create(Self);
  LblLoss.Parent := Self;
  LblLoss.Left := 200;
  LblLoss.Top := 510;
  LblLoss.Caption := 'Loss: ...';

  // Create Save Button
  BtnSave := TButton.Create(Self);
  BtnSave.Parent := Self;
  BtnSave.Left := 10;
  BtnSave.Top := 550;
  BtnSave.Width := 100;
  BtnSave.Caption := 'Save Brain';
  BtnSave.OnClick := BtnSaveClick;

  // Create Load Button
  BtnLoad := TButton.Create(Self);
  BtnLoad.Parent := Self;
  BtnLoad.Left := 120;
  BtnLoad.Top := 550;
  BtnLoad.Width := 100;
  BtnLoad.Caption := 'Load Brain';
  BtnLoad.OnClick := BtnLoadClick;

  Timer := TTimer.Create(Self);
  Timer.Interval := 10;
  Timer.OnTimer := TimerTimer;

  Buffer := TBitmap.Create;
  Buffer.SetSize(50, 50);
  Buffer.PixelFormat := pf32bit;

  PrepareData;
end;

procedure TMainForm.PrepareData;
begin
  Randomize;
  Network := TNetwork.Create;

  // XOR dataset: 4 samples
  X[0] := [0.0, 0.0];
  Targets[0] := 0.0;
  X[1] := [0.0, 1.0];
  Targets[1] := 1.0;
  X[2] := [1.0, 0.0];
  Targets[2] := 1.0;
  X[3] := [1.0, 1.0];
  Targets[3] := 0.0;
end;

procedure TMainForm.FormDestroy(Sender: TObject);
begin
  Network.Free;
  Buffer.Free;
end;

procedure TMainForm.PaintBoxClick(Sender: TObject);
begin
  Network.Reset;
  EpochCount := 0;
  Caption := 'Brain Reset!';
end;

procedure TMainForm.BtnSaveClick(Sender: TObject);
begin
  // TODO: Implement save functionality
  Caption := 'Save not yet implemented';
end;

procedure TMainForm.BtnLoadClick(Sender: TObject);
begin
  // TODO: Implement load functionality
  Caption := 'Load not yet implemented';
end;

procedure TMainForm.TimerTimer(Sender: TObject);
var
  i: Integer;
begin
  for i := 1 to 10 do
    TrainEpoch;
  RenderBrain;
  PaintBox.Invalidate;
end;

procedure TMainForm.TrainEpoch;
var
  i: Integer;
  TotalLoss, Loss: Single;
begin
  TotalLoss := 0;
  for i := 0 to 3 do
  begin
    Loss := Network.TrainStep(X[i], Targets[i], 0.5);
    TotalLoss := TotalLoss + Loss;
  end;
  Inc(EpochCount);
  LblEpoch.Caption := 'Epoch: ' + IntToStr(EpochCount);
  LblLoss.Caption := 'Loss: ' + FloatToStrF(TotalLoss / 4, ffFixed, 4, 5);
end;

procedure TMainForm.RenderBrain;
var
  xx, yy: Integer;
  Inputs: TArray<Single>;
  Val: Single;
  Row: PRGBQuad;
begin
  SetLength(Inputs, 2);
  for yy := 0 to Buffer.Height - 1 do
  begin
    Row := Buffer.ScanLine[yy];
    for xx := 0 to Buffer.Width - 1 do
    begin
      Inputs[0] := xx / Buffer.Width;
      Inputs[1] := yy / Buffer.Height;
      Val := Network.Forward(Inputs);
      if Val < 0 then
        Val := 0;
      if Val > 1 then
        Val := 1;
      Row.rgbRed := Round((1 - Val) * 255);
      Row.rgbGreen := Round((1 - Abs(Val - 0.5) * 2) * 100);
      Row.rgbBlue := Round(Val * 255);
      Row.rgbReserved := 255;
      Inc(Row);
    end;
  end;
end;

procedure TMainForm.PaintBoxPaint(Sender: TObject);
begin
  PaintBox.Canvas.StretchDraw(Rect(0, 0, PaintBox.Width, PaintBox.Height), Buffer);
  with PaintBox.Canvas do
  begin
    Pen.Color := clWhite;
    Pen.Width := 2;
    Font.Size := 14;
    Font.Style := [fsBold];
    Brush.Style := bsClear;
    Brush.Color := clRed;
    Ellipse(10, 10, 40, 40);
    TextOut(15, 15, '0');
    Brush.Color := clBlue;
    Ellipse(10, 460, 40, 490);
    TextOut(15, 465, '1');
    Brush.Color := clBlue;
    Ellipse(460, 10, 490, 40);
    TextOut(465, 15, '1');
    Brush.Color := clRed;
    Ellipse(460, 460, 490, 490);
    TextOut(465, 465, '0');
  end;
end;

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TMainForm, MainForm);
  Application.Run;
end.

