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
    FHiddenSize: Integer;
    // Parameter node indices (persistent)
    FW1Idx: Integer;  // Weights: HxIN
    FB1Idx: Integer;  // Bias: Hx1
    FW2Idx: Integer;  // Weights: 1xH
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
    constructor Create(HiddenSize: Integer = 16);
    destructor Destroy; override;
    
    // Forward pass with input values
    function Forward(const Inputs: TArray<Single>): Single;
    
    // Training step
    function TrainStep(const Inputs: TArray<Single>; Target: Single; LearningRate: Single; GradClip: Single = 5.0): Single;
    
    // Reset parameters to random values
    procedure Reset;
    
    property Graph: TGraph read FGraph;
    property HiddenSize: Integer read FHiddenSize;
  end;

constructor TNetwork.Create(HiddenSize: Integer);
begin
  FHiddenSize := HiddenSize;
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
  // HiddenSize controls capacity: larger = more expressive but slower
  FW1Idx := FGraph.Param([FHiddenSize, 2]);  // HxIN weight matrix
  FB1Idx := FGraph.Param([FHiddenSize, 1]);  // Hx1 bias vector
  FW2Idx := FGraph.Param([1, FHiddenSize]);  // 1xH weight matrix
  FB2Idx := FGraph.Param([1, 1]);            // 1x1 bias scalar
  
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
  FInputIdx := FGraph.Input([2, 1]);
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
  
  // Target input (for training) - shape must match output [1, 1]
  FTargetIdx := FGraph.Input([1, 1]);
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

function TNetwork.TrainStep(const Inputs: TArray<Single>; Target: Single; LearningRate: Single; GradClip: Single): Single;
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
  FGraph.Step(LearningRate, GradClip);
  
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
    BtnStartStop: TButton;
    BtnReset: TButton;
    // Hyperparameter controls
    LblLR: TLabel;
    EditLR: TEdit;
    LblHidden: TLabel;
    EditHidden: TEdit;
    LblGradClip: TLabel;
    EditGradClip: TEdit;
    PanelControls: TPanel;
    constructor Create(AOwner: TComponent); override;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure TimerTimer(Sender: TObject);
    procedure PaintBoxPaint(Sender: TObject);
    procedure BtnStartStopClick(Sender: TObject);
    procedure BtnResetClick(Sender: TObject);
  private
    Network: TNetwork;
    X: array[0..3] of TArray<Single>;
    Targets: array[0..3] of Single;
    EpochCount: Integer;
    Buffer: TBitmap;
    IsTraining: Boolean;
    LearningRate: Single;
    HiddenSize: Integer;
    GradClip: Single;
    procedure PrepareData;
    procedure TrainEpoch;
    procedure RenderBrain;
    procedure CreateNetwork;
    procedure UpdateButtonState;
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
  ClientHeight := 650;
  Position := poScreenCenter;
  Caption := 'NeuralDelphi - XOR Demo';
  DoubleBuffered := True;
  
  // Default hyperparameters
  LearningRate := 0.5;
  HiddenSize := 16;
  GradClip := 5.0;
  IsTraining := False;

  // Visualization area
  PaintBox := TPaintBox.Create(Self);
  PaintBox.Parent := Self;
  PaintBox.Align := alTop;
  PaintBox.Height := 450;
  PaintBox.OnPaint := PaintBoxPaint;

  // Control panel
  PanelControls := TPanel.Create(Self);
  PanelControls.Parent := Self;
  PanelControls.Align := alClient;
  PanelControls.BevelOuter := bvNone;
  PanelControls.Color := clBtnFace;

  // Status labels
  LblEpoch := TLabel.Create(Self);
  LblEpoch.Parent := PanelControls;
  LblEpoch.Left := 10;
  LblEpoch.Top := 10;
  LblEpoch.Caption := 'Epoch: 0';
  LblEpoch.Font.Size := 10;

  LblLoss := TLabel.Create(Self);
  LblLoss.Parent := PanelControls;
  LblLoss.Left := 150;
  LblLoss.Top := 10;
  LblLoss.Caption := 'Loss: (not training)';
  LblLoss.Font.Size := 10;

  // Learning Rate
  LblLR := TLabel.Create(Self);
  LblLR.Parent := PanelControls;
  LblLR.Left := 10;
  LblLR.Top := 45;
  LblLR.Caption := 'Learning Rate:';
  
  EditLR := TEdit.Create(Self);
  EditLR.Parent := PanelControls;
  EditLR.Left := 100;
  EditLR.Top := 42;
  EditLR.Width := 60;
  EditLR.Text := '0.5';

  // Hidden Size
  LblHidden := TLabel.Create(Self);
  LblHidden.Parent := PanelControls;
  LblHidden.Left := 180;
  LblHidden.Top := 45;
  LblHidden.Caption := 'Hidden Neurons:';
  
  EditHidden := TEdit.Create(Self);
  EditHidden.Parent := PanelControls;
  EditHidden.Left := 285;
  EditHidden.Top := 42;
  EditHidden.Width := 50;
  EditHidden.Text := '16';

  // Gradient Clipping
  LblGradClip := TLabel.Create(Self);
  LblGradClip.Parent := PanelControls;
  LblGradClip.Left := 350;
  LblGradClip.Top := 45;
  LblGradClip.Caption := 'Grad Clip:';
  
  EditGradClip := TEdit.Create(Self);
  EditGradClip.Parent := PanelControls;
  EditGradClip.Left := 420;
  EditGradClip.Top := 42;
  EditGradClip.Width := 50;
  EditGradClip.Text := '5.0';

  // Start/Stop Button
  BtnStartStop := TButton.Create(Self);
  BtnStartStop.Parent := PanelControls;
  BtnStartStop.Left := 10;
  BtnStartStop.Top := 80;
  BtnStartStop.Width := 120;
  BtnStartStop.Height := 35;
  BtnStartStop.Caption := 'Start Training';
  BtnStartStop.Font.Style := [fsBold];
  BtnStartStop.OnClick := BtnStartStopClick;

  // Reset Button
  BtnReset := TButton.Create(Self);
  BtnReset.Parent := PanelControls;
  BtnReset.Left := 140;
  BtnReset.Top := 80;
  BtnReset.Width := 120;
  BtnReset.Height := 35;
  BtnReset.Caption := 'Reset Network';
  BtnReset.OnClick := BtnResetClick;

  Timer := TTimer.Create(Self);
  Timer.Interval := 10;
  Timer.OnTimer := TimerTimer;
  Timer.Enabled := False;  // Start disabled

  Buffer := TBitmap.Create;
  Buffer.SetSize(50, 50);
  Buffer.PixelFormat := pf32bit;

  PrepareData;
  RenderBrain;  // Initial render
end;

procedure TMainForm.PrepareData;
begin
  Randomize;
  
  // XOR dataset: 4 samples
  X[0] := [0.0, 0.0];
  Targets[0] := 0.0;
  X[1] := [0.0, 1.0];
  Targets[1] := 1.0;
  X[2] := [1.0, 0.0];
  Targets[2] := 1.0;
  X[3] := [1.0, 1.0];
  Targets[3] := 0.0;
  
  CreateNetwork;
end;

procedure TMainForm.CreateNetwork;
begin
  if Assigned(Network) then
    Network.Free;
  Network := TNetwork.Create(HiddenSize);
  EpochCount := 0;
end;

procedure TMainForm.UpdateButtonState;
begin
  if IsTraining then
  begin
    BtnStartStop.Caption := 'Stop Training';
    EditLR.Enabled := False;
    EditHidden.Enabled := False;
    EditGradClip.Enabled := False;
  end
  else
  begin
    BtnStartStop.Caption := 'Start Training';
    EditLR.Enabled := True;
    EditHidden.Enabled := True;
    EditGradClip.Enabled := True;
  end;
end;

procedure TMainForm.FormDestroy(Sender: TObject);
begin
  Network.Free;
  Buffer.Free;
end;

procedure TMainForm.BtnStartStopClick(Sender: TObject);
var
  NewHiddenSize: Integer;
begin
  if not IsTraining then
  begin
    // Read hyperparameters from UI
    LearningRate := StrToFloatDef(EditLR.Text, 0.5);
    NewHiddenSize := StrToIntDef(EditHidden.Text, 16);
    GradClip := StrToFloatDef(EditGradClip.Text, 5.0);
    
    // Clamp values to reasonable ranges
    if LearningRate <= 0 then LearningRate := 0.01;
    if LearningRate > 10 then LearningRate := 10;
    if NewHiddenSize < 2 then NewHiddenSize := 2;
    if NewHiddenSize > 256 then NewHiddenSize := 256;
    if GradClip <= 0 then GradClip := 0.1;
    if GradClip > 100 then GradClip := 100;
    
    // Recreate network if hidden size changed
    if NewHiddenSize <> HiddenSize then
    begin
      HiddenSize := NewHiddenSize;
      CreateNetwork;
    end;
    
    IsTraining := True;
    Timer.Enabled := True;
    Caption := 'NeuralDelphi - XOR Demo (Training...)';
  end
  else
  begin
    IsTraining := False;
    Timer.Enabled := False;
    Caption := 'NeuralDelphi - XOR Demo (Paused)';
  end;
  
  UpdateButtonState;
end;

procedure TMainForm.BtnResetClick(Sender: TObject);
var
  WasTraining: Boolean;
begin
  WasTraining := IsTraining;
  
  // Stop training temporarily
  IsTraining := False;
  Timer.Enabled := False;
  
  // Read new hidden size
  HiddenSize := StrToIntDef(EditHidden.Text, 16);
  if HiddenSize < 2 then HiddenSize := 2;
  if HiddenSize > 256 then HiddenSize := 256;
  
  // Recreate network with fresh random weights
  CreateNetwork;
  RenderBrain;
  PaintBox.Invalidate;
  
  LblEpoch.Caption := 'Epoch: 0';
  LblLoss.Caption := 'Loss: (reset)';
  Caption := 'NeuralDelphi - XOR Demo (Reset)';
  
  // Resume if was training
  if WasTraining then
  begin
    IsTraining := True;
    Timer.Enabled := True;
    Caption := 'NeuralDelphi - XOR Demo (Training...)';
  end;
  
  UpdateButtonState;
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
    Loss := Network.TrainStep(X[i], Targets[i], LearningRate, GradClip);
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

