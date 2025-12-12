program XOR_Demo;

{$R-} // Disable range checking for ML operations

uses
  Winapi.Windows,
  Vcl.Forms,
  Vcl.Graphics,
  Vcl.Controls,
  Vcl.ExtCtrls,
  Vcl.StdCtrls,
  Vcl.Dialogs,
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
    // Build forward pass with actual data (resets activations, keeps params)
    procedure BuildForward(const Inputs: TArray<Single>; const Target: Single; IsTraining: Boolean);
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
  // BuildForward is called on first Forward() or TrainStep() with actual data
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

procedure TNetwork.BuildForward(const Inputs: TArray<Single>; const Target: Single; IsTraining: Boolean);
begin
  // Reset only activations, keep parameters intact
  FGraph.ResetActivations;
  
  // CRITICAL: Create input node and SET DATA IMMEDIATELY before any math ops
  FInputIdx := FGraph.Input([2, 1]);
  FGraph.SetInputValue(FInputIdx, Inputs);  // Data must be set BEFORE operations use it!
  
  // Layer 1: Input(2) -> Hidden(H)
  var H1PreIdx: Integer;
  H1PreIdx := FGraph.MatMul(FW1Idx, FInputIdx);  // W1 * x (now has actual data)
  var H1BiasIdx: Integer;
  H1BiasIdx := FGraph.Add(H1PreIdx, FB1Idx);     // W1 * x + b1
  FH1Idx := FGraph.LeakyReLU(H1BiasIdx, 0.01);   // LeakyReLU(W1 * x + b1)
  
  // Layer 2: Hidden(H) -> Output(1)
  var OutPreIdx: Integer;
  OutPreIdx := FGraph.MatMul(FW2Idx, FH1Idx);    // W2 * h1
  var OutBiasIdx: Integer;
  OutBiasIdx := FGraph.Add(OutPreIdx, FB2Idx);   // W2 * h1 + b2
  FOutputIdx := FGraph.Sigmoid(OutBiasIdx);      // Sigmoid(W2 * h1 + b2)
  
  // Target input (for training only)
  if IsTraining then
  begin
    FTargetIdx := FGraph.Input([1, 1]);
    FGraph.SetInputValue(FTargetIdx, [Target]);  // Set target data immediately too
  end;
end;

function TNetwork.Forward(const Inputs: TArray<Single>): Single;
begin
  // Build forward pass with actual input data (inference mode)
  BuildForward(Inputs, 0, False);
  
  // Output is already computed during BuildForward
  Result := FGraph.GetOutputValue(FOutputIdx);
end;

function TNetwork.TrainStep(const Inputs: TArray<Single>; Target: Single; LearningRate: Single; GradClip: Single): Single;
var
  LossValue: Single;
begin
  // Build forward pass with actual input AND target data (training mode)
  BuildForward(Inputs, Target, True);
  
  // Compute loss (output and target are already set with real data)
  FLossIdx := FGraph.MSE(FOutputIdx, FTargetIdx);
  LossValue := FGraph.GetOutputValue(FLossIdx);
  
  // Zero gradients for params (they persist and accumulate)
  FGraph.ZeroGrad;
  
  // Backward pass
  FGraph.Backward(FLossIdx);
  
  // Update parameters
  FGraph.Step(LearningRate, GradClip);
  
  Result := LossValue;
end;

procedure TNetwork.Reset;
begin
  // Full reset: reinitialize parameters with new random values
  InitializeParams;
  // BuildForward is called on next Forward() or TrainStep() with actual data
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
    BtnSave: TButton;
    BtnLoad: TButton;
    // Hyperparameter controls
    LblLR: TLabel;
    EditLR: TEdit;
    LblHidden: TLabel;
    EditHidden: TEdit;
    LblGradClip: TLabel;
    EditGradClip: TEdit;
    PanelControls: TPanel;
    ChartPanel: TPaintBox;
    constructor Create(AOwner: TComponent); override;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure TimerTimer(Sender: TObject);
    procedure PaintBoxPaint(Sender: TObject);
    procedure BtnStartStopClick(Sender: TObject);
    procedure BtnResetClick(Sender: TObject);
    procedure BtnSaveClick(Sender: TObject);
    procedure BtnLoadClick(Sender: TObject);
    procedure ChartPanelPaint(Sender: TObject);
  private
    Network: TNetwork;
    TrainX: TArray<TArray<Single>>;
    TrainY: TArray<Single>;
    NumSamples: Integer;
    EpochCount: Integer;
    Buffer: TBitmap;
    IsTraining: Boolean;
    LearningRate: Single;
    HiddenSize: Integer;
    GradClip: Single;
    RenderCounter: Integer;  // Skip rendering on most timer ticks
    LossHistory: TList<Single>;  // Loss history for chart
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
  ClientHeight := 680;  // Taller to fit loss chart at bottom
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

  // Save Button
  BtnSave := TButton.Create(Self);
  BtnSave.Parent := PanelControls;
  BtnSave.Left := 270;
  BtnSave.Top := 80;
  BtnSave.Width := 100;
  BtnSave.Height := 35;
  BtnSave.Caption := 'Save Model';
  BtnSave.OnClick := BtnSaveClick;

  // Load Button
  BtnLoad := TButton.Create(Self);
  BtnLoad.Parent := PanelControls;
  BtnLoad.Left := 380;
  BtnLoad.Top := 80;
  BtnLoad.Width := 100;
  BtnLoad.Height := 35;
  BtnLoad.Caption := 'Load Model';
  BtnLoad.OnClick := BtnLoadClick;

  Timer := TTimer.Create(Self);
  Timer.Interval := 10;
  Timer.OnTimer := TimerTimer;
  Timer.Enabled := False;  // Start disabled

  Buffer := TBitmap.Create;
  Buffer.SetSize(40, 40);  // Balanced: faster than 50x50, smoother than 25x25
  Buffer.PixelFormat := pf32bit;
  RenderCounter := 0;

  // Loss history chart (at bottom of window)
  LossHistory := TList<Single>.Create;
  ChartPanel := TPaintBox.Create(Self);
  ChartPanel.Parent := Self;
  ChartPanel.Left := 10;
  ChartPanel.Top := 590;
  ChartPanel.Width := 480;
  ChartPanel.Height := 80;
  ChartPanel.OnPaint := ChartPanelPaint;

  PrepareData;
  RenderBrain;  // Initial render
end;

procedure TMainForm.PrepareData;
begin
  Randomize;
  
  // Expanded XOR dataset with interpolated points for better generalization
  // This helps the network learn the classic "X" pattern
  NumSamples := 16;
  SetLength(TrainX, NumSamples);
  SetLength(TrainY, NumSamples);
  
  // Original 4 corners
  TrainX[0] := [0.0, 0.0];  TrainY[0] := 0.0;  // Red corner
  TrainX[1] := [0.0, 1.0];  TrainY[1] := 1.0;  // Blue corner
  TrainX[2] := [1.0, 0.0];  TrainY[2] := 1.0;  // Blue corner
  TrainX[3] := [1.0, 1.0];  TrainY[3] := 0.0;  // Red corner
  
  // Points along the "0" diagonal (top-left to bottom-right) - should be RED (0)
  TrainX[4] := [0.25, 0.25];  TrainY[4] := 0.0;
  TrainX[5] := [0.5, 0.5];    TrainY[5] := 0.0;  // Center
  TrainX[6] := [0.75, 0.75];  TrainY[6] := 0.0;
  
  // Points along the "1" diagonal (top-right to bottom-left) - should be BLUE (1)
  TrainX[7] := [0.75, 0.25];  TrainY[7] := 1.0;
  TrainX[8] := [0.25, 0.75];  TrainY[8] := 1.0;
  
  // Additional points in the quadrants
  TrainX[9] := [0.1, 0.1];    TrainY[9] := 0.0;   // Near (0,0)
  TrainX[10] := [0.9, 0.9];   TrainY[10] := 0.0;  // Near (1,1)
  TrainX[11] := [0.1, 0.9];   TrainY[11] := 1.0;  // Near (0,1)
  TrainX[12] := [0.9, 0.1];   TrainY[12] := 1.0;  // Near (1,0)
  
  // Edge midpoints
  TrainX[13] := [0.0, 0.5];   TrainY[13] := 0.5;  // Left edge - ambiguous
  TrainX[14] := [1.0, 0.5];   TrainY[14] := 0.5;  // Right edge - ambiguous
  TrainX[15] := [0.5, 0.0];   TrainY[15] := 0.5;  // Top edge - ambiguous
  
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
  
  // Restart training if it was running
  if WasTraining then
  begin
    IsTraining := True;
    Timer.Enabled := True;
  end;
  
  UpdateButtonState;
end;

procedure TMainForm.BtnSaveClick(Sender: TObject);
var
  SaveDialog: TSaveDialog;
begin
  if Network = nil then
  begin
    ShowMessage('No network to save');
    Exit;
  end;
  
  SaveDialog := TSaveDialog.Create(Self);
  try
    SaveDialog.Filter := 'NeuralDelphi Model (*.ndml)|*.ndml|All Files (*.*)|*.*';
    SaveDialog.DefaultExt := 'ndml';
    SaveDialog.FileName := 'xor_model.ndml';
    if SaveDialog.Execute then
    begin
      try
        Network.Graph.SaveModel(SaveDialog.FileName);
        ShowMessage('Model saved successfully');
      except
        on E: Exception do
          ShowMessage('Error saving model: ' + E.Message);
      end;
    end;
  finally
    SaveDialog.Free;
  end;
end;

procedure TMainForm.BtnLoadClick(Sender: TObject);
var
  OpenDialog: TOpenDialog;
  PredBefore, PredAfter: Single;
  TestInput: TArray<Single>;
begin
  if Network = nil then
  begin
    ShowMessage('Network not initialized');
    Exit;
  end;
  
  OpenDialog := TOpenDialog.Create(Self);
  try
    OpenDialog.Filter := 'NeuralDelphi Model (*.ndml)|*.ndml|All Files (*.*)|*.*';
    OpenDialog.DefaultExt := 'ndml';
    if OpenDialog.Execute then
    begin
      try
        // Test prediction before loading
        TestInput := [0.0, 1.0];
        PredBefore := Network.Forward(TestInput);
        
        // Load model
        Network.Graph.LoadModel(OpenDialog.FileName);
        
        // Test prediction after loading
        PredAfter := Network.Forward(TestInput);
        
        // Verify predictions match (they should if model was saved correctly)
        ShowMessageFmt('Model loaded successfully'#13#10 +
          'Prediction before load: %.6f'#13#10 +
          'Prediction after load: %.6f'#13#10 +
          'Difference: %.6f',
          [PredBefore, PredAfter, Abs(PredBefore - PredAfter)]);
        
        RenderBrain;
      except
        on E: Exception do
          ShowMessage('Error loading model: ' + E.Message);
      end;
    end;
  finally
    OpenDialog.Free;
  end;
end;

procedure TMainForm.TimerTimer(Sender: TObject);
var
  i: Integer;
begin
  for i := 1 to 10 do
    TrainEpoch;
  
  // Render less frequently for better performance
  Inc(RenderCounter);
  if RenderCounter mod 5 = 0 then  // Render every 50 epochs instead of every 10
  begin
    RenderBrain;
    PaintBox.Invalidate;
  end;
end;

procedure TMainForm.TrainEpoch;
var
  i, j, SwapIdx: Integer;
  TotalLoss, Loss: Single;
  Pred: array[0..3] of Single;
  TempX: TArray<Single>;
  TempY: Single;
  Indices: TArray<Integer>;
begin
  TotalLoss := 0;
  
  // Shuffle training samples to break deterministic order
  // This helps escape local minima and find different solutions
  SetLength(Indices, NumSamples);
  for i := 0 to NumSamples - 1 do
    Indices[i] := i;
  
  // Fisher-Yates shuffle
  for i := NumSamples - 1 downto 1 do
  begin
    SwapIdx := Random(i + 1);
    // Swap indices
    j := Indices[i];
    Indices[i] := Indices[SwapIdx];
    Indices[SwapIdx] := j;
  end;
  
  // Train on shuffled samples
  for i := 0 to NumSamples - 1 do
  begin
    Loss := Network.TrainStep(TrainX[Indices[i]], TrainY[Indices[i]], LearningRate, GradClip);
    TotalLoss := TotalLoss + Loss;
  end;
  Inc(EpochCount);
  
  // Debug: check what network predicts for the 4 corners
  for i := 0 to 3 do
    Pred[i] := Network.Forward(TrainX[i]);
  
  LblEpoch.Caption := 'Epoch: ' + IntToStr(EpochCount);
  // Show predictions in loss label for debugging
  LblLoss.Caption := Format('L:%.4f [%.2f,%.2f,%.2f,%.2f]', 
    [TotalLoss / NumSamples, Pred[0], Pred[1], Pred[2], Pred[3]]);
  
  // Record loss for chart
  LossHistory.Add(TotalLoss / NumSamples);
  if EpochCount mod 10 = 0 then
    ChartPanel.Invalidate;
end;

procedure TMainForm.RenderBrain;
var
  xx, yy: Integer;
  Inputs: TArray<Single>;
  Val: Single;
  Row: PRGBQuad;
  MaxX, MaxY: Integer;
begin
  SetLength(Inputs, 2);
  MaxX := Buffer.Width - 1;
  MaxY := Buffer.Height - 1;
  for yy := 0 to MaxY do
  begin
    Row := Buffer.ScanLine[yy];
    for xx := 0 to MaxX do
    begin
      // Map to [0,1] range so corners hit exactly 0.0 and 1.0
      Inputs[0] := xx / MaxX;
      Inputs[1] := yy / MaxY;
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

procedure TMainForm.ChartPanelPaint(Sender: TObject);
var
  Canvas: TCanvas;
  i: Integer;
  MaxLoss, MinLoss: Single;
  X, Y, PrevX, PrevY: Integer;
  ScaleX, ScaleY: Single;
  W, H: Integer;
begin
  Canvas := ChartPanel.Canvas;
  W := ChartPanel.Width;
  H := ChartPanel.Height;
  
  // Background
  Canvas.Brush.Color := clWhite;
  Canvas.FillRect(Rect(0, 0, W, H));
  Canvas.Pen.Color := clBlack;
  Canvas.Rectangle(0, 0, W, H);
  
  Canvas.Font.Size := 7;
  Canvas.TextOut(3, 2, 'Loss');
  
  if LossHistory.Count < 2 then Exit;
  
  // Find min/max
  MaxLoss := LossHistory[0];
  MinLoss := LossHistory[0];
  for i := 1 to LossHistory.Count - 1 do
  begin
    if LossHistory[i] > MaxLoss then MaxLoss := LossHistory[i];
    if LossHistory[i] < MinLoss then MinLoss := LossHistory[i];
  end;
  if MaxLoss <= MinLoss then MaxLoss := MinLoss + 0.1;
  
  ScaleX := (W - 10) / Max(1, LossHistory.Count - 1);
  ScaleY := (H - 20) / (MaxLoss - MinLoss);
  
  Canvas.Pen.Color := clBlue;
  Canvas.Pen.Width := 1;
  
  PrevX := 5;
  PrevY := H - 10 - Round((LossHistory[0] - MinLoss) * ScaleY);
  
  for i := 1 to LossHistory.Count - 1 do
  begin
    X := 5 + Round(i * ScaleX);
    Y := H - 10 - Round((LossHistory[i] - MinLoss) * ScaleY);
    Canvas.MoveTo(PrevX, PrevY);
    Canvas.LineTo(X, Y);
    PrevX := X;
    PrevY := Y;
  end;
end;

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TMainForm, MainForm);
  Application.Run;
end.

