program MNIST_Demo;

{$R-} // Disable range checking for ML operations

uses
  Winapi.Windows,
  Vcl.Forms,
  Vcl.Graphics,
  Vcl.Controls,
  Vcl.ExtCtrls,
  Vcl.StdCtrls,
  Vcl.Dialogs,
  Vcl.FileCtrl,
  System.SysUtils,
  System.Generics.Collections,
  System.Classes,
  System.Types,
  System.Math,
  System.IOUtils,
  ML.Arena,
  ML.Tensor,
  ML.Ops,
  ML.Graph,
  MNIST_Loader;

type
  PSingleArray = ^TSingleArray;
  TSingleArray = array[0..MaxInt div SizeOf(Single) - 1] of Single;

{ ============================================================================
  PART 1: THE MNIST NETWORK (Conv2D Architecture)
  ============================================================================ }

type
  TMNISTNetwork = class
  private
    FGraph: TGraph;
    // Parameter node indices (persistent)
    FConv1WeightIdx: Integer;  // [16, 1, 3, 3]
    FConv1BiasIdx: Integer;    // [16]
    FConv2WeightIdx: Integer;  // [32, 16, 3, 3]
    FConv2BiasIdx: Integer;    // [32]
    FDenseWeightIdx: Integer;  // [128, 25088]  (32*28*28 = 25088)
    FDenseBiasIdx: Integer;    // [128]
    FOutWeightIdx: Integer;    // [10, 128]
    FOutBiasIdx: Integer;     // [10]
    // Activation node indices (rebuilt each forward pass)
    FInputIdx: Integer;
    FConv1ReluIdx: Integer;
    FConv2ReluIdx: Integer;
    FDenseReluIdx: Integer;
    FOutputIdx: Integer;
    FTargetIdx: Integer;
    FLossIdx: Integer;
    
    procedure InitializeParams;
    procedure BuildForward(const Input: TArray<Single>; const Target: TArray<Single>; IsTraining: Boolean);
  public
    constructor Create;
    destructor Destroy; override;
    
    // Forward pass: returns prediction probabilities [10]
    function Forward(const Input: TArray<Single>): TArray<Single>;
    
    // Training step: returns loss (single sample, updates immediately)
    function TrainStep(const Input: TArray<Single>; const Target: TArray<Single>;
      LearningRate: Single; GradClip: Single = 5.0): Single;
    
    // Batch training: accumulate gradients without updating
    function AccumulateGradient(const Input: TArray<Single>; const Target: TArray<Single>): Single;
    // Apply accumulated gradients (call after BatchSize samples)
    procedure ApplyGradients(LearningRate: Single; BatchSize: Integer; GradClip: Single = 5.0);
    // Zero gradients (call at start of batch)
    procedure ZeroGradients;
    
    // Reset parameters to random values
    procedure Reset;
    
    property Graph: TGraph read FGraph;
  end;

constructor TMNISTNetwork.Create;
begin
  FGraph := TGraph.Create(512); // 512MB arena for larger network
  InitializeParams;
end;

destructor TMNISTNetwork.Destroy;
begin
  FGraph.Free;
  inherited;
end;

procedure TMNISTNetwork.InitializeParams;
begin
  FGraph.Reset;
  
  // Conv2D Layer 1: 1 input channel -> 16 output channels, 3x3 kernel, stride=2 (28->14)
  FConv1WeightIdx := FGraph.Param([16, 1, 3, 3]);
  FConv1BiasIdx := FGraph.Param([1, 16, 1, 1]);
  
  // Conv2D Layer 2: 16 input channels -> 32 output channels, 3x3 kernel, stride=2 (14->7)
  FConv2WeightIdx := FGraph.Param([32, 16, 3, 3]);
  FConv2BiasIdx := FGraph.Param([1, 32, 1, 1]);
  
  // Dense Layer: 32*7*7 = 1568 -> 128
  FDenseWeightIdx := FGraph.Param([128, 1568]);
  FDenseBiasIdx := FGraph.Param([128, 1]);
  
  // Output Layer: 128 -> 10
  FOutWeightIdx := FGraph.Param([10, 128]);
  FOutBiasIdx := FGraph.Param([10, 1]);
  
  FGraph.MarkParamsEnd;
end;

procedure TMNISTNetwork.BuildForward(const Input: TArray<Single>; const Target: TArray<Single>; IsTraining: Boolean);
var
  Conv1OutIdx, Conv1BiasAddIdx: Integer;
  Conv2OutIdx, Conv2BiasAddIdx: Integer;
  FlattenIdx, DenseOutIdx, DenseBiasAddIdx: Integer;
  OutPreIdx, OutBiasAddIdx: Integer;
begin
  FGraph.ResetActivations;
  
  // Input: [1, 28, 28] (batch=1, channels=1, height=28, width=28)
  FInputIdx := FGraph.Input([1, 1, 28, 28]);
  FGraph.SetInputValue(FInputIdx, Input);
  
  // Conv2D Layer 1: padding=1, stride=2 (28x28 -> 14x14)
  Conv1OutIdx := FGraph.Conv2D(FInputIdx, FConv1WeightIdx, 1, 2);
  Conv1BiasAddIdx := FGraph.Add(Conv1OutIdx, FConv1BiasIdx);
  FConv1ReluIdx := FGraph.ReLU(Conv1BiasAddIdx);
  
  // Conv2D Layer 2: padding=1, stride=2 (14x14 -> 7x7)
  Conv2OutIdx := FGraph.Conv2D(FConv1ReluIdx, FConv2WeightIdx, 1, 2);
  Conv2BiasAddIdx := FGraph.Add(Conv2OutIdx, FConv2BiasIdx);
  FConv2ReluIdx := FGraph.ReLU(Conv2BiasAddIdx);
  
  // Flatten: [1, 32, 7, 7] -> [1568, 1]
  FlattenIdx := FGraph.Reshape(FConv2ReluIdx, [1568, 1]);
  
  // Dense Layer
  DenseOutIdx := FGraph.MatMul(FDenseWeightIdx, FlattenIdx);
  DenseBiasAddIdx := FGraph.Add(DenseOutIdx, FDenseBiasIdx);
  FDenseReluIdx := FGraph.ReLU(DenseBiasAddIdx);
  
  // Output Layer
  OutPreIdx := FGraph.MatMul(FOutWeightIdx, FDenseReluIdx);
  OutBiasAddIdx := FGraph.Add(OutPreIdx, FOutBiasIdx);
  FOutputIdx := FGraph.Softmax(OutBiasAddIdx);
  
  // Target (for training only)
  if IsTraining then
  begin
    FTargetIdx := FGraph.Input([10, 1]);  // Match Softmax output shape [10, 1]
    FGraph.SetInputValue(FTargetIdx, Target);
    FLossIdx := FGraph.CrossEntropy(FOutputIdx, FTargetIdx);
  end;
end;

function TMNISTNetwork.Forward(const Input: TArray<Single>): TArray<Single>;
begin
  BuildForward(Input, [], False);
  Result := FGraph.GetOutputValues(FOutputIdx, 10);
end;

function TMNISTNetwork.TrainStep(const Input: TArray<Single>; const Target: TArray<Single>;
  LearningRate: Single; GradClip: Single): Single;
begin
  BuildForward(Input, Target, True);
  
  FGraph.ZeroGrad;
  FGraph.Backward(FLossIdx);
  FGraph.Step(LearningRate, GradClip);
  
  Result := FGraph.GetOutputValue(FLossIdx);
end;

procedure TMNISTNetwork.ZeroGradients;
begin
  FGraph.ZeroGrad;
end;

function TMNISTNetwork.AccumulateGradient(const Input: TArray<Single>; const Target: TArray<Single>): Single;
begin
  // Forward + backward only, no weight update
  BuildForward(Input, Target, True);
  FGraph.Backward(FLossIdx);
  Result := FGraph.GetOutputValue(FLossIdx);
end;

procedure TMNISTNetwork.ApplyGradients(LearningRate: Single; BatchSize: Integer; GradClip: Single);
begin
  // Scale learning rate by 1/BatchSize to get average gradient effect
  // (gradients accumulate, so we divide LR to average them)
  FGraph.Step(LearningRate / BatchSize, GradClip);
end;

procedure TMNISTNetwork.Reset;
begin
  InitializeParams;
end;

{ ============================================================================
  PART 2: THE UI AND TRAINING LOOP
  ============================================================================ }

type
  TMainForm = class(TForm)
    PaintBox: TPaintBox;
    LblEpoch: TLabel;
    LblLoss: TLabel;
    LblTrainAcc: TLabel;
    LblTestAcc: TLabel;
    LblPrediction: TLabel;
    LblStatus: TStaticText;
    BtnStartStop: TButton;
    BtnReset: TButton;
    BtnNext: TButton;
    BtnPrev: TButton;
    BtnSave: TButton;
    BtnLoad: TButton;
    EditLR: TEdit;
    LblLR: TLabel;
    Timer: TTimer;
    // Chart and confusion matrix panels
    ChartPanel: TPaintBox;
    ConfusionPanel: TPaintBox;
    constructor Create(AOwner: TComponent); override;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure TimerTimer(Sender: TObject);
    procedure PaintBoxPaint(Sender: TObject);
    procedure BtnStartStopClick(Sender: TObject);
    procedure BtnResetClick(Sender: TObject);
    procedure BtnNextClick(Sender: TObject);
    procedure BtnPrevClick(Sender: TObject);
    procedure BtnSaveClick(Sender: TObject);
    procedure BtnLoadClick(Sender: TObject);
    procedure ChartPanelPaint(Sender: TObject);
    procedure ConfusionPanelPaint(Sender: TObject);
  private
    Network: TMNISTNetwork;
    TrainData, TestData: TMNISTData;
    CurrentSampleIdx: Integer;
    EpochCount: Integer;
    IsTraining: Boolean;
    LearningRate: Single;
    TrainSampleIdx: Integer;  // Current training sample index
    TrainShuffledIndices: TArray<Integer>;  // Shuffled indices for current epoch
    // Batch training
    BatchSize: Integer;
    BatchSampleCount: Integer;  // Samples processed in current batch
    BatchLossAccum: Single;     // Accumulated loss for current batch
    // Quick win features
    LossHistory: TList<Single>;  // Training loss history for chart
    ConfusionMatrix: array[0..9, 0..9] of Integer;  // Actual x Predicted
    procedure LoadData;
    procedure TrainEpoch;
    procedure TrainSingleSample;  // Train one sample (called by timer)
    procedure EvaluateTest;
    procedure RenderImage;
    procedure UpdateUI;
    procedure ClearConfusionMatrix;
  end;

var
  MainForm: TMainForm;

constructor TMainForm.Create(AOwner: TComponent);
begin
  inherited CreateNew(AOwner);  // CreateNew doesn't try to load .dfm
  FormCreate(Self);
end;

procedure TMainForm.FormCreate(Sender: TObject);
begin
  ClientWidth := 700;  // Widened for chart and confusion matrix
  ClientHeight := 520;  // Slightly taller for better layout
  Position := poScreenCenter;
  Caption := 'NeuralDelphi - MNIST Demo';
  DoubleBuffered := True;
  
  LearningRate := 0.01;  // Higher LR for faster convergence
  IsTraining := False;
  EpochCount := 0;
  CurrentSampleIdx := 0;
  TrainSampleIdx := 0;
  SetLength(TrainShuffledIndices, 0);
  
  // Batch training settings
  BatchSize := 32;
  BatchSampleCount := 0;
  BatchLossAccum := 0;
  
  // Image display (fixed position, not alTop so it doesn't overlap right panels)
  PaintBox := TPaintBox.Create(Self);
  PaintBox.Parent := Self;
  PaintBox.Left := 10;
  PaintBox.Top := 10;
  PaintBox.Width := 450;
  PaintBox.Height := 300;
  PaintBox.OnPaint := PaintBoxPaint;
  
  // Labels
  LblEpoch := TLabel.Create(Self);
  LblEpoch.Parent := Self;
  LblEpoch.Left := 10;
  LblEpoch.Top := 310;
  LblEpoch.Caption := 'Epoch: 0';
  
  LblLoss := TLabel.Create(Self);
  LblLoss.Parent := Self;
  LblLoss.Left := 150;
  LblLoss.Top := 310;
  LblLoss.Caption := 'Loss: -';
  
  LblTrainAcc := TLabel.Create(Self);
  LblTrainAcc.Parent := Self;
  LblTrainAcc.Left := 10;
  LblTrainAcc.Top := 335;
  LblTrainAcc.Caption := 'Train Acc: -';
  
  LblTestAcc := TLabel.Create(Self);
  LblTestAcc.Parent := Self;
  LblTestAcc.Left := 150;
  LblTestAcc.Top := 335;
  LblTestAcc.Caption := 'Test Acc: -';
  
  LblPrediction := TLabel.Create(Self);
  LblPrediction.Parent := Self;
  LblPrediction.Left := 10;
  LblPrediction.Top := 360;
  LblPrediction.Caption := 'Prediction: -';
  LblPrediction.Font.Size := 12;
  LblPrediction.Font.Style := [fsBold];
  
  // Status label for loading progress (above Learning Rate) - use full width
  // Use TStaticText instead of TLabel for better word wrapping support
  LblStatus := TStaticText.Create(Self);
  LblStatus.Parent := Self;
  LblStatus.Left := 10;
  LblStatus.Top := 385;
  LblStatus.Width := ClientWidth - 20;  // Full width minus 10px margin on each side
  LblStatus.Height := 40;  // Increased height for word wrapping
  LblStatus.Caption := 'Ready';
  LblStatus.Font.Color := clBlue;
  LblStatus.AutoSize := False;
  // TStaticText doesn't have WordWrap property - it wraps automatically when AutoSize=False
  LblStatus.Alignment := taLeftJustify;  // Left-align text
  LblStatus.Anchors := [akLeft, akTop, akRight];  // Anchor to right edge so it resizes with form
  
  // Learning Rate (moved down to avoid overlap with status label that may wrap)
  LblLR := TLabel.Create(Self);
  LblLR.Parent := Self;
  LblLR.Left := 10;
  LblLR.Top := 430;  // Moved down to accommodate wrapped status text
  LblLR.Caption := 'Learning Rate:';
  
  EditLR := TEdit.Create(Self);
  EditLR.Parent := Self;
  EditLR.Left := 100;
  EditLR.Top := 427;  // Aligned with Learning Rate label
  EditLR.Width := 80;
  EditLR.Text := '0.001';
  
  // Buttons (all aligned on same row, moved down for status label)
  BtnStartStop := TButton.Create(Self);
  BtnStartStop.Parent := Self;
  BtnStartStop.Left := 10;
  BtnStartStop.Top := 460;  // Moved down to accommodate status label
  BtnStartStop.Width := 120;
  BtnStartStop.Height := 35;
  BtnStartStop.Caption := 'Start Training';
  BtnStartStop.OnClick := BtnStartStopClick;
  
  BtnReset := TButton.Create(Self);
  BtnReset.Parent := Self;
  BtnReset.Left := 140;
  BtnReset.Top := 460;  // Aligned with Start button
  BtnReset.Width := 120;
  BtnReset.Height := 35;
  BtnReset.Caption := 'Reset Network';
  BtnReset.OnClick := BtnResetClick;
  
  BtnNext := TButton.Create(Self);
  BtnNext.Parent := Self;
  BtnNext.Left := 270;
  BtnNext.Top := 460;  // Aligned with other buttons
  BtnNext.Width := 80;
  BtnNext.Height := 35;
  BtnNext.Caption := 'Next';
  BtnNext.OnClick := BtnNextClick;
  
  BtnPrev := TButton.Create(Self);
  BtnPrev.Parent := Self;
  BtnPrev.Left := 360;
  BtnPrev.Top := 460;  // Aligned with other buttons
  BtnPrev.Width := 80;
  BtnPrev.Height := 35;
  BtnPrev.Caption := 'Prev';
  BtnPrev.OnClick := BtnPrevClick;
  
  // Save/Load buttons (second row)
  BtnSave := TButton.Create(Self);
  BtnSave.Parent := Self;
  BtnSave.Left := 450;
  BtnSave.Top := 460;
  BtnSave.Width := 80;
  BtnSave.Height := 35;
  BtnSave.Caption := 'Save';
  BtnSave.OnClick := BtnSaveClick;
  
  BtnLoad := TButton.Create(Self);
  BtnLoad.Parent := Self;
  BtnLoad.Left := 540;
  BtnLoad.Top := 460;
  BtnLoad.Width := 80;
  BtnLoad.Height := 35;
  BtnLoad.Caption := 'Load';
  BtnLoad.OnClick := BtnLoadClick;
  
  // Loss history chart panel (right side of image)
  ChartPanel := TPaintBox.Create(Self);
  ChartPanel.Parent := Self;
  ChartPanel.Left := 470;
  ChartPanel.Top := 10;
  ChartPanel.Width := 200;
  ChartPanel.Height := 150;
  ChartPanel.OnPaint := ChartPanelPaint;
  
  // Confusion matrix panel (below chart)
  ConfusionPanel := TPaintBox.Create(Self);
  ConfusionPanel.Parent := Self;
  ConfusionPanel.Left := 470;
  ConfusionPanel.Top := 170;
  ConfusionPanel.Width := 200;
  ConfusionPanel.Height := 200;
  ConfusionPanel.OnPaint := ConfusionPanelPaint;
  
  // Initialize loss history
  LossHistory := TList<Single>.Create;
  ClearConfusionMatrix;
  
  Timer := TTimer.Create(Self);
  Timer.Interval := 50;  // Faster updates for better responsiveness
  Timer.Enabled := False;
  Timer.OnTimer := TimerTimer;
  
  Network := TMNISTNetwork.Create;
  
  // Ensure window is fully visible before loading
  Show;
  Update;
  Application.ProcessMessages;
  
  // Try to load data (will prompt user if files not found)
  try
    LoadData;
    RenderImage;
    UpdateUI;
    if Assigned(LblStatus) then
      LblStatus.Caption := 'Data loaded successfully';
  except
    on E: Exception do
    begin
      Caption := 'NeuralDelphi - MNIST Demo';
      ShowMessage('Could not load MNIST data.'#13#10#13#10'Error: ' + E.Message + 
        #13#10#13#10'You can try loading the data files later using the menu or by placing them in the executable directory.');
      // Initialize empty data so app doesn't crash
      TrainData.Count := 0;
      TestData.Count := 0;
      SetLength(TrainData.Images, 0);
      SetLength(TrainData.Labels, 0);
      SetLength(TestData.Images, 0);
      SetLength(TestData.Labels, 0);
      UpdateUI;
    end;
  end;
end;

procedure TMainForm.FormDestroy(Sender: TObject);
begin
  LossHistory.Free;
  Network.Free;
end;

procedure TMainForm.LoadData;
var
  ExeDir, DataDir: string;
  TrainImagesFile, TrainLabelsFile, TestImagesFile, TestLabelsFile: string;
begin
  ExeDir := ExtractFilePath(Application.ExeName);
  
  // Try default locations first
  TrainImagesFile := TPath.Combine(ExeDir, 'train-images-idx3-ubyte');
  TrainLabelsFile := TPath.Combine(ExeDir, 'train-labels-idx1-ubyte');
  TestImagesFile := TPath.Combine(ExeDir, 't10k-images-idx3-ubyte');
  TestLabelsFile := TPath.Combine(ExeDir, 't10k-labels-idx1-ubyte');
  
  // If files don't exist, prompt user to select the folder
  if not (FileExists(TrainImagesFile) and FileExists(TrainLabelsFile) and
          FileExists(TestImagesFile) and FileExists(TestLabelsFile)) then
  begin
    DataDir := ExeDir;
    if not SelectDirectory('Select MNIST Dataset Folder', '', DataDir,
        [sdNewUI, sdShowEdit, sdNewFolder, sdShowShares, sdValidateDir, sdShowFiles]) then
      raise Exception.Create('MNIST data folder selection was cancelled.');
    
    // Look for all 4 files in the selected folder
    TrainImagesFile := TPath.Combine(DataDir, 'train-images-idx3-ubyte');
    TrainLabelsFile := TPath.Combine(DataDir, 'train-labels-idx1-ubyte');
    TestImagesFile := TPath.Combine(DataDir, 't10k-images-idx3-ubyte');
    TestLabelsFile := TPath.Combine(DataDir, 't10k-labels-idx1-ubyte');
    
    // Verify all files exist
    if not FileExists(TrainImagesFile) then
      raise Exception.CreateFmt('train-images-idx3-ubyte not found in: %s', [DataDir]);
    if not FileExists(TrainLabelsFile) then
      raise Exception.CreateFmt('train-labels-idx1-ubyte not found in: %s', [DataDir]);
    if not FileExists(TestImagesFile) then
      raise Exception.CreateFmt('t10k-images-idx3-ubyte not found in: %s', [DataDir]);
    if not FileExists(TestLabelsFile) then
      raise Exception.CreateFmt('t10k-labels-idx1-ubyte not found in: %s', [DataDir]);
  end;
  
  // Load the datasets with progress updates
  Application.ProcessMessages;
  TrainData := LoadMNISTDataset(TrainImagesFile, TrainLabelsFile,
    function(Current, Total: Integer; const Message: string): Boolean
    begin
      if Assigned(LblStatus) then
      begin
        LblStatus.Caption := Message;
        LblStatus.Width := ClientWidth - 20;  // Recalculate width to ensure full width
        LblStatus.Invalidate;  // Force repaint
        LblStatus.Repaint;
        Application.ProcessMessages;  // Process messages before continuing
      end;
      Result := True;  // Continue loading
    end);
  
  Application.ProcessMessages;
  TestData := LoadMNISTDataset(TestImagesFile, TestLabelsFile,
    function(Current, Total: Integer; const Message: string): Boolean
    begin
      if Assigned(LblStatus) then
      begin
        LblStatus.Caption := Message;
        LblStatus.Width := ClientWidth - 20;  // Recalculate width to ensure full width
        LblStatus.Invalidate;  // Force repaint
        LblStatus.Repaint;
        Application.ProcessMessages;  // Process messages before continuing
      end;
      Result := True;  // Continue loading
    end);
  
  if Assigned(LblStatus) then
    LblStatus.Caption := 'Data loaded successfully';
end;

// Initialize training epoch (shuffle indices)
procedure TMainForm.TrainEpoch;
var
  i, j: Integer;
begin
  if TrainData.Count = 0 then Exit;
  
  // Create shuffled indices for new epoch
  SetLength(TrainShuffledIndices, TrainData.Count);
  for i := 0 to TrainData.Count - 1 do
    TrainShuffledIndices[i] := i;
  
  // Fisher-Yates shuffle
  for i := TrainData.Count - 1 downto 1 do
  begin
    j := Random(i + 1);
    var Temp := TrainShuffledIndices[i];
    TrainShuffledIndices[i] := TrainShuffledIndices[j];
    TrainShuffledIndices[j] := Temp;
  end;
  
  TrainSampleIdx := 0;  // Reset sample counter
end;

// Train a single sample (called by timer, like XOR demo)
procedure TMainForm.TrainSingleSample;
var
  j: Integer;
  Loss: Single;
  Input4D: TArray<Single>;
  TargetOneHot: TArray<Single>;
  SamplesToTrain: Integer;
begin
  if TrainData.Count = 0 then Exit;
  
  // Initialize epoch if needed
  if TrainSampleIdx = 0 then
    TrainEpoch;
  
  SamplesToTrain := Min(1000, TrainData.Count);
  
  // Check if epoch is complete
  if TrainSampleIdx >= SamplesToTrain then
  begin
    Inc(EpochCount);
    TrainSampleIdx := 0;  // Start new epoch
    TrainEpoch;  // Reshuffle for new epoch
    
    // Update UI immediately when epoch completes
    LblEpoch.Caption := Format('Epoch: %d', [EpochCount]);
    Application.ProcessMessages;
  end;
  
  // Start of new batch? Zero gradients
  if BatchSampleCount = 0 then
    Network.ZeroGradients;
  
  // Train on current sample (accumulate gradient)
  var Idx := TrainShuffledIndices[TrainSampleIdx];
  
  // Convert [784] to [1, 1, 28, 28]
  SetLength(Input4D, 784);
  for j := 0 to 783 do
    Input4D[j] := TrainData.Images[Idx][j];
  
  // One-hot encode target
  TargetOneHot := OneHotEncode(TrainData.Labels[Idx], 10);
  
  // Accumulate gradient (no update yet)
  Loss := Network.AccumulateGradient(Input4D, TargetOneHot);
  BatchLossAccum := BatchLossAccum + Loss;
  Inc(BatchSampleCount);
  
  // End of batch? Apply gradients
  if BatchSampleCount >= BatchSize then
  begin
    Network.ApplyGradients(LearningRate, BatchSize, 5.0);
    
    // Record average loss for chart
    LossHistory.Add(BatchLossAccum / BatchSize);
    
    // Reset batch counters
    BatchSampleCount := 0;
    BatchLossAccum := 0;
  end;
  
  // Update loss display periodically
  if TrainSampleIdx mod 50 = 0 then
  begin
    LblLoss.Caption := Format('Loss: %.4f (Sample %d/%d, Batch %d)', 
      [Loss, TrainSampleIdx + 1, SamplesToTrain, BatchSize]);
    LblEpoch.Caption := Format('Epoch: %d', [EpochCount]);
    ChartPanel.Invalidate;
  end;
  
  Inc(TrainSampleIdx);
end;

procedure TMainForm.EvaluateTest;
var
  i, Correct: Integer;
  Predictions: TArray<Single>;
  PredictedClass: Integer;
  MaxProb: Single;
  ActualClass: Integer;
begin
  if TestData.Count = 0 then Exit;
  
  // Clear confusion matrix before evaluation
  ClearConfusionMatrix;
  
  Correct := 0;
  var SamplesToTest := Min(100, TestData.Count); // Test on 100 samples for speed
  
  for i := 0 to SamplesToTest - 1 do
  begin
    Predictions := Network.Forward(TestData.Images[i]);
    ActualClass := TestData.Labels[i];
    
    // Find predicted class (highest probability)
    PredictedClass := 0;
    MaxProb := Predictions[0];
    var j: Integer;
    for j := 1 to 9 do
      if Predictions[j] > MaxProb then
      begin
        MaxProb := Predictions[j];
        PredictedClass := j;
      end;
    
    // Update confusion matrix [actual, predicted]
    Inc(ConfusionMatrix[ActualClass, PredictedClass]);
    
    if PredictedClass = ActualClass then
      Inc(Correct);
  end;
  
  LblTestAcc.Caption := Format('Test Acc: %.1f%%', [100.0 * Correct / SamplesToTest]);
  ConfusionPanel.Invalidate;  // Refresh confusion matrix display
end;

procedure TMainForm.RenderImage;
var
  i, x, y: Integer;
  Pixel: Single;
  Canvas: TCanvas;
begin
  if (TestData.Count = 0) or (CurrentSampleIdx >= TestData.Count) then
    Exit;
  
  Canvas := PaintBox.Canvas;
  Canvas.Brush.Color := clWhite;
  Canvas.FillRect(PaintBox.ClientRect);
  
  // Draw 28x28 image, scaled up
  var Scale := Min(PaintBox.Width div 28, PaintBox.Height div 28);
  var OffsetX := (PaintBox.Width - 28 * Scale) div 2;
  var OffsetY := (PaintBox.Height - 28 * Scale) div 2;
  
  for y := 0 to 27 do
    for x := 0 to 27 do
    begin
      Pixel := TestData.Images[CurrentSampleIdx][y * 28 + x];
      var Gray := Round(Pixel * 255);
      Canvas.Brush.Color := RGB(Gray, Gray, Gray);
      Canvas.FillRect(Rect(OffsetX + x * Scale, OffsetY + y * Scale,
        OffsetX + (x + 1) * Scale, OffsetY + (y + 1) * Scale));
    end;
  
  // Show prediction and compute loss for this test sample
  var Predictions := Network.Forward(TestData.Images[CurrentSampleIdx]);
  var PredictedClass := 0;
  var MaxProb := Predictions[0];
  for i := 1 to 9 do
    if Predictions[i] > MaxProb then
    begin
      MaxProb := Predictions[i];
      PredictedClass := i;
    end;
  
  // Compute loss for this test sample to show why prediction might be wrong
  var TargetOneHot := OneHotEncode(TestData.Labels[CurrentSampleIdx], 10);
  var TestLoss: Single := 0;
  for i := 0 to 9 do
  begin
    var P := Predictions[i];
    if P < 1e-7 then P := 1e-7;
    if P > 1 - 1e-7 then P := 1 - 1e-7;
    TestLoss := TestLoss - (TargetOneHot[i] * Ln(P));
  end;
  TestLoss := TestLoss / 10;
  
  LblPrediction.Caption := Format('Prediction: %d (%.1f%%) | Actual: %d | Test Loss: %.4f',
    [PredictedClass, MaxProb * 100, TestData.Labels[CurrentSampleIdx], TestLoss]);
end;

procedure TMainForm.UpdateUI;
begin
  LblEpoch.Caption := Format('Epoch: %d', [EpochCount]);
  LearningRate := StrToFloatDef(EditLR.Text, 0.001);
  
  if IsTraining then
  begin
    BtnStartStop.Caption := 'Stop Training';
    Caption := 'NeuralDelphi - MNIST Demo (Training...)';
  end
  else
  begin
    BtnStartStop.Caption := 'Start Training';
    Caption := 'NeuralDelphi - MNIST Demo';
  end;
end;

procedure TMainForm.TimerTimer(Sender: TObject);
var
  SamplesPerTick: Integer;
begin
  if IsTraining then
  begin
    // Process multiple samples per tick for better throughput
    // Still keep UI responsive by processing in small batches
    SamplesPerTick := 5;  // Process 5 samples per timer tick (50ms)
    while SamplesPerTick > 0 do
    begin
      TrainSingleSample;
      Dec(SamplesPerTick);
      
      // Check if epoch completed
      if TrainSampleIdx = 0 then
        Break;
    end;
    
    // Update UI every 50 samples for better responsiveness
    if TrainSampleIdx mod 50 = 0 then
    begin
      EvaluateTest;
      // Cycle through test samples to show different images during training
      CurrentSampleIdx := (CurrentSampleIdx + 1) mod TestData.Count;
      RenderImage;
      UpdateUI;
    end;
  end;
end;

procedure TMainForm.PaintBoxPaint(Sender: TObject);
begin
  RenderImage;
end;

procedure TMainForm.BtnStartStopClick(Sender: TObject);
begin
  IsTraining := not IsTraining;
  Timer.Enabled := IsTraining;
  if IsTraining then
  begin
    TrainSampleIdx := 0;  // Reset to start new epoch
    TrainEpoch;  // Initialize shuffled indices
  end;
  UpdateUI;
end;

procedure TMainForm.BtnResetClick(Sender: TObject);
begin
  Network.Reset;
  EpochCount := 0;
  UpdateUI;
  RenderImage;
end;

procedure TMainForm.BtnNextClick(Sender: TObject);
begin
  if TestData.Count > 0 then
  begin
    CurrentSampleIdx := (CurrentSampleIdx + 1) mod TestData.Count;
    RenderImage;
  end;
end;

procedure TMainForm.BtnPrevClick(Sender: TObject);
begin
  if TestData.Count > 0 then
  begin
    CurrentSampleIdx := (CurrentSampleIdx - 1 + TestData.Count) mod TestData.Count;
    RenderImage;
  end;
end;

procedure TMainForm.BtnSaveClick(Sender: TObject);
var
  Dialog: TSaveDialog;
begin
  Dialog := TSaveDialog.Create(nil);
  try
    Dialog.Filter := 'Model Weights (*.weights)|*.weights|All Files (*.*)|*.*';
    Dialog.DefaultExt := 'weights';
    Dialog.FileName := 'mnist_model.weights';
    if Dialog.Execute then
    begin
      Network.Graph.SaveModel(Dialog.FileName);
      LblStatus.Caption := 'Model saved to ' + ExtractFileName(Dialog.FileName);
    end;
  finally
    Dialog.Free;
  end;
end;

procedure TMainForm.BtnLoadClick(Sender: TObject);
var
  Dialog: TOpenDialog;
begin
  Dialog := TOpenDialog.Create(nil);
  try
    Dialog.Filter := 'Model Weights (*.weights)|*.weights|All Files (*.*)|*.*';
    if Dialog.Execute then
    begin
      Network.Graph.LoadModel(Dialog.FileName);
      LblStatus.Caption := 'Model loaded from ' + ExtractFileName(Dialog.FileName);
      // Evaluate to see how loaded model performs
      EvaluateTest;
      RenderImage;
    end;
  finally
    Dialog.Free;
  end;
end;

procedure TMainForm.ClearConfusionMatrix;
var
  i, j: Integer;
begin
  for i := 0 to 9 do
    for j := 0 to 9 do
      ConfusionMatrix[i, j] := 0;
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
  
  // Border
  Canvas.Pen.Color := clBlack;
  Canvas.Pen.Width := 1;
  Canvas.Rectangle(0, 0, W, H);
  
  // Title
  Canvas.Font.Size := 8;
  Canvas.Font.Color := clBlack;
  Canvas.TextOut(5, 2, 'Loss History');
  
  if LossHistory.Count < 2 then
  begin
    Canvas.TextOut(W div 2 - 30, H div 2, 'No data yet');
    Exit;
  end;
  
  // Find min/max
  MaxLoss := LossHistory[0];
  MinLoss := LossHistory[0];
  for i := 1 to LossHistory.Count - 1 do
  begin
    if LossHistory[i] > MaxLoss then MaxLoss := LossHistory[i];
    if LossHistory[i] < MinLoss then MinLoss := LossHistory[i];
  end;
  
  if MaxLoss <= MinLoss then MaxLoss := MinLoss + 0.1;
  
  // Scaling
  ScaleX := (W - 20) / Max(1, LossHistory.Count - 1);
  ScaleY := (H - 30) / (MaxLoss - MinLoss);
  
  // Draw line chart
  Canvas.Pen.Color := clBlue;
  Canvas.Pen.Width := 2;
  
  PrevX := 10;
  PrevY := H - 15 - Round((LossHistory[0] - MinLoss) * ScaleY);
  
  for i := 1 to LossHistory.Count - 1 do
  begin
    X := 10 + Round(i * ScaleX);
    Y := H - 15 - Round((LossHistory[i] - MinLoss) * ScaleY);
    Canvas.MoveTo(PrevX, PrevY);
    Canvas.LineTo(X, Y);
    PrevX := X;
    PrevY := Y;
  end;
  
  // Y-axis labels
  Canvas.Font.Size := 7;
  Canvas.TextOut(2, 15, Format('%.2f', [MaxLoss]));
  Canvas.TextOut(2, H - 20, Format('%.2f', [MinLoss]));
end;

procedure TMainForm.ConfusionPanelPaint(Sender: TObject);
var
  Canvas: TCanvas;
  i, j: Integer;
  CellW, CellH: Integer;
  MaxVal: Integer;
  Intensity: Integer;
  X, Y: Integer;
begin
  Canvas := ConfusionPanel.Canvas;
  
  // Background
  Canvas.Brush.Color := clWhite;
  Canvas.FillRect(Rect(0, 0, ConfusionPanel.Width, ConfusionPanel.Height));
  
  // Title
  Canvas.Font.Size := 8;
  Canvas.Font.Color := clBlack;
  Canvas.TextOut(5, 2, 'Confusion Matrix (Row=Actual)');
  
  // Cell dimensions
  CellW := (ConfusionPanel.Width - 20) div 10;
  CellH := (ConfusionPanel.Height - 35) div 10;
  
  // Find max value for coloring
  MaxVal := 1;
  for i := 0 to 9 do
    for j := 0 to 9 do
      if ConfusionMatrix[i, j] > MaxVal then
        MaxVal := ConfusionMatrix[i, j];
  
  // Draw cells
  for i := 0 to 9 do
    for j := 0 to 9 do
    begin
      X := 15 + j * CellW;
      Y := 20 + i * CellH;
      
      // Color intensity based on value
      if i = j then
        // Diagonal (correct) = green
        Intensity := 255 - Min(255, (ConfusionMatrix[i, j] * 200) div MaxVal)
      else
        // Off-diagonal (errors) = red
        Intensity := 255 - Min(255, (ConfusionMatrix[i, j] * 200) div MaxVal);
      
      if i = j then
        Canvas.Brush.Color := RGB(Intensity, 255, Intensity)  // Green for correct
      else if ConfusionMatrix[i, j] > 0 then
        Canvas.Brush.Color := RGB(255, Intensity, Intensity)  // Red for errors
      else
        Canvas.Brush.Color := clWhite;
      
      Canvas.FillRect(Rect(X, Y, X + CellW - 1, Y + CellH - 1));
      Canvas.Pen.Color := clGray;
      Canvas.Rectangle(X, Y, X + CellW, Y + CellH);
    end;
  
  // Row/column labels
  Canvas.Font.Size := 7;
  for i := 0 to 9 do
  begin
    Canvas.TextOut(15 + i * CellW + CellW div 2 - 3, 20 + 10 * CellH + 2, IntToStr(i));
    Canvas.TextOut(3, 20 + i * CellH + CellH div 2 - 5, IntToStr(i));
  end;
end;

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TMainForm, MainForm);
  Application.Run;
end.

