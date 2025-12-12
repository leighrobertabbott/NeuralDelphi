unit MNIST_Loader;

interface

uses
  System.SysUtils,
  System.Classes,
  System.IOUtils,
  Vcl.Forms;

type
  TMNISTData = record
    Images: TArray<TArray<Single>>;  // [Count][784] for 28x28 images
    Labels: TArray<Integer>;         // [Count]
    Count: Integer;
  end;
  
  // Progress callback: Current, Total, returns True to continue, False to cancel
  TProgressCallback = reference to function(Current, Total: Integer; const Message: string): Boolean;

// Load MNIST images from IDX file format
function LoadMNISTImages(const FileName: string; ProgressCallback: TProgressCallback = nil): TArray<TArray<Single>>;

// Load MNIST labels from IDX file format
function LoadMNISTLabels(const FileName: string; ProgressCallback: TProgressCallback = nil): TArray<Integer>;

// Load complete MNIST dataset (images + labels)
function LoadMNISTDataset(const ImagesFile, LabelsFile: string; ProgressCallback: TProgressCallback = nil): TMNISTData;

// One-hot encode labels for classification (0-9 -> [10] array)
function OneHotEncode(LabelValue: Integer; NumClasses: Integer = 10): TArray<Single>;

implementation

function LoadMNISTImages(const FileName: string; ProgressCallback: TProgressCallback): TArray<TArray<Single>>;
var
  Stream: TFileStream;
  Magic, NumImages, Rows, Cols: UInt32;
  i, j: Integer;
  ImageSize: Integer;
  ImageBuffer: TArray<Byte>;  // Buffer for entire image
begin
  if not FileExists(FileName) then
    raise Exception.CreateFmt('MNIST images file not found: %s', [FileName]);
  
  Stream := TFileStream.Create(FileName, fmOpenRead);
  try
    // Read magic number (should be 0x00000803 for images)
    Stream.Read(Magic, 4);
    // Convert from big-endian to little-endian
    Magic := ((Magic and $FF000000) shr 24) or
             ((Magic and $00FF0000) shr 8) or
             ((Magic and $0000FF00) shl 8) or
             ((Magic and $000000FF) shl 24);
    
    if Magic <> $00000803 then
      raise Exception.CreateFmt('Invalid MNIST images file: magic number 0x%x (expected 0x803)', [Magic]);
    
    // Read dimensions
    Stream.Read(NumImages, 4);
    NumImages := ((NumImages and $FF000000) shr 24) or
                 ((NumImages and $00FF0000) shr 8) or
                 ((NumImages and $0000FF00) shl 8) or
                 ((NumImages and $000000FF) shl 24);
    
    Stream.Read(Rows, 4);
    Rows := ((Rows and $FF000000) shr 24) or
            ((Rows and $00FF0000) shr 8) or
            ((Rows and $0000FF00) shl 8) or
            ((Rows and $000000FF) shl 24);
    
    Stream.Read(Cols, 4);
    Cols := ((Cols and $FF000000) shr 24) or
            ((Cols and $00FF0000) shr 8) or
            ((Cols and $0000FF00) shl 8) or
            ((Cols and $000000FF) shl 24);
    
    ImageSize := Rows * Cols;  // Should be 784 for 28x28
    
    SetLength(Result, NumImages);
    SetLength(ImageBuffer, ImageSize);  // Reusable buffer
    
    // Read each image (bulk read instead of byte-by-byte)
    for i := 0 to NumImages - 1 do
    begin
      SetLength(Result[i], ImageSize);
      
      // Read entire image at once (784 bytes)
      Stream.ReadBuffer(ImageBuffer[0], ImageSize);
      
      // Convert bytes to normalized floats
      for j := 0 to ImageSize - 1 do
        Result[i][j] := ImageBuffer[j] / 255.0;
      
      // Report progress every 1000 images
      if Assigned(ProgressCallback) and ((i mod 1000 = 0) or (i = Integer(NumImages) - 1)) then
      begin
        var Percent := Round((i + 1) * 100.0 / NumImages);
        if not ProgressCallback(i + 1, Integer(NumImages), Format('Images: %d%%', [Percent])) then
          Break;
      end;
    end;
  finally
    Stream.Free;
  end;
end;

function LoadMNISTLabels(const FileName: string; ProgressCallback: TProgressCallback): TArray<Integer>;
var
  Stream: TFileStream;
  Magic, NumLabels: UInt32;
  i: Integer;
  LabelBuffer: TArray<Byte>;  // Buffer for all labels
begin
  if not FileExists(FileName) then
    raise Exception.CreateFmt('MNIST labels file not found: %s', [FileName]);
  
  Stream := TFileStream.Create(FileName, fmOpenRead);
  try
    // Read magic number (should be 0x00000801 for labels)
    Stream.Read(Magic, 4);
    // Convert from big-endian to little-endian
    Magic := ((Magic and $FF000000) shr 24) or
             ((Magic and $00FF0000) shr 8) or
             ((Magic and $0000FF00) shl 8) or
             ((Magic and $000000FF) shl 24);
    
    if Magic <> $00000801 then
      raise Exception.CreateFmt('Invalid MNIST labels file: magic number 0x%x (expected 0x801)', [Magic]);
    
    // Read number of labels
    Stream.Read(NumLabels, 4);
    NumLabels := ((NumLabels and $FF000000) shr 24) or
                 ((NumLabels and $00FF0000) shr 8) or
                 ((NumLabels and $0000FF00) shl 8) or
                 ((NumLabels and $000000FF) shl 24);
    
    // Read ALL labels at once
    SetLength(LabelBuffer, NumLabels);
    Stream.ReadBuffer(LabelBuffer[0], NumLabels);
    
    // Convert to integers
    SetLength(Result, NumLabels);
    for i := 0 to NumLabels - 1 do
      Result[i] := LabelBuffer[i];
    
    if Assigned(ProgressCallback) then
      ProgressCallback(Integer(NumLabels), Integer(NumLabels), 'Labels: 100%');
  finally
    Stream.Free;
  end;
end;

function LoadMNISTDataset(const ImagesFile, LabelsFile: string; ProgressCallback: TProgressCallback): TMNISTData;
var
  Images: TArray<TArray<Single>>;
  Labels: TArray<Integer>;
begin
  Images := LoadMNISTImages(ImagesFile, ProgressCallback);
  Labels := LoadMNISTLabels(LabelsFile, ProgressCallback);
  
  if Length(Images) <> Length(Labels) then
    raise Exception.CreateFmt('Image count (%d) does not match label count (%d)',
      [Length(Images), Length(Labels)]);
  
  Result.Images := Images;
  Result.Labels := Labels;
  Result.Count := Length(Images);
end;

function OneHotEncode(LabelValue: Integer; NumClasses: Integer): TArray<Single>;
var
  i: Integer;
begin
  SetLength(Result, NumClasses);
  for i := 0 to NumClasses - 1 do
    Result[i] := 0.0;
  
  if (LabelValue >= 0) and (LabelValue < NumClasses) then
    Result[LabelValue] := 1.0;
end;

end.

