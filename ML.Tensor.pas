unit ML.Tensor;

{$R-} // Disable range checking for performance - we do manual bounds checking

interface

uses
  System.SysUtils,
  ML.Arena;

type
  TTensor = record
    DataPtr: TMemPtr;
    GradPtr: TMemPtr;
    Shape: TArray<Integer>;    // e.g., [32, 3, 224, 224] for 4D tensor
    Strides: TArray<Integer>;  // e.g., [150528, 50176, 224, 1] for row-major
    RequiresGrad: Boolean;
    
    // Raw pointer access
    function RawData(Arena: TArena): PSingle; inline;
    function RawGrad(Arena: TArena): PSingle; inline;
    
    // Core properties
    function NDim: Integer; inline;
    function ElementCount: Integer;
    function IsContiguous: Boolean;
    function IsScalar: Boolean; inline;
    
    // Indexing
    function GetLinearIndex(const Indices: array of Integer): Integer;
    
    // View operations (zero-copy)
    function Reshape(const NewShape: array of Integer): TTensor;
    function Transpose(Dim0, Dim1: Integer): TTensor;
    function Squeeze(Dim: Integer = -1): TTensor;
    function Unsqueeze(Dim: Integer): TTensor;
    
    // Shape utilities
    function SameShape(const Other: TTensor): Boolean;
    
    // Factory
    class function Create(Arena: TArena; const AShape: array of Integer;
      ARequiresGrad: Boolean = False): TTensor; static;
  end;

function ComputeStrides(const Shape: array of Integer): TArray<Integer>;

implementation

function ComputeStrides(const Shape: array of Integer): TArray<Integer>;
var
  i, Stride: Integer;
begin
  SetLength(Result, Length(Shape));
  Stride := 1;
  // Compute strides from right to left (last dimension has stride 1)
  for i := High(Shape) downto 0 do
  begin
    Result[i] := Stride;
    Stride := Stride * Shape[i];
  end;
end;

function TTensor.RawData(Arena: TArena): PSingle;
begin
  Result := Arena.GetPtr(DataPtr);
end;

function TTensor.RawGrad(Arena: TArena): PSingle;
begin
  if GradPtr < 0 then
    Result := nil
  else
    Result := Arena.GetPtr(GradPtr);
end;

function TTensor.NDim: Integer;
begin
  Result := Length(Shape);
end;

function TTensor.ElementCount: Integer;
var
  i: Integer;
begin
  Result := 1;
  for i := 0 to High(Shape) do
    Result := Result * Shape[i];
end;

function TTensor.IsContiguous: Boolean;
var
  Expected: TArray<Integer>;
  i: Integer;
begin
  if Length(Shape) = 0 then
    Exit(True);
    
  Expected := ComputeStrides(Shape);
  if Length(Strides) <> Length(Expected) then
    Exit(False);
    
  for i := 0 to High(Strides) do
    if Strides[i] <> Expected[i] then
      Exit(False);
      
  Result := True;
end;

function TTensor.IsScalar: Boolean;
begin
  Result := (Length(Shape) = 0) or ((Length(Shape) = 1) and (Shape[0] = 1)) or
            ((Length(Shape) = 2) and (Shape[0] = 1) and (Shape[1] = 1));
end;

function TTensor.GetLinearIndex(const Indices: array of Integer): Integer;
var
  i: Integer;
begin
  if Length(Indices) <> Length(Shape) then
    raise Exception.CreateFmt('GetLinearIndex: Index count (%d) != Shape dims (%d)',
      [Length(Indices), Length(Shape)]);
      
  Result := 0;
  for i := 0 to High(Indices) do
  begin
    if (Indices[i] < 0) or (Indices[i] >= Shape[i]) then
      raise Exception.CreateFmt('GetLinearIndex: Index %d out of range [0, %d)',
        [Indices[i], Shape[i]]);
    Result := Result + Indices[i] * Strides[i];
  end;
end;

function TTensor.Reshape(const NewShape: array of Integer): TTensor;
var
  NewCount, OldCount: Integer;
  i: Integer;
begin
  // Compute element count for new shape
  NewCount := 1;
  for i := 0 to High(NewShape) do
  begin
    if NewShape[i] < 0 then
      raise Exception.Create('Reshape: Negative dimensions not supported');
    NewCount := NewCount * NewShape[i];
  end;
  
  // Compute old element count
  OldCount := ElementCount;
  
  if NewCount <> OldCount then
    raise Exception.CreateFmt('Reshape: Cannot reshape [%d] elements to [%d] elements',
      [OldCount, NewCount]);
  
  // Create view with new shape but same data
  Result.DataPtr := DataPtr;
  Result.GradPtr := GradPtr;
  Result.RequiresGrad := RequiresGrad;
  SetLength(Result.Shape, Length(NewShape));
  for i := 0 to High(NewShape) do
    Result.Shape[i] := NewShape[i];
  Result.Strides := ComputeStrides(Result.Shape);
end;

function TTensor.Transpose(Dim0, Dim1: Integer): TTensor;
var
  i: Integer;
begin
  if (Dim0 < 0) or (Dim0 >= Length(Shape)) or (Dim1 < 0) or (Dim1 >= Length(Shape)) then
    raise Exception.CreateFmt('Transpose: Invalid dimensions %d, %d (NDim=%d)',
      [Dim0, Dim1, Length(Shape)]);
      
  if Dim0 = Dim1 then
  begin
    // No-op: return copy of self
    Result := Self;
    Exit;
  end;
  
  // Create view with swapped dimensions
  Result.DataPtr := DataPtr;
  Result.GradPtr := GradPtr;
  Result.RequiresGrad := RequiresGrad;
  SetLength(Result.Shape, Length(Shape));
  SetLength(Result.Strides, Length(Strides));
  
  // Copy shape and strides
  for i := 0 to High(Shape) do
  begin
    Result.Shape[i] := Shape[i];
    Result.Strides[i] := Strides[i];
  end;
  
  // Swap dimensions
  i := Result.Shape[Dim0];
  Result.Shape[Dim0] := Result.Shape[Dim1];
  Result.Shape[Dim1] := i;
  
  i := Result.Strides[Dim0];
  Result.Strides[Dim0] := Result.Strides[Dim1];
  Result.Strides[Dim1] := i;
end;

function TTensor.Squeeze(Dim: Integer): TTensor;
var
  i, j: Integer;
begin
  Result.DataPtr := DataPtr;
  Result.GradPtr := GradPtr;
  Result.RequiresGrad := RequiresGrad;
  
  if Length(Shape) = 0 then
  begin
    // Already scalar
    Result.Shape := Shape;
    Result.Strides := Strides;
    Exit;
  end;
  
  if Dim = -1 then
  begin
    // Remove all dimensions of size 1
    SetLength(Result.Shape, 0);
    SetLength(Result.Strides, 0);
    for i := 0 to High(Shape) do
    begin
      if Shape[i] <> 1 then
      begin
        SetLength(Result.Shape, Length(Result.Shape) + 1);
        SetLength(Result.Strides, Length(Result.Strides) + 1);
        Result.Shape[High(Result.Shape)] := Shape[i];
        Result.Strides[High(Result.Strides)] := Strides[i];
      end;
    end;
  end
  else
  begin
    // Remove specific dimension
    if (Dim < 0) or (Dim >= Length(Shape)) then
      raise Exception.CreateFmt('Squeeze: Invalid dimension %d (NDim=%d)', [Dim, Length(Shape)]);
      
    if Shape[Dim] <> 1 then
      raise Exception.CreateFmt('Squeeze: Dimension %d has size %d, not 1', [Dim, Shape[Dim]]);
      
    SetLength(Result.Shape, Length(Shape) - 1);
    SetLength(Result.Strides, Length(Strides) - 1);
    j := 0;
    for i := 0 to High(Shape) do
    begin
      if i <> Dim then
      begin
        Result.Shape[j] := Shape[i];
        Result.Strides[j] := Strides[i];
        Inc(j);
      end;
    end;
  end;
end;

function TTensor.Unsqueeze(Dim: Integer): TTensor;
var
  i: Integer;
begin
  if Dim < 0 then
    Dim := Length(Shape) + Dim + 1;
    
  if (Dim < 0) or (Dim > Length(Shape)) then
    raise Exception.CreateFmt('Unsqueeze: Invalid dimension %d (NDim=%d)',
      [Dim, Length(Shape)]);
  
  Result.DataPtr := DataPtr;
  Result.GradPtr := GradPtr;
  Result.RequiresGrad := RequiresGrad;
  SetLength(Result.Shape, Length(Shape) + 1);
  SetLength(Result.Strides, Length(Strides) + 1);
  
  // Insert dimension of size 1
  for i := 0 to Dim - 1 do
  begin
    Result.Shape[i] := Shape[i];
    Result.Strides[i] := Strides[i];
  end;
  Result.Shape[Dim] := 1;
  if Dim < Length(Shape) then
    Result.Strides[Dim] := Strides[Dim]
  else
    Result.Strides[Dim] := 1;
  for i := Dim to High(Shape) do
  begin
    Result.Shape[i + 1] := Shape[i];
    Result.Strides[i + 1] := Strides[i];
  end;
end;

function TTensor.SameShape(const Other: TTensor): Boolean;
var
  i: Integer;
begin
  if Length(Shape) <> Length(Other.Shape) then
    Exit(False);
    
  for i := 0 to High(Shape) do
    if Shape[i] <> Other.Shape[i] then
      Exit(False);
      
  Result := True;
end;

class function TTensor.Create(Arena: TArena; const AShape: array of Integer;
  ARequiresGrad: Boolean): TTensor;
var
  i, Count: Integer;
begin
  if Length(AShape) = 0 then
    raise Exception.Create('TTensor.Create: Shape cannot be empty');
    
  // Validate shape
  Count := 1;
  for i := 0 to High(AShape) do
  begin
    if AShape[i] <= 0 then
      raise Exception.CreateFmt('TTensor.Create: Invalid shape dimension %d: %d',
        [i, AShape[i]]);
    Count := Count * AShape[i];
  end;
  
  Result.DataPtr := Arena.Alloc(Count);
  Result.GradPtr := -1;
  Result.RequiresGrad := ARequiresGrad;
  
  SetLength(Result.Shape, Length(AShape));
  for i := 0 to High(AShape) do
    Result.Shape[i] := AShape[i];
    
  Result.Strides := ComputeStrides(Result.Shape);
end;

end.
