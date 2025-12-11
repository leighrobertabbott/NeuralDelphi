unit ML.Tensor;

{$R-} // Disable range checking for performance - we do manual bounds checking

interface

uses
  System.SysUtils,
  ML.Arena;

type
  TTensor = record
    DataPtr: TMemPtr;
    GradPtr: TMemPtr;  // Gradients live right next to data, or allocated on demand
    Rows, Cols: Integer;
    RequiresGrad: Boolean;
    
    // Convenience to get raw pointer
    function RawData(Arena: TArena): PSingle; inline;
    function RawGrad(Arena: TArena): PSingle; inline;
    
    // Shape utilities
    function ElementCount: Integer; inline;
    function IsScalar: Boolean; inline;
    function SameShape(const Other: TTensor): Boolean; inline;
    
    // Factory
    class function CreateTensor(Arena: TArena; ARows, ACols: Integer; 
      ARequiresGrad: Boolean = False): TTensor; static;
  end;

function CreateTensor(Arena: TArena; Rows, Cols: Integer; 
  RequiresGrad: Boolean = False): TTensor; inline;

implementation

function TTensor.RawData(Arena: TArena): PSingle;
begin
  Result := Arena.GetPtr(DataPtr);
end;

function TTensor.RawGrad(Arena: TArena): PSingle;
begin
  if GradPtr < 0 then
    Result := nil  // Return nil if gradient not allocated - caller should check
  else
    Result := Arena.GetPtr(GradPtr);
end;

function TTensor.ElementCount: Integer;
begin
  Result := Rows * Cols;
end;

function TTensor.IsScalar: Boolean;
begin
  Result := (Rows = 1) and (Cols = 1);
end;

function TTensor.SameShape(const Other: TTensor): Boolean;
begin
  Result := (Rows = Other.Rows) and (Cols = Other.Cols);
end;

class function TTensor.CreateTensor(Arena: TArena; ARows, ACols: Integer; 
  ARequiresGrad: Boolean): TTensor;
begin
  Result.Rows := ARows;
  Result.Cols := ACols;
  Result.DataPtr := Arena.Alloc(ARows * ACols);
  Result.RequiresGrad := ARequiresGrad;
  Result.GradPtr := -1; // Allocated on demand during backward pass
end;

function CreateTensor(Arena: TArena; Rows, Cols: Integer; 
  RequiresGrad: Boolean): TTensor;
begin
  Result := TTensor.CreateTensor(Arena, Rows, Cols, RequiresGrad);
end;

end.

