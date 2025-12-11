unit ML.Arena;

{$R-} // Disable range checking for performance - we do manual bounds checking

interface

uses
  System.SysUtils;

type
  // The 'Handle' for any data in our system.
  // It is just an index, not a pointer. Safe and 4 bytes.
  TMemPtr = Integer;

  TArena = class
  private
    FMemory: TArray<Single>;
    FHead: Integer;
    FCapacity: Integer;
  public
    constructor Create(SizeMB: Integer = 256);
    
    // Returns the index of the start of the block
    function Alloc(Count: Integer): TMemPtr;
    
    // Fast pointer access for the Math Kernel
    function GetPtr(Ptr: TMemPtr): PSingle; inline;
    
    // Resets the 'Head' to 0 (or a save point) instantly wiping 'memory'
    procedure Reset;
    
    // Get current head position for save point
    function GetSavePoint: Integer;
    
    // Restore to a save point
    procedure Restore(SavePoint: Integer);
    
    // Properties
    property Capacity: Integer read FCapacity;
    property Head: Integer read FHead;
  end;

implementation

constructor TArena.Create(SizeMB: Integer);
begin
  // SizeMB * 1024 * 1024 / 4 bytes per float
  FCapacity := (SizeMB * 1024 * 1024) div SizeOf(Single);
  SetLength(FMemory, FCapacity);
  FHead := 0;
end;

function TArena.Alloc(Count: Integer): TMemPtr;
begin
  if FHead + Count > FCapacity then
    raise Exception.CreateFmt('Arena Out of Memory (OOM): Requested %d, Available %d', 
      [Count, FCapacity - FHead]);
  
  Result := FHead;
  Inc(FHead, Count);
end;

function TArena.GetPtr(Ptr: TMemPtr): PSingle;
var
  P: PSingle;
begin
  // Return direct memory address for raw speed
  // Bounds check to prevent range check errors
  if (Ptr < 0) or (Ptr >= FCapacity) then
    raise Exception.CreateFmt('Arena: Invalid pointer %d (Capacity: %d)', [Ptr, FCapacity]);
  // Use pointer arithmetic to avoid range checking issues
  P := @FMemory[0];
  Inc(P, Ptr);
  Result := P;
end;

procedure TArena.Reset;
begin
  FHead := 0; // The fastest garbage collector in the world
end;

function TArena.GetSavePoint: Integer;
begin
  Result := FHead;
end;

procedure TArena.Restore(SavePoint: Integer);
begin
  if SavePoint < 0 then SavePoint := 0;
  if SavePoint > FCapacity then SavePoint := FCapacity;
  FHead := SavePoint;
end;

end.

