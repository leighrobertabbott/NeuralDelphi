unit ML.Arena;

{$R-} // Disable range checking for performance - we do manual bounds checking

interface

uses
  System.SysUtils;

type
  // The 'Handle' for any data in our system.
  // It is just an index, not a pointer. Safe and 4 bytes.
  TMemPtr = Integer;

  TArenaStats = record
    TotalCapacity: Integer;      // Total capacity in bytes
    UsedBytes: Integer;           // Currently used bytes
    AvailableBytes: Integer;      // Available bytes
    UsagePercent: Single;         // Usage percentage (0-100)
    AllocationCount: Integer;     // Number of allocations
  end;

  TArena = class
  private
    FMemory: TArray<Single>;
    FHead: Integer;
    FCapacity: Integer;
    FAllocationCount: Integer;  // Track number of allocations
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
    
    // Memory monitoring
    function GetUsageStats: TArenaStats;
    
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
  FAllocationCount := 0;
end;

function TArena.Alloc(Count: Integer): TMemPtr;
begin
  if FHead + Count > FCapacity then
    raise Exception.CreateFmt('Arena Out of Memory (OOM): Requested %d, Available %d', 
      [Count, FCapacity - FHead]);
  
  Result := FHead;
  Inc(FHead, Count);
  Inc(FAllocationCount);
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
  FHead := 0;
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

function TArena.GetUsageStats: TArenaStats;
var
  TotalBytes, UsedBytes: Integer;
begin
  TotalBytes := FCapacity * SizeOf(Single);
  UsedBytes := FHead * SizeOf(Single);
  
  Result.TotalCapacity := TotalBytes;
  Result.UsedBytes := UsedBytes;
  Result.AvailableBytes := TotalBytes - UsedBytes;
  if TotalBytes > 0 then
    Result.UsagePercent := (UsedBytes / TotalBytes) * 100.0
  else
    Result.UsagePercent := 0.0;
  Result.AllocationCount := FAllocationCount;
end;

end.

