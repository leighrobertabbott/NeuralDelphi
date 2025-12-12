unit ML.Cuda;

{$R-} // Disable range checking for performance

interface

uses
  System.SysUtils, Winapi.Windows;

type
  // CUDA Driver API types
  TCUresult = Integer;
  TCUdevice = Integer;
  TCUcontext = Pointer;
  TCUmodule = Pointer;
  TCUfunction = Pointer;
  TCUstream = Pointer;
  TCUdeviceptr = UInt64;  // GPU memory pointer (64-bit even on 32-bit systems)

const
  // Common CUDA error codes
  CUDA_SUCCESS = 0;
  CUDA_ERROR_INVALID_VALUE = 1;
  CUDA_ERROR_OUT_OF_MEMORY = 2;
  CUDA_ERROR_NOT_INITIALIZED = 3;
  CUDA_ERROR_NO_DEVICE = 100;
  CUDA_ERROR_INVALID_CONTEXT = 201;
  CUDA_ERROR_INVALID_HANDLE = 400;

type
  // Function pointer types for dynamic loading
  TcuInit = function(Flags: Cardinal): TCUresult; cdecl;
  TcuDriverGetVersion = function(var DriverVersion: Integer): TCUresult; cdecl;
  TcuDeviceGetCount = function(var Count: Integer): TCUresult; cdecl;
  TcuDeviceGet = function(var Device: TCUdevice; Ordinal: Integer): TCUresult; cdecl;
  TcuDeviceGetName = function(Name: PAnsiChar; Len: Integer; Dev: TCUdevice): TCUresult; cdecl;
  TcuDeviceTotalMem = function(var Bytes: NativeUInt; Dev: TCUdevice): TCUresult; cdecl;
  TcuCtxCreate = function(var Ctx: TCUcontext; Flags: Cardinal; Dev: TCUdevice): TCUresult; cdecl;
  TcuCtxDestroy = function(Ctx: TCUcontext): TCUresult; cdecl;
  TcuCtxSynchronize = function: TCUresult; cdecl;
  TcuMemAlloc = function(var DevPtr: TCUdeviceptr; ByteSize: NativeUInt): TCUresult; cdecl;
  TcuMemFree = function(DevPtr: TCUdeviceptr): TCUresult; cdecl;
  TcuMemcpyHtoD = function(DstDevice: TCUdeviceptr; SrcHost: Pointer; ByteCount: NativeUInt): TCUresult; cdecl;
  TcuMemcpyDtoH = function(DstHost: Pointer; SrcDevice: TCUdeviceptr; ByteCount: NativeUInt): TCUresult; cdecl;
  TcuModuleLoadData = function(var Module: TCUmodule; Image: PAnsiChar): TCUresult; cdecl;
  TcuModuleUnload = function(Module: TCUmodule): TCUresult; cdecl;
  TcuModuleGetFunction = function(var Func: TCUfunction; Module: TCUmodule; Name: PAnsiChar): TCUresult; cdecl;
  TcuLaunchKernel = function(
    F: TCUfunction;
    GridDimX, GridDimY, GridDimZ: Cardinal;
    BlockDimX, BlockDimY, BlockDimZ: Cardinal;
    SharedMemBytes: Cardinal;
    Stream: TCUstream;
    KernelParams: PPointer;
    Extra: PPointer): TCUresult; cdecl;

var
  // Dynamically loaded function pointers
  cuInit: TcuInit = nil;
  cuDriverGetVersion: TcuDriverGetVersion = nil;
  cuDeviceGetCount: TcuDeviceGetCount = nil;
  cuDeviceGet: TcuDeviceGet = nil;
  cuDeviceGetName: TcuDeviceGetName = nil;
  cuDeviceTotalMem: TcuDeviceTotalMem = nil;
  cuCtxCreate: TcuCtxCreate = nil;
  cuCtxDestroy: TcuCtxDestroy = nil;
  cuCtxSynchronize: TcuCtxSynchronize = nil;
  cuMemAlloc: TcuMemAlloc = nil;
  cuMemFree: TcuMemFree = nil;
  cuMemcpyHtoD: TcuMemcpyHtoD = nil;
  cuMemcpyDtoH: TcuMemcpyDtoH = nil;
  cuModuleLoadData: TcuModuleLoadData = nil;
  cuModuleUnload: TcuModuleUnload = nil;
  cuModuleGetFunction: TcuModuleGetFunction = nil;
  cuLaunchKernel: TcuLaunchKernel = nil;

type
  // High-level wrapper class
  TCudaContext = class
  private
    FDevice: TCUdevice;
    FContext: TCUcontext;
    FDeviceName: string;
    FTotalMemory: NativeUInt;
    FDriverVersion: Integer;
    FInitialized: Boolean;
  public
    constructor Create(DeviceOrdinal: Integer = 0);
    destructor Destroy; override;
    
    // Memory operations
    function AllocDevice(ByteSize: NativeUInt): TCUdeviceptr;
    procedure FreeDevice(DevPtr: TCUdeviceptr);
    procedure CopyToDevice(DevPtr: TCUdeviceptr; HostPtr: Pointer; ByteSize: NativeUInt);
    procedure CopyFromDevice(HostPtr: Pointer; DevPtr: TCUdeviceptr; ByteSize: NativeUInt);
    procedure Synchronize;
    
    // Module/Kernel operations
    function LoadModule(const PTXCode: AnsiString): TCUmodule;
    procedure UnloadModule(Module: TCUmodule);
    function GetFunction(Module: TCUmodule; const FuncName: AnsiString): TCUfunction;
    procedure LaunchKernel(Func: TCUfunction; 
      GridDimX, GridDimY, GridDimZ: Cardinal;
      BlockDimX, BlockDimY, BlockDimZ: Cardinal;
      SharedMemBytes: Cardinal;
      const Args: array of Pointer);
    
    property Initialized: Boolean read FInitialized;
    property DeviceName: string read FDeviceName;
    property TotalMemory: NativeUInt read FTotalMemory;
    property DriverVersion: Integer read FDriverVersion;
  end;

// Global functions
function CudaAvailable: Boolean;
function LoadCudaDriver: Boolean;
procedure UnloadCudaDriver;
function CudaResultToString(Res: TCUresult): string;

implementation

var
  CudaDllHandle: HMODULE = 0;
  CudaLoaded: Boolean = False;

function CudaAvailable: Boolean;
begin
  Result := CudaLoaded and Assigned(cuInit);
end;

function LoadCudaDriver: Boolean;

  function GetProc(const Name: string): Pointer;
  begin
    Result := GetProcAddress(CudaDllHandle, PChar(Name));
  end;

begin
  if CudaLoaded then
    Exit(True);
    
  // Try to load nvcuda.dll (installed with NVIDIA drivers)
  CudaDllHandle := LoadLibrary('nvcuda.dll');
  if CudaDllHandle = 0 then
    Exit(False);
    
  // Load all function pointers
  @cuInit := GetProc('cuInit');
  @cuDriverGetVersion := GetProc('cuDriverGetVersion');
  @cuDeviceGetCount := GetProc('cuDeviceGetCount');
  @cuDeviceGet := GetProc('cuDeviceGet');
  @cuDeviceGetName := GetProc('cuDeviceGetName');
  @cuDeviceTotalMem := GetProc('cuDeviceTotalMem_v2');
  @cuCtxCreate := GetProc('cuCtxCreate_v2');
  @cuCtxDestroy := GetProc('cuCtxDestroy_v2');
  @cuCtxSynchronize := GetProc('cuCtxSynchronize');
  @cuMemAlloc := GetProc('cuMemAlloc_v2');
  @cuMemFree := GetProc('cuMemFree_v2');
  @cuMemcpyHtoD := GetProc('cuMemcpyHtoD_v2');
  @cuMemcpyDtoH := GetProc('cuMemcpyDtoH_v2');
  @cuModuleLoadData := GetProc('cuModuleLoadData');
  @cuModuleUnload := GetProc('cuModuleUnload');
  @cuModuleGetFunction := GetProc('cuModuleGetFunction');
  @cuLaunchKernel := GetProc('cuLaunchKernel');
  
  // Verify essential functions are available
  if not Assigned(cuInit) or not Assigned(cuDeviceGet) or 
     not Assigned(cuCtxCreate) or not Assigned(cuMemAlloc) then
  begin
    FreeLibrary(CudaDllHandle);
    CudaDllHandle := 0;
    Exit(False);
  end;
  
  // Initialize CUDA
  if cuInit(0) <> CUDA_SUCCESS then
  begin
    FreeLibrary(CudaDllHandle);
    CudaDllHandle := 0;
    Exit(False);
  end;
  
  CudaLoaded := True;
  Result := True;
end;

procedure UnloadCudaDriver;
begin
  if CudaDllHandle <> 0 then
  begin
    FreeLibrary(CudaDllHandle);
    CudaDllHandle := 0;
  end;
  CudaLoaded := False;
  
  // Clear function pointers
  cuInit := nil;
  cuDriverGetVersion := nil;
  cuDeviceGetCount := nil;
  cuDeviceGet := nil;
  cuDeviceGetName := nil;
  cuDeviceTotalMem := nil;
  cuCtxCreate := nil;
  cuCtxDestroy := nil;
  cuCtxSynchronize := nil;
  cuMemAlloc := nil;
  cuMemFree := nil;
  cuMemcpyHtoD := nil;
  cuMemcpyDtoH := nil;
  cuModuleLoadData := nil;
  cuModuleUnload := nil;
  cuModuleGetFunction := nil;
  cuLaunchKernel := nil;
end;

function CudaResultToString(Res: TCUresult): string;
begin
  case Res of
    CUDA_SUCCESS: Result := 'Success';
    CUDA_ERROR_INVALID_VALUE: Result := 'Invalid value';
    CUDA_ERROR_OUT_OF_MEMORY: Result := 'Out of memory';
    CUDA_ERROR_NOT_INITIALIZED: Result := 'Not initialized';
    CUDA_ERROR_NO_DEVICE: Result := 'No CUDA device';
    CUDA_ERROR_INVALID_CONTEXT: Result := 'Invalid context';
    CUDA_ERROR_INVALID_HANDLE: Result := 'Invalid handle';
  else
    Result := Format('Unknown error %d', [Res]);
  end;
end;

{ TCudaContext }

constructor TCudaContext.Create(DeviceOrdinal: Integer);
var
  Res: TCUresult;
  NameBuf: array[0..255] of AnsiChar;
begin
  inherited Create;
  FInitialized := False;
  
  if not LoadCudaDriver then
    raise Exception.Create('CUDA driver not available. Is NVIDIA driver installed?');
  
  // Get device
  Res := cuDeviceGet(FDevice, DeviceOrdinal);
  if Res <> CUDA_SUCCESS then
    raise Exception.CreateFmt('cuDeviceGet failed: %s', [CudaResultToString(Res)]);
  
  // Get device name
  Res := cuDeviceGetName(@NameBuf[0], Length(NameBuf), FDevice);
  if Res = CUDA_SUCCESS then
    FDeviceName := string(AnsiString(NameBuf))
  else
    FDeviceName := 'Unknown';
  
  // Get total memory
  Res := cuDeviceTotalMem(FTotalMemory, FDevice);
  if Res <> CUDA_SUCCESS then
    FTotalMemory := 0;
  
  // Get driver version
  if Assigned(cuDriverGetVersion) then
    cuDriverGetVersion(FDriverVersion)
  else
    FDriverVersion := 0;
  
  // Create context
  Res := cuCtxCreate(FContext, 0, FDevice);
  if Res <> CUDA_SUCCESS then
    raise Exception.CreateFmt('cuCtxCreate failed: %s', [CudaResultToString(Res)]);
  
  FInitialized := True;
end;

destructor TCudaContext.Destroy;
begin
  if FInitialized and (FContext <> nil) then
    cuCtxDestroy(FContext);
  inherited;
end;

function TCudaContext.AllocDevice(ByteSize: NativeUInt): TCUdeviceptr;
var
  Res: TCUresult;
begin
  Result := 0;
  if not FInitialized then Exit;
  
  Res := cuMemAlloc(Result, ByteSize);
  if Res <> CUDA_SUCCESS then
    raise Exception.CreateFmt('cuMemAlloc failed: %s', [CudaResultToString(Res)]);
end;

procedure TCudaContext.FreeDevice(DevPtr: TCUdeviceptr);
begin
  if FInitialized and (DevPtr <> 0) then
    cuMemFree(DevPtr);
end;

procedure TCudaContext.CopyToDevice(DevPtr: TCUdeviceptr; HostPtr: Pointer; ByteSize: NativeUInt);
var
  Res: TCUresult;
begin
  if not FInitialized then Exit;
  
  Res := cuMemcpyHtoD(DevPtr, HostPtr, ByteSize);
  if Res <> CUDA_SUCCESS then
    raise Exception.CreateFmt('cuMemcpyHtoD failed: %s', [CudaResultToString(Res)]);
end;

procedure TCudaContext.CopyFromDevice(HostPtr: Pointer; DevPtr: TCUdeviceptr; ByteSize: NativeUInt);
var
  Res: TCUresult;
begin
  if not FInitialized then Exit;
  
  Res := cuMemcpyDtoH(HostPtr, DevPtr, ByteSize);
  if Res <> CUDA_SUCCESS then
    raise Exception.CreateFmt('cuMemcpyDtoH failed: %s', [CudaResultToString(Res)]);
end;

procedure TCudaContext.Synchronize;
begin
  if FInitialized and Assigned(cuCtxSynchronize) then
    cuCtxSynchronize;
end;

function TCudaContext.LoadModule(const PTXCode: AnsiString): TCUmodule;
var
  Res: TCUresult;
begin
  Result := nil;
  if not FInitialized then Exit;
  
  Res := cuModuleLoadData(Result, PAnsiChar(PTXCode));
  if Res <> CUDA_SUCCESS then
    raise Exception.CreateFmt('cuModuleLoadData failed: %s', [CudaResultToString(Res)]);
end;

procedure TCudaContext.UnloadModule(Module: TCUmodule);
begin
  if FInitialized and (Module <> nil) then
    cuModuleUnload(Module);
end;

function TCudaContext.GetFunction(Module: TCUmodule; const FuncName: AnsiString): TCUfunction;
var
  Res: TCUresult;
begin
  Result := nil;
  if not FInitialized or (Module = nil) then Exit;
  
  Res := cuModuleGetFunction(Result, Module, PAnsiChar(FuncName));
  if Res <> CUDA_SUCCESS then
    raise Exception.CreateFmt('cuModuleGetFunction failed for "%s": %s', 
      [FuncName, CudaResultToString(Res)]);
end;

procedure TCudaContext.LaunchKernel(Func: TCUfunction;
  GridDimX, GridDimY, GridDimZ: Cardinal;
  BlockDimX, BlockDimY, BlockDimZ: Cardinal;
  SharedMemBytes: Cardinal;
  const Args: array of Pointer);
var
  Res: TCUresult;
  ArgPtrs: array of Pointer;
  i: Integer;
begin
  if not FInitialized or (Func = nil) then Exit;
  
  // Build argument pointer array
  SetLength(ArgPtrs, Length(Args));
  for i := 0 to High(Args) do
    ArgPtrs[i] := Args[i];
  
  Res := cuLaunchKernel(Func,
    GridDimX, GridDimY, GridDimZ,
    BlockDimX, BlockDimY, BlockDimZ,
    SharedMemBytes,
    nil,  // Default stream
    @ArgPtrs[0],
    nil);  // No extra options
    
  if Res <> CUDA_SUCCESS then
    raise Exception.CreateFmt('cuLaunchKernel failed: %s', [CudaResultToString(Res)]);
end;

initialization

finalization
  UnloadCudaDriver;
  
end.
