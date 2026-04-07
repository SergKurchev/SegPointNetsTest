param(
    [string]$Image = "strawberry-oneformer3d:latest",
    [string]$Config = "/workspace/configs/strawberry/oneformer3d_strawberry.py",
    [string]$WorkDir = "/workspace/runs/oneformer3d_strawberry",
    [string]$Checkpoint = "",
    [switch]$TestOnly
)

$workspace = ".\SegPointNetsTest"
$mount = "${workspace}:/workspace"

if ($TestOnly -and $Checkpoint -eq "") {
    throw "Checkpoint is required for -TestOnly"
}

$command = if ($TestOnly) {
@"
cd /workspace/official/oneformer3d
python tools/test.py $Config $Checkpoint --work-dir $WorkDir
"@
} else {
@"
cd /workspace/official/oneformer3d
python tools/train.py $Config --work-dir $WorkDir
"@
}

docker run --rm --gpus all -v $mount -w /workspace/official/oneformer3d $Image bash -lc $command
