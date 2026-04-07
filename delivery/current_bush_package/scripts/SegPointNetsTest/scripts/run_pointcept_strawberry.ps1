param(
    [string]$Image = "pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel",
    [string]$Config = "/workspace/configs/strawberry/pointcept_ptv3_strawberry.py",
    [string]$SavePath = "/workspace/runs/pointcept_strawberry",
    [string]$Weight = "",
    [switch]$TestOnly
)

$workspace = "D:\MyProjects\Robostrawberry\SegPointNetsTest"
$mount = "${workspace}:/workspace"

$command = @"
cd /workspace/official/Pointcept
python tools/train.py --config-file $Config --num-gpus 1 --options save_path=$SavePath test_only=$($TestOnly.IsPresent.ToString().ToLower()) $(if ($Weight -ne "") { "weight=$Weight" } else { "" })
"@

docker run --rm --gpus all -v $mount -w /workspace/official/Pointcept $Image bash -lc $command
