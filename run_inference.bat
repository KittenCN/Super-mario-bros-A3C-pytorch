@echo off
echo 正在运行 Super Mario Bros A3C 模型推理...
echo 模型: trained_models\run01\a3c_world1_stage1_0008000.pt
echo.

REM 尝试激活conda环境
conda activate mario-a3c 2>nul
if %errorlevel% neq 0 (
    echo 警告: 未能激活mario-a3c环境，使用默认Python环境
    echo 请确保已安装所需依赖: pip install -r requirements.txt
    echo.
)

REM 运行快速推理
python quick_inference.py

echo.
echo 推理完成! 按任意键退出...
pause >nul