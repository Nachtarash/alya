@echo off
title Alya Build
color a
echo =========================================
echo --- Build Ai.exe with CMake + Ninja   ---
echo =========================================
echo.

if not defined VSCMD_ARG_TGT_ARCH (
    :: call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

if not exist build mkdir build
cd build

echo --- CMake configure...
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100
if errorlevel 1 goto error

echo --- CMake build...
cmake --build .
if errorlevel 1 goto error

echo --- Build successful!
echo.
cd ..
goto end

:error
echo --- Build failed ---
cd ..

:end
echo.
title finish!
echo ========================================
echo --- Finish!!! ---
echo ========================================
pause