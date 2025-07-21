@echo off
setlocal enabledelayedexpansion

:: === CONFIG ===
set JAVA_HOME=C:\Program Files\Java\jdk-19
set JNI_INC="%JAVA_HOME%\include"
set JNI_WIN="%JAVA_HOME%\include\win32"

:: === CLEANUP ===
echo Cleaning up old build files...
del /q *.class *.dll *.obj *.lib *.exp >nul 2>&1

:: === STEP 1: Compile Java and generate JNI header ===
echo Compiling Java...
javac -h . CUDASolution.java

:: === STEP 2: Compile with NVCC ===
echo Compiling and linking native code with NVCC...
nvcc -I"%JAVA_HOME%\include" -I"%JAVA_HOME%\include\win32" -Xcompiler "/MD" -shared -o cudaHasher.dll gpu_cracker.cpp sha256_kernel.cu md5_kernel.cu


:: === DONE ===
echo.
echo Build complete! Output: cudaHasher.dll
echo.
pause
