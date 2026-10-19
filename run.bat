@echo off
title Real-Time Face Detection and Feature Tracking

:: Initial header (shown once at launch)
echo ======================================================
echo    Real-Time Face Detection and Feature Tracking
echo ======================================================
echo.

:: Fast dependency check (runs once at launch)
python -c "import cv2, numpy" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [Checking and installing dependencies - first run only...]
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies. Please ensure Python and pip are installed.
        pause
        exit /b 1
    )
    echo.
)

:menu
echo.
echo ======================================================
echo Please select an option:
echo   [1] Live Webcam Detection (Face, Eyes, Smile)
echo   [2] Test on Sample Image (Generates detection_result.jpg)
echo   [3] Run Automated Test Suite
echo   [4] Exit
echo ======================================================
set "choice="
set /p "choice=Enter choice [1-4]: "

if "%choice%"=="1" goto opt1
if "%choice%"=="2" goto opt2
if "%choice%"=="3" goto opt3
if "%choice%"=="4" goto opt4
if /i "%choice%"=="q" goto opt4
echo [Invalid choice] Please enter 1, 2, 3, or 4.
goto menu

:opt1
echo.
echo ------------------------------------------------------
echo [Starting Live Webcam Detection...]
echo Controls: 'q'=Quit, 's'=Snapshot, 'd'=Switch Detector, 'e'=Eyes, 'm'=Smiles
echo ------------------------------------------------------
python face_detection.py webcam --detector dnn --eyes --smiles
echo.
echo ------------------------------------------------------
echo [Live Webcam Detection session ended]
echo ------------------------------------------------------
goto menu

:opt2
echo.
echo ------------------------------------------------------
echo [Running face detection on sample_face.jpg...]
echo ------------------------------------------------------
python face_detection.py image sample_face.jpg --detector dnn --eyes --smiles --output detection_result.jpg
echo.
echo ------------------------------------------------------
echo [Completed] Output saved as detection_result.jpg
echo ------------------------------------------------------
goto menu

:opt3
echo.
echo ------------------------------------------------------
echo [Running automated unit tests...]
echo ------------------------------------------------------
python test_face_detection.py
echo.
echo ------------------------------------------------------
echo [Unit tests completed]
echo ------------------------------------------------------
goto menu

:opt4
echo.
echo Exiting application...
exit /b 0
