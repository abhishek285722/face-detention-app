#!/usr/bin/env bash

echo "======================================================"
echo "   Real-Time Face Detection and Feature Tracking"
echo "======================================================"
echo ""

# Fast dependency check
python3 -c "import cv2, numpy" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[Checking and installing dependencies...]"
    python3 -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies."
        exit 1
    fi
    echo ""
fi

while true; do
    echo ""
    echo "======================================================"
    echo "Please select an option:"
    echo "  [1] Live Webcam Detection (Face, Eyes, Smile)"
    echo "  [2] Test on Sample Image (Generates detection_result.jpg)"
    echo "  [3] Run Automated Test Suite"
    echo "  [4] Exit"
    echo "======================================================"
    read -p "Enter choice [1-4]: " choice

    if [ "$choice" == "2" ]; then
        echo ""
        echo "------------------------------------------------------"
        echo "[Running face detection on sample_face.jpg...]"
        echo "------------------------------------------------------"
        python3 face_detection.py image sample_face.jpg --detector dnn --eyes --smiles --output detection_result.jpg
        echo ""
        echo "------------------------------------------------------"
        echo "[Completed] Output saved as detection_result.jpg"
        echo "------------------------------------------------------"
    elif [ "$choice" == "3" ]; then
        echo ""
        echo "------------------------------------------------------"
        echo "[Running automated unit tests...]"
        echo "------------------------------------------------------"
        python3 test_face_detection.py
        echo ""
        echo "------------------------------------------------------"
        echo "[Unit tests completed]"
        echo "------------------------------------------------------"
    elif [ "$choice" == "4" ] || [ "$choice" == "q" ]; then
        echo ""
        echo "Exiting..."
        exit 0
    else
        echo ""
        echo "------------------------------------------------------"
        echo "[Starting Live Webcam Detection...]"
        echo "Controls: 'q'=Quit, 's'=Snapshot, 'd'=Switch Detector, 'e'=Eyes, 'm'=Smiles"
        echo "------------------------------------------------------"
        python3 face_detection.py webcam --detector dnn --eyes --smiles
        echo ""
        echo "------------------------------------------------------"
        echo "[Live Webcam Detection session ended]"
        echo "------------------------------------------------------"
    fi
done
