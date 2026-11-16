
@echo off
title Smart Waste Classification System

cd /d "C:\Users\kaksh\Smart-Waste-Classification"

if not exist "venv311\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found!
    pause
    exit /b
)

echo Starting Smart Waste Classification...

"venv311\Scripts\python.exe" -m streamlit run app.py

pause
