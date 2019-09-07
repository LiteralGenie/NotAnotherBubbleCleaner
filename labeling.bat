@echo off
echo %DOC_NAME%

C:/Programming/Bubbles/venv/Scripts/python.exe C:/Programming/Bubbles/labeling_PS.py %DOC_NAME%
set /p trash=Enter any input to end. 