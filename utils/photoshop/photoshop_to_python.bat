@echo off
echo %DOC_NAME%

C:/Programming/Bubbles/venv/Scripts/python.exe C:/Programming/Bubbles/utils/photoshop/photoshop_postprocessing.py %DOC_NAME%
REN set /p trash=Enter any input to end. 