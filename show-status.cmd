@echo off
cd /d "%~dp0"
"%~dp0.venv310\Scripts\python.exe" "%~dp0scripts\update_status.py" 
if not "%SEG_NO_PAUSE%"=="1" pause
