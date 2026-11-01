@echo off
cd /d "%~dp0"
if exist "DSCMR\runs\pascal_selection5000_refit1000_seed42\result.json" goto clip4cmr
".venv310\Scripts\python.exe" scripts\train_selected.py --method DSCMR --max-epochs 5000 --refit-epochs 1000 --out DSCMR/runs/pascal_selection5000_refit1000_seed42
if errorlevel 1 exit /b 1
:clip4cmr
if exist "CLIP4CMR\runs\pascal_selection5000_refit1000_seed42\result.json" goto done
".venv310\Scripts\python.exe" scripts\train_selected.py --method CLIP4CMR --max-epochs 5000 --refit-epochs 1000 --data CLIP4CMR/data/pascal_uiuc_seed42 --out CLIP4CMR/runs/pascal_selection5000_refit1000_seed42
if errorlevel 1 exit /b 1
:done
".venv310\Scripts\python.exe" scripts\update_status.py
