@echo off
setlocal
cd /d "%~dp0"
python "%~dp0run_equilibrium_tensile_all_cases.py"
pause
