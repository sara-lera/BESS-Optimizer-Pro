@echo off
title Eco-Optimizer BESS Dashboard
color 0B
cd /d "%~dp0"

echo ========================================================
echo       Eco-Optimizer BESS - Panel de Predicciones
echo ========================================================
echo.
echo Iniciando el dashboard interactivo...
echo.

set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
python -m streamlit run dashboard.py

pause
