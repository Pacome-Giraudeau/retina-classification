@echo off
echo ============================================================
echo Pipeline ML pour classification de maladies retiniennes
echo ============================================================

:menu
echo.
echo Commandes disponibles:
echo   1. setup      - Installer les dependances
echo   2. train      - Executer l'entrainement
echo   3. eval       - Executer l'evaluation
echo   4. full       - Pipeline complet (train + eval)
echo   5. clean      - Nettoyer les resultats
echo   6. exit       - Quitter
echo.

set /p choice="Choisissez une option (1-6): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto train
if "%choice%"=="3" goto eval
if "%choice%"=="4" goto full
if "%choice%"=="5" goto clean
if "%choice%"=="6" goto end
echo Option invalide
goto menu

:setup
echo.
echo Installation des dependances...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Erreur lors de l'installation
    pause
)
goto menu

:train
echo.
echo Demarrage de l'entrainement...
python scripts/train.py
if %errorlevel% neq 0 (
    echo Erreur pendant l'entrainement
    pause
)
goto menu

:eval
echo.
echo Demarrage de l'evaluation...
python scripts/evaluate.py
if %errorlevel% neq 0 (
    echo Erreur pendant l'evaluation
    pause
)
goto menu

:full
echo.
echo Pipeline complet...
call :train
call :eval
goto menu

:clean
echo.
echo Nettoyage des fichiers generes...
if exist models rmdir /s /q models
if exist results rmdir /s /q results
if exist logs rmdir /s /q logs
mkdir models
mkdir results
mkdir logs
echo Nettoyage termine!
goto menu

:end
echo.
echo Au revoir!
pause