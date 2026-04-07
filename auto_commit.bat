@echo off
cd /d C:\Graph_Brain
git add -A >nul 2>&1
for /f "delims=" %%i in ('git status --porcelain') do (
    git commit -m "Auto-backup %date% %time%" >nul 2>&1
    goto :done
)
:done
