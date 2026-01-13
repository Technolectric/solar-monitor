cd /d C:\solar-monitor

REM 1. Create requirements.txt
echo Flask==3.1.2 > requirements.txt
echo requests==2.32.5 >> requirements.txt
echo numpy==2.2.6 >> requirements.txt
echo gunicorn==21.2.0 >> requirements.txt

REM 2. Create Procfile
echo web: gunicorn claude_fixed:app > Procfile

REM 3. Create runtime.txt
echo python-3.10.13 > runtime.txt

REM 4. Create .gitignore
echo __pycache__/ > .gitignore
echo *.pyc >> .gitignore
echo .env >> .gitignore

REM 5. Create railway.json
echo { > railway.json
echo   "build": { >> railway.json
echo     "builder": "NIXPACKS" >> railway.json
echo   }, >> railway.json
echo   "deploy": { >> railway.json
echo     "startCommand": "gunicorn claude_fixed:app" >> railway.json
echo   } >> railway.json
echo } >> railway.json

REM 6. Create README.md
echo # Solar Monitor > README.md

REM 7. Check files
dir