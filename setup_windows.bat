@echo off
setlocal EnableDelayedExpansion

echo === checking for winget ===
where winget >nul 2>&1 || (
    echo winget not found! please install app installer from the microsoft store.
    exit /b 1
)

echo === checking for python ===
where python >nul 2>&1 || (
    echo installing python...
    winget install -e --id Python.Python.3 || exit /b 1
)

echo === checking for openscad ===
where openscad >nul 2>&1 || (
    echo installing openscad...
    winget install -e --id OpenSCAD.OpenSCAD || exit /b 1
)

where openscad >nul 2>&1 || (
    if exist "%ProgramFiles%\OpenSCAD\openscad.exe" set "OSPATH=%ProgramFiles%\OpenSCAD"
    if exist "%ProgramFiles(x86)%\OpenSCAD\openscad.exe" set "OSPATH=%ProgramFiles(x86)%\OpenSCAD"

    if defined OSPATH (
        echo adding openscad to PATH...
        setx PATH "%PATH%;%OSPATH%"
        set "PATH=%PATH%;%OSPATH%"
    )
)

where python >nul 2>&1 || (
    for /f "delims=" %%P in ('dir /b /ad "%LocalAppData%\Programs\Python" 2^>nul') do (
        if exist "%LocalAppData%\Programs\Python\%%P\python.exe" set "PYPATH=%LocalAppData%\Programs\Python\%%P"
    )
    if defined PYPATH (
        echo adding python to PATH...
        setx PATH "%PATH%;%PYPATH%"
        set "PATH=%PATH%;%PYPATH%"
    )
)

echo === installing requirements ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo done! run 'python app.py' to get started with a11yshape.