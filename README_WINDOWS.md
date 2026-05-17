# Windows launch

Double-click:

```text
PLAY_WINDOWS.bat
```

What it does:

```text
1. Looks for Python 3.11+ using py/python/python3.
2. Creates a local environment in .alife_env/windows_venv if needed.
3. Installs requirements.txt only when it has changed.
4. Starts run.py.
```

You can also launch from PowerShell or Command Prompt:

```powershell
.\PLAY_WINDOWS.bat
```

With arguments:

```powershell
.\PLAY_WINDOWS.bat --seed 123 --width 320 --height 160
```

If Python is missing, install Python 3.11 or newer from:

```text
https://www.python.org/downloads/windows/
```

During installation, enable:

```text
Add python.exe to PATH
```

The Windows environment is stored under `.alife_env/`, which is already ignored by Git.
