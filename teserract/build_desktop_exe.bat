@echo off
setlocal

if exist .venv\Scripts\python.exe (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo PyInstaller kuruluyor...
%PYTHON% -m pip install pyinstaller tkinterdnd2
if errorlevel 1 goto :error

echo Proje bagimliliklari kuruluyor...
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo Desktop EXE paketleniyor...
%PYTHON% -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name OCRDesktopStudio ^
  --paths "src" ^
  --hidden-import common ^
  --hidden-import pipelines ^
  --hidden-import pytesseract ^
  --hidden-import pandas ^
  --hidden-import pypdfium2 ^
  --hidden-import pypdfium2._helpers ^
  --hidden-import pypdfium2.raw ^
  --hidden-import rapidfuzz ^
  --hidden-import rapidfuzz.fuzz ^
  --hidden-import rapidfuzz.process ^
  --hidden-import openpyxl ^
  --hidden-import tkinterdnd2 ^
  --collect-all pypdfium2 ^
  --add-data "src;src" ^
  --add-data "tessdata;tessdata" ^
  --add-data "images;images" ^
  --add-data "templates;templates" ^
  desktop_app.py
if errorlevel 1 goto :error

echo.
echo Hazir: dist\OCRDesktopStudio.exe
goto :eof

:error
echo Paketleme sirasinda hata olustu.
exit /b 1
