@echo off
setlocal

:: Verifica se as bibliotecas estão instaladas
set REQUISITOS=numpy matplotlib plotly pandas ipywidgets scipy ipython xlsxwriter dash openpyxl

for %%R in (%REQUISITOS%) do (
    pip show %%R >nul 2>&1
    if errorlevel 1 (
        echo Instalando %%R...
        pip install %%R
    )
)

:: Instala os requisitos do arquivo requirements.txt
pip install -r requirements.txt

