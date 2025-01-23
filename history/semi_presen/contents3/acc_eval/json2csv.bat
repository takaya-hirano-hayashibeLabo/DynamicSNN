@echo off 

cd /d %~dp0
call C:\Users\3meko\miniconda3\Scripts\activate.bat
call conda activate mink-env

for /d %%D in ("%1\*") do (
    @REM echo %%D
    python json2csv.py --jsonpath %%D
)