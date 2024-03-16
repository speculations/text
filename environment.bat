@ECHO OFF
REM Virtual Environment

SET prefix=J:\programs\anaconda3\envs\transcribe
start "BUILD" /B /wait conda env create -f environment.yml -p %prefix%
exit