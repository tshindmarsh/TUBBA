{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 @echo off\
REM TUBBA installer for Windows\
echo \uc0\u55356 \u57263  Installing TUBBA...\
\
REM Check if conda exists\
where conda >nul 2>nul\
if errorlevel 1 (\
    echo \uc0\u10060  Conda not found. Please install Miniconda or Anaconda first.\
    exit /b 1\
)\
\
echo \uc0\u55357 \u56550  Creating conda environment "tubba"...\
call conda create -n tubba python=3.12 -y\
\
echo \uc0\u55357 \u56615  Activating environment...\
call conda activate tubba\
\
echo \uc0\u55357 \u56549  Installing dependencies...\
call conda install -c conda-forge pyqt matplotlib seaborn scikit-learn h5py joblib opencv numpy pandas xgboost tqdm pytables -y\
\
echo \uc0\u55357 \u56613  Installing PyTorch...\
set /p use_gpu=Do you have an NVIDIA GPU and want CUDA acceleration? (y/n): \
\
if /i "%use_gpu%"=="y" (\
    call conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y\
) else (\
    call conda install pytorch cpuonly -c pytorch -y\
)\
\
echo \uc0\u9989  Installation complete!\
echo To use TUBBA:\
echo    conda activate tubba\
echo    cd src\
echo    python TUBBA.py\
pause}