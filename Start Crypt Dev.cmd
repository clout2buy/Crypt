@echo off
setlocal
cd /d "%~dp0"
npm --prefix desktop run electron:dev
