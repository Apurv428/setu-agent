@echo off
for /f "tokens=2 delims==" %%a in ('findstr /i "SARVAM_API_KEY=" "%~dp0.env"') do set "SARVAM_API_KEY=%%a"
"%~dp0.venv\Scripts\python.exe" "%~dp0mcp_server.py"
