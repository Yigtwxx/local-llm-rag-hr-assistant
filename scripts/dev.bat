@echo off
rem ---------------------------------------------------------------------
rem  Starts the backend and the frontend together for local development.
rem
rem  Each server gets its own console window and keeps its full, unfiltered
rem  log there. Two consoles rather than one is deliberate: cmd.exe cannot
rem  merge two live output streams without swallowing one of them, and a
rem  window opened with `cmd /k` stays up after a crash — so a traceback is
rem  still on screen instead of vanishing with the process.
rem
rem  Usage:  scripts\dev.bat
rem          set BACKEND_PORT=8001 && scripts\dev.bat
rem ---------------------------------------------------------------------
setlocal

rem Resolve to a real absolute path; "%~dp0.." would otherwise carry the
rem trailing ".." into every message and into taskkill's filters.
for %%I in ("%~dp0..") do set "ROOT=%%~fI"

if not defined BACKEND_PORT set "BACKEND_PORT=8000"
if not defined FRONTEND_PORT set "FRONTEND_PORT=5173"

rem --- Preflight -------------------------------------------------------
rem Checked here rather than left to fail mid-startup, where the error
rem surfaces as a stack trace from whichever server lost the race.

where uv >nul 2>&1
if errorlevel 1 (
    echo hata: uv bulunamadi -- https://docs.astral.sh/uv/
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo hata: npm bulunamadi -- Node.js 20 veya uzeri gerekiyor
    exit /b 1
)

call :check_port %BACKEND_PORT% BACKEND_PORT || exit /b 1
call :check_port %FRONTEND_PORT% FRONTEND_PORT || exit /b 1

rem The vector store is built by `uv run python -m app.ingest`; without it
rem every question comes back as "bilmiyorum" and the cause is not visible
rem in the UI.
if not exist "%ROOT%\backend\storage\chroma.sqlite3" (
    echo hata: indeks yok -- once backend klasorunde: uv run python -m app.ingest
    exit /b 1
)

rem Ollama being down is recoverable while the app runs, so it is a warning.
where curl >nul 2>&1
if not errorlevel 1 (
    curl -fsS --max-time 2 http://127.0.0.1:11434 >nul 2>&1
    if errorlevel 1 echo uyari: Ollama 11434 portunda yanit vermiyor -- "ollama serve" calistirin
)

if not exist "%ROOT%\frontend\node_modules" (
    echo -^> frontend bagimliliklari kuruluyor...
    pushd "%ROOT%\frontend"
    rem `call` is required: npm is itself a batch file, and without it the
    rem shell never returns here.
    call npm install
    popd
)

rem --- Run -------------------------------------------------------------
rem Without this uvicorn buffers its log lines and they arrive in batches,
rem long after the request that produced them.
set "PYTHONUNBUFFERED=1"
set "FORCE_COLOR=1"

rem /D sets each window's working directory, so no nested quoting is needed
rem inside the command itself.
start "RAG backend" /D "%ROOT%\backend" cmd /k uv run uvicorn app.api:app --reload --port %BACKEND_PORT%
start "RAG frontend" /D "%ROOT%\frontend" cmd /k npm run dev -- --port %FRONTEND_PORT%

rem Opens the UI itself rather than leaving it to a clicked link. A browser
rem that is not already running restores its own session or start page first,
rem so the link lands behind whatever was open last. Chrome's `--app` window
rem has no session restore, no start page and no tab strip.
if not defined NO_OPEN (
    call :wait_for_ui
    where chrome >nul 2>&1
    if errorlevel 1 (
        start "" "http://localhost:%FRONTEND_PORT%"
    ) else (
        start "" chrome --app="http://localhost:%FRONTEND_PORT%"
    )
)

echo.
echo   backend   http://127.0.0.1:%BACKEND_PORT%   (dokumantasyon: /docs)
echo   arayuz    http://localhost:%FRONTEND_PORT%
echo.
echo   Loglar acilan iki pencerede akiyor. Durdurmak icin bu pencerede
echo   bir tusa basin -- iki sunucu da kapatilir.
echo.
pause >nul

rem Killed by listening port rather than by window title: `cmd /k` rewrites
rem its own title to the running command, so a title filter misses them.
call :kill_port %BACKEND_PORT%
call :kill_port %FRONTEND_PORT%
echo Sunucular durduruldu.

endlocal
exit /b 0

rem --- Helpers ---------------------------------------------------------

:check_port
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /r /c:":%~1 .*LISTENING"') do (
    echo hata: %~1 portu dolu -- %~2 degiskeniyle degistirebilirsiniz
    exit /b 1
)
exit /b 0

:kill_port
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /r /c:":%~1 .*LISTENING"') do (
    taskkill /PID %%P /T /F >nul 2>&1
)
exit /b 0

rem Waits for a real response instead of a fixed sleep -- opening early shows
rem a connection error page that does not refresh itself.
:wait_for_ui
where curl >nul 2>&1
if errorlevel 1 (
    timeout /t 3 /nobreak >nul
    exit /b 0
)
for /l %%N in (1,1,30) do (
    curl -fsS --max-time 1 "http://localhost:%FRONTEND_PORT%" >nul 2>&1
    if not errorlevel 1 exit /b 0
    timeout /t 1 /nobreak >nul
)
exit /b 0
