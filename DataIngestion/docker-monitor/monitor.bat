@echo off
REM Batch file wrapper for Windows command line users
REM This simply passes all arguments to the Bun implementation

bun run start-monitor.js %* 