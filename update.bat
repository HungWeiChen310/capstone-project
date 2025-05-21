chcp 65001
@echo off
cd /d "%~dp0"
echo -----------------------------------
echo 自動更新 Git 主源代碼
echo 目前資料夾：%cd%
echo -----------------------------------

where git >nul 2>nul
if errorlevel 1 (
    echo [錯誤] 未偵測到 git，請先安裝 Git 並設定環境變數。
    pause
    exit /b 1
)

REM 顯示目前 git 狀態
echo.
git status

REM 嘗試從主遠端更新 main 分支
echo.
echo [步驟] 開始同步 origin/main...
git pull origin main

if errorlevel 1 (
    echo.
    echo [錯誤] 無法從 main 同步，請檢查遠端名稱與分支是否正確。
    pause
    exit /b 1
)

echo.
echo 同步完成！
git log -1

pause
exit /b 0
