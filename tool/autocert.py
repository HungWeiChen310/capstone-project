#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # capstone-project 根目錄
    project_root = Path(__file__).resolve().parent.parent

    # wacs.exe 的位置（專案內的 win-acme 資料夾）
    WACS = project_root / "win-acme" / "wacs.exe"

    # 憑證輸出資料夾
    cert_print = project_root / "certs"
    cert_print.mkdir(parents=True, exist_ok=True)

    # 從環境變數讀 Cloudflare API Token
    SSL_API = os.environ.get("SSL_API")
    if SSL_API is None or not SSL_API.strip():
        raise RuntimeError("請設定 SSL_API 環境變數")

    # 共用的 wacs 參數
    wacs_args = [
        "--source", "manual",
        "--host", "capstone-project.me",
        "--validation", "cloudflare",
        "--cloudflareapitoken", SSL_API,
        "--store", "pemfiles",
        "--pemfilespath", str(cert_print),
        "--emailaddress", "evan060893@gmail.com",
        "--accepttos",
        "--verbose",
        "--closeonfinish",
    ]

    # 根據平台決定如何呼叫 wacs
    if os.name == "nt":
        # Windows：直接跑 wacs.exe
        cmd = [str(WACS), *wacs_args]
    else:
        # Linux / macOS：透過 wine 跑 wacs.exe
        # 如果沒有裝 wine，這裡會丟 FileNotFoundError
        cmd = ["wine", str(WACS), *wacs_args]

    print("執行指令：", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            errors="ignore",
            encoding="utf-8", 
            cwd=str(project_root),  # 讓 wacs 的工作目錄在專案根
        )
    except FileNotFoundError as e:
        # 通常是找不到 wine 或 wacs.exe
        print("執行失敗：找不到可執行檔。")
        print("錯誤訊息：", e)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("執行逾時（超過 600 秒）。")
        sys.exit(1)

    print("Exit code:", proc.returncode)
    print("STDOUT:\n", proc.stdout)
    print("STDERR:\n", proc.stderr)

    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
