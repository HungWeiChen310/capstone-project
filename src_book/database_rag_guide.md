# 資料表查詢指引（RAG）

本文件用來讓 RAG/LLM 判斷「要查哪一張統計表」。

## 月度「異常總情況」要用哪張表？

- `stats_operational_monthly`：每台設備 *每月* 的總體營運與停機(異常)彙總。適用於「異常總情況」「停機總秒數」「停機率」等「總覽/總和」問題。
- `stats_abnormal_monthly`：每台設備 *每月*、*每個 detected_anomaly_type* 的異常分項統計。適用於「各類異常」「異常類型分佈」「哪一種異常最多」等需要依異常類型拆分的問題。

因此，使用者問：
「我想要查詢 eq001 在 2025 年 1 月的異常總情況」
→ 優先使用 `stats_operational_monthly`（條件：`equipment_id`、`year`、`month`），取得 `downtime_sec` / `downtime_rate_percent` 等總覽數據。

若使用者問：
「eq001 在 2025 年 1 月各異常類型的停機秒數/占比」
→ 使用 `stats_abnormal_monthly`（條件：`equipment_id`、`year`、`month`、`detected_anomaly_type`），列出或彙整各異常類型的數據。

## 季度/年度對應

- `stats_operational_quarterly` / `stats_operational_yearly`：總覽（總體異常/停機彙總）
- `stats_abnormal_quarterly` / `stats_abnormal_yearly`：分項（依 `detected_anomaly_type` 拆分）
