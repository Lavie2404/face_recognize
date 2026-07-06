@echo off
title Update ECC for Face Recognize Project
echo =========================================================
echo Tu dong cap nhat rule/workflow tu D:\ECC vao D:\Face Recognize
echo =========================================================

:: Chuyển vị trí làm việc sang thư mục project gốc ở ổ D
cd /d D:\Face Recognize

:: Gọi trực tiếp Git Bash để chạy file sh
:: Đường dẫn đã được trỏ chính xác về ổ D (/d/ECC/install.sh)
"C:\Program Files\Git\bin\bash.exe" /d/ECC/install.sh --target antigravity

echo.
echo [THONG BAO] Cap nhat hoan tat! Thu muc .agent da duoc lam moi.
pause