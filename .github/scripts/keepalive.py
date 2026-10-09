"""Streamlit Community Cloud 앱이 잠들지 않도록 실제 브라우저로 방문하고, 잠들어 있으면 깨우기 버튼을 누릅니다.
GitHub Actions(.github/workflows/keepalive.yml)에서 주기적으로 실행됩니다."""
import os
import sys

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://jinwoo-apt.streamlit.app/")
WAKE_BUTTON = "Yes, get this app back up"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(8_000)

    button = page.get_by_role("button", name=WAKE_BUTTON)
    if button.count() > 0:
        print("앱이 잠들어 있음 → 깨우기 버튼 클릭")
        button.first.click()
        page.wait_for_timeout(90_000)   # 깨어나는 데 보통 30초~1분
        if page.get_by_role("button", name=WAKE_BUTTON).count() > 0:
            print("깨우기 실패: 버튼이 아직 남아 있음")
            browser.close()
            sys.exit(1)
        print("깨우기 완료")
    else:
        print("앱이 깨어 있음 → 접속 유지")
        page.wait_for_timeout(20_000)   # 접속으로 인정되도록 잠시 머무름

    browser.close()
