import os
import time
import re
import sqlite3
import cv2
import numpy as np
import pytesseract

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

DB_NAME = "damacai.db"
URL = "https://www.damacai.com.my/past-draw-result"
DEBUG_DIR = "debug_ocr_roi"

ROI_TOP = {
    "1p3d_prizes": (0.03, 0.44, 0.47, 0.58),
    "3p3d_prizes": (0.52, 0.44, 0.92, 0.58),
}

ROI_BOTTOM = {
    "3d_prizes": (0.03, 0.02, 0.47, 0.14),
}

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS damacai_ocr (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            draw_date TEXT,
            draw_no TEXT,
            game TEXT,
            prize TEXT,
            number TEXT,
            UNIQUE(draw_date, draw_no, game, prize, number)
        )
    """)
    conn.commit()
    conn.close()

def save(draw_date, draw_no, game, prize, number):
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "INSERT OR IGNORE INTO damacai_ocr VALUES (NULL, ?, ?, ?, ?, ?)",
        (draw_date, draw_no, game, prize, number)
    )
    conn.commit()
    conn.close()

def ensure_debug_dir():
    os.makedirs(DEBUG_DIR, exist_ok=True)

def crop_roi(img_bgr, roi):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = roi
    xa = max(0, int(x1 * w))
    ya = max(0, int(y1 * h))
    xb = min(w, int(x2 * w))
    yb = min(h, int(y2 * h))
    if xb <= xa or yb <= ya:
        return None
    return img_bgr[ya:yb, xa:xb]

def preprocess_for_ocr(img_bgr, debug_name=None):
    img_bgr = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    if debug_name:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_name}_th.png"), th)

    return th

def ocr_digits(th_img):
    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(th_img, config=config)
    return re.findall(r"\d+", text)

def parse_1p3d(nums):
    nums4 = [n for n in nums if len(n) == 4]
    return nums4[-3:] if len(nums4) >= 3 else []

def parse_3d(nums):
    nums3 = [n for n in nums if len(n) == 3]
    return nums3[:3] if len(nums3) >= 3 else []

def parse_3p3d(nums):
    out = []
    i = 0
    while i < len(nums):
        a = nums[i]
        if len(a) == 6:
            out.append(a)
            i += 1
            continue
        if len(a) == 3 and i + 1 < len(nums) and len(nums[i + 1]) == 3:
            out.append(a + nums[i + 1])
            i += 2
            continue
        i += 1
    return out[:3] if len(out) >= 3 else []

def make_driver():
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1600,1200")
    # options.add_argument("--headless=new")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

def take_screenshot_cv(driver, name):
    png = driver.get_screenshot_as_png()
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}.png"), img)
    return img

def detect_draw_meta(driver):
    body_text = driver.find_element(By.TAG_NAME, "body").text

    draw_date = "UNKNOWN_DATE"
    draw_no = "UNKNOWN_DRAWNO"

    mdate = re.search(r"draw\s*date\s*:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})", body_text, re.IGNORECASE)
    if mdate:
        draw_date = mdate.group(1)

    mno = re.search(r"draw\s*no\.?\s*:?\s*([0-9]{3,5}/[0-9]{2})", body_text, re.IGNORECASE)
    if mno:
        draw_no = mno.group(1)

    return draw_date, draw_no

def scrape_one_draw_roi():
    ensure_debug_dir()
    driver = make_driver()

    try:
        driver.get(URL)
        time.sleep(10)

        draw_date, draw_no = detect_draw_meta(driver)
        print(f"\nDetected:\n draw_date\t{draw_date}\n draw_no\t{draw_no}\n")

        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        top = take_screenshot_cv(driver, "full_top")
        if top is None:
            print("top screenshot decode failed")
            return

        roi = crop_roi(top, ROI_TOP["1p3d_prizes"])
        if roi is None:
            print("roi_1p3d crop failed")
        else:
            cv2.imwrite(os.path.join(DEBUG_DIR, "roi_1p3d.png"), roi)
            th = preprocess_for_ocr(roi, debug_name="roi_1p3d")
            nums = ocr_digits(th)
            prizes = parse_1p3d(nums)
            if prizes:
                save(draw_date, draw_no, "1+3D", "1st", prizes[0])
                save(draw_date, draw_no, "1+3D", "2nd", prizes[1])
                save(draw_date, draw_no, "1+3D", "3rd", prizes[2])
                print(f"1+3D:\n 1st\t{prizes[0]}\n 2nd\t{prizes[1]}\n 3rd\t{prizes[2]}\n")
            else:
                print(f"1+3D OCR fail nums={nums}")

        roi = crop_roi(top, ROI_TOP["3p3d_prizes"])
        if roi is None:
            print("roi_3p3d crop failed")
        else:
            cv2.imwrite(os.path.join(DEBUG_DIR, "roi_3p3d.png"), roi)
            th = preprocess_for_ocr(roi, debug_name="roi_3p3d")
            nums = ocr_digits(th)
            prizes = parse_3p3d(nums)
            if prizes:
                save(draw_date, draw_no, "3+3D", "1st", prizes[0])
                save(draw_date, draw_no, "3+3D", "2nd", prizes[1])
                save(draw_date, draw_no, "3+3D", "3rd", prizes[2])
                print(f"3+3D:\n 1st\t{prizes[0]}\n 2nd\t{prizes[1]}\n 3rd\t{prizes[2]}\n")
            else:
                print(f"3+3D OCR fail nums={nums}")

        driver.execute_script("window.scrollTo(0, 1150);")
        time.sleep(2)

        bottom = take_screenshot_cv(driver, "full_bottom")
        if bottom is None:
            print("bottom screenshot decode failed")
            return

        roi = crop_roi(bottom, ROI_BOTTOM["3d_prizes"])
        if roi is None:
            print("roi_3d crop failed")
        else:
            cv2.imwrite(os.path.join(DEBUG_DIR, "roi_3d.png"), roi)
            th = preprocess_for_ocr(roi, debug_name="roi_3d")
            nums = ocr_digits(th)
            prizes = parse_3d(nums)
            if prizes:
                save(draw_date, draw_no, "3D", "1st", prizes[0])
                save(draw_date, draw_no, "3D", "2nd", prizes[1])
                save(draw_date, draw_no, "3D", "3rd", prizes[2])
                print(f"3D:\n 1st\t{prizes[0]}\n 2nd\t{prizes[1]}\n 3rd\t{prizes[2]}\n")
            else:
                print(f"3D OCR fail nums={nums}")


    finally:
        driver.quit()

if __name__ == "__main__":
    init_db()
    scrape_one_draw_roi()
