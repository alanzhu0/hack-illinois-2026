import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from random import random

# ---- CONFIG ----
URL = "https://app.polysights.xyz/insider-finder"
OUTPUT_CSV = "insider_data.csv"
WAIT_TIME = 30  # seconds
MARKET_INDEX = 3
MAX_PAGES = 805
MAX_WORKERS = 8


def create_driver():
    options = webdriver.EdgeOptions()
    time.sleep(random() * 60)  # random delay to reduce load 
    return webdriver.ChromiumEdge(options)


def setup_table_view(driver):
    wait = WebDriverWait(driver, WAIT_TIME)

    driver.get(URL)
    time.sleep(10)

    try:
        toggle = wait.until(EC.element_to_be_clickable((By.ID, "hide-closed-markets")))
        toggle.click()
        time.sleep(1)
    except Exception as error:
        print("Toggle not found or already off:", error)

    try:
        table_view_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Table View')]"))
        )
        table_view_btn.click()
        time.sleep(3)
    except Exception as error:
        print("Table view button not found:", error)


def scrape_current_page(driver):
    wait = WebDriverWait(driver, WAIT_TIME)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))

    headers = driver.find_elements(By.CSS_SELECTOR, "table thead th")
    header_text = [header.text.strip() for header in headers] + ["Market Link"]

    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    page_data = []
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        row_values = [cell.text.strip() for cell in cells]

        market_cell = cells[MARKET_INDEX]
        try:
            market_link = market_cell.find_element(By.TAG_NAME, "a").get_attribute("href")
        except Exception:
            market_link = ""

        row_values.append(market_link)
        page_data.append(row_values)

    return header_text, page_data


def go_to_page(driver, target_page):
    print(f"Navigating to page {target_page}...")
    wait = WebDriverWait(driver, WAIT_TIME)
    current_page = 1

    while current_page < target_page:
        next_btn = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "svg.lucide-chevron-right"))
        )
        next_btn = next_btn.find_element(By.XPATH, "./ancestor::button")

        if not next_btn.is_enabled():
            return False

        next_btn.click()
        current_page += 1

    return True


def split_page_ranges(max_pages, worker_count):
    if max_pages <= 0 or worker_count <= 0:
        return []

    worker_count = min(worker_count, max_pages)
    base_size = max_pages // worker_count
    extra = max_pages % worker_count

    ranges = []
    start = 1
    for worker_index in range(worker_count):
        size = base_size + (1 if worker_index < extra else 0)
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1

    return ranges


def scrape_page_range(start_page, end_page):
    driver = create_driver()
    try:
        print(f"Processing pages {start_page}-{end_page}...")
        setup_table_view(driver)

        if not go_to_page(driver, start_page):
            return None, {}

        page_results = {}
        headers = None

        for page_number in range(start_page, end_page + 1):
            print(f"Scraping page {page_number}...")
            page_headers, page_data = scrape_current_page(driver)
            page_results[page_number] = page_data
            if headers is None and page_headers is not None:
                headers = page_headers

            if page_number < end_page:
                wait = WebDriverWait(driver, WAIT_TIME)
                next_btn = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "svg.lucide-chevron-right"))
                )
                next_btn = next_btn.find_element(By.XPATH, "./ancestor::button")

                if not next_btn.is_enabled():
                    break

                next_btn.click()

        return headers, page_results
    finally:
        driver.quit()


def main():
    pages_to_fetch = list(range(1, MAX_PAGES + 1))
    worker_count = min(MAX_WORKERS, len(pages_to_fetch))
    page_ranges = split_page_ranges(MAX_PAGES, worker_count)

    page_results = {}
    headers = None

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(scrape_page_range, start_page, end_page): (start_page, end_page)
            for start_page, end_page in page_ranges
        }
        for future in as_completed(futures):
            start_page, end_page = futures[future]
            try:
                page_headers, range_results = future.result()
                page_results.update(range_results)
                if headers is None and page_headers is not None:
                    headers = page_headers
            except Exception as error:
                print(f"Failed to scrape pages {start_page}-{end_page}: {error}")
                for page in range(start_page, end_page + 1):
                    page_results[page] = []

    collected_data = []
    for page in sorted(page_results):
        collected_data.extend(page_results[page])

    if collected_data and headers:
        df = pd.DataFrame(collected_data, columns=headers)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    else:
        print("No data found!")


if __name__ == "__main__":
    main()