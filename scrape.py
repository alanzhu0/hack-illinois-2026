import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---- CONFIG ----
URL = "https://app.polysights.xyz/insider-finder"
OUTPUT_CSV = "insider_data.csv"
WAIT_TIME = 10  # seconds
MARKET_INDEX = 3

# ---- SETUP BROWSER ----
options = webdriver.FirefoxOptions()
driver = webdriver.Firefox(options=options)
wait = WebDriverWait(driver, WAIT_TIME)

driver.get(URL)
time.sleep(5)  # initial page load

# ---- TURN OFF "HIDE CLOSED MARKETS" ----
try:
    toggle = wait.until(EC.element_to_be_clickable((By.ID, "hide-closed-markets")))
    # if toggle.is_selected():
    toggle.click()
    time.sleep(1)
except Exception as e:
    print("Toggle not found or already off:", e)

# Switch to table view
try:
    table_view_btn = wait.until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Table View')]"))
    )
    table_view_btn.click()
    time.sleep(3)  # wait for table to load
except Exception as e:
    print("Table view button not found:", e)


header_written = False
total_rows = 0
csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig")
writer = csv.writer(csv_file)

# ---- PAGINATION LOOP ----
page = 1
while True:
    # wait for table to render
    print("Processing page...", page)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))

    # read all rows from the table (bulk extraction in browser for speed)
    page_rows = driver.execute_script(
        """
        const marketIndex = arguments[0];
        const rows = Array.from(document.querySelectorAll('table tbody tr'));

        return rows.map((row) => {
            const cells = Array.from(row.querySelectorAll('td'));
            const rowData = cells.map((cell) => (cell.innerText || '').trim());

            let marketLink = '';
            if (cells.length > marketIndex) {
                const anchor = cells[marketIndex].querySelector('a[href]');
                marketLink = anchor ? anchor.href : '';
            }

            rowData.push(marketLink);
            return rowData;
        });
        """,
        MARKET_INDEX,
    )

    # ---- SAVE TO CSV ----
    if page_rows:
        if not header_written:
            headers = driver.find_elements(By.CSS_SELECTOR, "table thead th")
            header_text = [h.text.strip() for h in headers] + ["Market Link"]
            writer.writerow(header_text)
            header_written = True

        writer.writerows(page_rows)
        total_rows += len(page_rows)
        print(f"Saved {total_rows} rows to {OUTPUT_CSV}")
    else:
        print("No data found on this page!")

    # check for next page button
    try:
        next_btn = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "svg.lucide-chevron-right")
            )
        )

        next_btn = next_btn.find_element(By.XPATH, "./ancestor::button")

        if not next_btn.is_enabled():
            break

        next_btn.click()
        page += 1

    except Exception as e:
        print("Error finding or clicking next page button:", e)
        break

csv_file.close()

# ---- SAVE TO CSV ----
if total_rows:
    print(f"Saved {total_rows} rows to {OUTPUT_CSV}")
else:
    print("No data found!")

driver.quit()
