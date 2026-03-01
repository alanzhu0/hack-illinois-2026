import time
import csv
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ---- CONFIG ----
URL = "https://app.polysights.xyz/insider-finder"
OUTPUT_CSV = "insider_data.csv"
WAIT_TIME = 10  # seconds
MARKET_INDEX = 3

# ---- SETUP BROWSER ----
options = webdriver.EdgeOptions()
# options.add_argument("--start-maximized")
driver = webdriver.ChromiumEdge(options)
wait = WebDriverWait(driver, WAIT_TIME)

driver.get(URL)
time.sleep(5)  # initial page load

# ---- TURN OFF "HIDE CLOSED MARKETS" ----
try:
    toggle = wait.until(
        EC.element_to_be_clickable((By.ID, "hide-closed-markets"))
    )
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



collected_data = []

# ---- PAGINATION LOOP ----
page = 1
while True:
    # wait for table to render
    print("Processing page...", page)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))

    # read all rows from the table
    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        collected_data.append([cell.text.strip() for cell in cells])

        market_cell = cells[MARKET_INDEX]
        try:
            market_link = market_cell.find_element(By.TAG_NAME, "a").get_attribute("href")
        except:
            market_link = ""
        collected_data[-1].append(market_link)  # append link to last row

    # check for next page button
    try:
        next_btn = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "svg.lucide-chevron-right"))
        )

        next_btn = next_btn.find_element(By.XPATH, "./ancestor::button")

        if not next_btn.is_enabled():
            break

        next_btn.click()
        page += 1

    except Exception as e:
        print("Error finding or clicking next page button:", e)
        break

    if page >= 4:
        break

# ---- SAVE TO CSV ----
if collected_data:
    # generate headers from the first page
    headers = driver.find_elements(By.CSS_SELECTOR, "table thead th")
    header_text = [h.text.strip() for h in headers] + ["Market Link"]  # add extra column for link

    df = pd.DataFrame(collected_data, columns=header_text)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
else:
    print("No data found!")

driver.quit()