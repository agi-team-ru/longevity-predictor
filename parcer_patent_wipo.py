from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

def wipo_patents_selenium_search(query, max_pages=35, output='wipo_patents.json', headless=True, date_from=None, date_to=None):
    options = Options()
    if headless:
        options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    base_url = "https://patentscope.wipo.int/search/en/"
    driver.get(base_url)

    wait = WebDriverWait(driver, 20)
    # Use the current id of the search field
    try:
        search_box = wait.until(EC.presence_of_element_located((By.ID, "simpleSearchForm:fpSearch:input")))
    except:
        print("ID simpleSearchForm:fpSearch:input not found. Please check the id manually using F12 in your browser!")
        driver.save_screenshot("wipo_error.png")
        driver.quit()
        return

    # Build the search query with date filter if specified
    if date_from and date_to:
        search_query = f"{query} AND DP:[{date_from} TO {date_to}]"
    else:
        search_query = query

    search_box.clear()
    search_box.send_keys(search_query)
    # Click the search button (magnifier) by class, not by id
    search_btn = driver.find_element(By.CSS_SELECTOR, "button.primary.js-default-button")
    search_btn.click()
    # Wait for at least one patent result to appear
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".ps-patent-result")))
    except:
        print("No patent results found! Please check the page structure.")
        driver.save_screenshot("wipo_error.png")
        driver.quit()
        return
    time.sleep(2)
    with open("wipo_results.html", "w", encoding="utf-8") as f:
       f.write(driver.page_source)

    results = []
    for page in range(max_pages):
        time.sleep(2)
        patents = driver.find_elements(By.CSS_SELECTOR, ".ps-patent-result")
        for patent in patents:
            try:
                # Title and link
                title_div = patent.find_element(By.CSS_SELECTOR, ".ps-patent-result--title")
                link_tag = title_div.find_element(By.TAG_NAME, "a")
                link = link_tag.get_attribute("href")
                if link and not link.startswith("http"):
                    link = base_url + link.lstrip("/")
                # Patent title
                title_span = title_div.find_element(By.CSS_SELECTOR, ".ps-patent-result--title--title")
                title = title_span.text.strip()
                # Patent code (publication number)
                pubnum_span = title_div.find_element(By.CSS_SELECTOR, ".ps-patent-result--title--patent-number")
                publication_number = pubnum_span.text.strip()
            except Exception as e:
                title = None
                link = None
                publication_number = None
            try:
                # Publication date
                date_div = patent.find_element(By.CSS_SELECTOR, ".ps-patent-result--title--ctr-pubdate")
                date = date_div.find_elements(By.TAG_NAME, "span")[-1].text.strip()
            except:
                date = None
            try:
                # Abstract
                abstract_div = patent.find_element(By.CSS_SELECTOR, ".ps-patent-result--abstract")
                abstract = abstract_div.text.strip()
            except:
                abstract = None
            try:
                # Applicant (affiliation)
                applicant_span = patent.find_element(By.CSS_SELECTOR, ".ps-patent-result--applicant")
                applicant = applicant_span.text.strip()
            except:
                applicant = None
            results.append({
                "id": publication_number,
                "title": title,
                "abstract": abstract,
                "date": date,
                "affiliation": applicant,
                "url": link
            })
        print(f"Page {page+1}: {len(results)} patents found")
        # Go to the next page
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "a.js-paginator-next")
            driver.execute_script("arguments[0].scrollIntoView();", next_btn)
            if "ui-state-disabled" in next_btn.get_attribute("class"):
                break
            next_btn.click()
            wait.until(EC.staleness_of(patents[0]))
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".ps-patent-result")))
        except Exception as e:
            print(f"Could not go to the next page: {e}")
            break

    driver.quit()
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} patents to {output}")

if __name__ == "__main__":
    query = '''(
      aging OR longevity OR senescence
    )
    AND
    (
      "cellular senescence" OR "DNA damage" OR "epigenesis" OR
      mitochondria OR "aging physiology" OR "biological clock*"
    )
    NOT
    (
      geriatr* OR sociolog* OR "resource allocation" OR
      "health policy" OR "health economics" OR "insurance coverage"
    )
    AND
    DP:[20210101 TO 20251231]'''.replace('\n', ' ')
    wipo_patents_selenium_search(query, max_pages=35, output='wipo_patents.json', headless=False)