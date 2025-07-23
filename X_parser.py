# pip install --upgrade snscrape requests-html selenium
# pip install undetected_chromedriver
# + должен быть установлен хром 
import importlib.machinery
import json
import logging
import time

if not hasattr(importlib.machinery.FileFinder, "find_module"):
    importlib.machinery.FileFinder.find_module = lambda *args, **kwargs: None

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logging.basicConfig(
    level=logging.INFO,
    format="[X PARSER] %(message)s"
)
logger = logging.getLogger("X PARSER")



def login_to_x(driver, username_or_email, password, phone_or_username=None):
    logger.info('открываем страница ввода логина')
    driver.get("https://x.com/login")

    user_input = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.NAME, "text"))
    )
    user_input.send_keys(username_or_email)
    logger.info('вводим почту')
    
    driver.find_element(By.XPATH, "//span[text()='Далее']").click()
    time.sleep(2)

    try:
        logger.info('проверка на доп проверку')
        phone_check = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.NAME, "text"))
        )
        if phone_check and phone_or_username:
            logger.info('доп проверка + логин')
            phone_check.send_keys(phone_or_username)
            driver.find_element(By.XPATH, "//span[text()='Далее']").click()
            time.sleep(2)
    except:
        logger.info('доп проверки пока нет')

    pass_input = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.NAME, "password"))
    )
    pass_input.send_keys(password)
    logger.info('вводим пароль')

    driver.find_element(By.XPATH, "//span[text()='Войти']").click()
    logger.info('кнопка войти')
    time.sleep(5)


def scrape_x_hashtag(hashtag, scroll_count=3, username=None, password=None, phone_or_username=None):
    url = f"https://x.com/search?q=%23{hashtag}&src=typed_query&f=live"
    logger.info('хром стартует, будем парсить по хештегу {hashtag}')
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")

    driver = uc.Chrome(options=options)
    uc.Chrome.__del__ = lambda self: None  # фикс WinError 6

    tweets_data = []

    try:
        if username and password:
            login_to_x(driver, username, password, phone_or_username)
        logger.info(f'попытка логина, переходим на {url}')
        driver.get(url)

        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article div[data-testid='tweetText']"))
            )
            logger.info('первые твиты')
        except:
            logger.info('либо не прошли логин либюо нет твитов')
            return []

        for i in range(scroll_count):
            logger.info(f'прокрутка #{i+1}')

            articles = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
            logger.info(f"{len(articles)} твитов на экране")

            added = 0
            for art in articles:
                try:
                    text_el = art.find_element(By.CSS_SELECTOR, "div[data-testid='tweetText']")
                    text = text_el.text.strip()

                    time_el = art.find_element(By.TAG_NAME, "time")
                    tweet_date = time_el.get_attribute("datetime")  # ISO формат

                    # проверка не добавляли ли уже такой твит
                    if not any(t["text"] == text for t in tweets_data):
                        tweets_data.append({
                            "type": "твит",
                            "pubdate": tweet_date,
                            "hashtag": hashtag,
                            "text": text
                        })
                        added += 1
                except:
                    continue

            logger.info(f'новых твитов: {added}')
            logger.info(f'листаем дальше')

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(4)

        logger.info(f"парсинг завершен, всего твитов собрано: {len(tweets_data)}")
        return tweets_data

    finally:
        logger.info('закрываем браузер')
        driver.quit()


if __name__ == "__main__":

    hashtag = "longevity"
    X_USERNAME = 'X_USERNAME'
    X_PHONE_OR_USERNAME = "X_PHONE_OR_USERNAME"
    X_PASSWORD = 'X_PASSWORD' # если X попросит доп. проверку

    result = scrape_x_hashtag(
        hashtag,
        scroll_count=2,
        username=X_USERNAME,
        password=X_PASSWORD,
        phone_or_username=X_PHONE_OR_USERNAME
    )

    with open("tweets.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("=== РЕЗУЛЬТАТ ===")
    logger.info(f"найдено {len(result)} твитов. сохранено в tweets.json")

# примеры тегов
# "longevity"
# "DNA"
# biomarker