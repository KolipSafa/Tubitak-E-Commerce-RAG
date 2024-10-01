from selenium import webdriver 
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.chrome.service import Service as ChromeService
from bs4 import BeautifulSoup
from time import sleep
from selenium.webdriver.common.by import By
import constants.laptop_constants as constants

def get_laptop_reviews(url,max_page_size):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install())) 
    driver.implicitly_wait(5)
    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")

    reviews = []

    # get first page reviews
    get_reviews_on_a_page(url,driver,reviews)

    page_numbers_div = soup.find('div', attrs={'class': constants.REVIEW_PAGE_NUMBERS_DIV})
    if page_numbers_div:
        page_count = int(page_numbers_div.contents[0].text.split(" ")[-1].strip()) 
        current_page_number = 2
        # get reviews of the remaining pages
        while current_page_number <= page_count:
            print(max_page_size)
            print(current_page_number)
            if max_page_size:
                if current_page_number > max_page_size:
                    break

            reviews_url = url + f"&page={current_page_number}"
            get_reviews_on_a_page(reviews_url ,driver, reviews)
            current_page_number  += 1

    driver.quit()
    return reviews

def get_reviews_on_a_page(url, webDriver, reviewArr, review_div_class = constants.REVIEW_PAGE_A_REVIEW_DIV):
    webDriver.get(url)
    # make hidden contents visible
    hidden_content_span_arr = webDriver.find_elements(By.CLASS_NAME, constants.HIDDEN_CONTENT_SPAN)
    for span in hidden_content_span_arr:
        span.click()

    sleep(10)

    updatedSoup = BeautifulSoup(webDriver.page_source, "html.parser")

    review_div_arr = updatedSoup.find_all('div', attrs={'class': review_div_class})

    for parent in review_div_arr:
        rating = parent.contents[0].contents[0].text
        title = parent.contents[0].contents[1].text

        content_parent_div = parent.contents[1].contents[0].contents[0]
        content = content_parent_div.contents[0].text

        last_part_div = parent.contents[-1] #row _3n8db9
        writtenBy = last_part_div.contents[0].contents[0].text

        like_container_div = last_part_div.contents[1].contents[0].contents[0] #_27aTsS
        numberOfLikes = like_container_div.contents[0].contents[1].text
        numberOfDislikes = like_container_div.contents[1].contents[1].text
        
        reviewArr.append({"rating": rating, "title": title, "content": content, 
                          "writtenBy": writtenBy, "numberOfLikes": numberOfLikes,
                          "numberOfDislikes":numberOfDislikes})
