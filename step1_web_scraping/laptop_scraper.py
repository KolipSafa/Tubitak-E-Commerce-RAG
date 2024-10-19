import time
from selenium import webdriver 
from bs4 import BeautifulSoup
import os
from classes.db.LaptopSQLiteDb import LaptopSQLiteDb
from classes.WebDriverThread import WebDriverThread
from dotenv import load_dotenv

from classes.laptop import Laptop
from helpers.get_laptop_reviews import get_laptop_reviews
import constants.laptop_constants as constants

MAX_LAPTOP_COUNT = 3
MAX_LAPTOP_REVIEW_PAGE_COUNT = 1 # 10 reviews per one page
CURR_PAGE_NUMBER = 1 

driver = webdriver.Chrome()
driver.implicitly_wait(5)
load_dotenv()

all_laptop_urls = []

# Getting All Laptop Urls
while len(all_laptop_urls) < MAX_LAPTOP_COUNT:
    driver.get(f"https://www.flipkart.com/laptops/pr?sid=6bo%2Cb5g&sort=popularity&page={CURR_PAGE_NUMBER}")
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")

    all_laptop_a = soup.findAll('a', href=True, attrs={'class': constants.ALL_LAPTOPS_A}, limit=(MAX_LAPTOP_COUNT - len(all_laptop_urls)))

    for a in all_laptop_a:
        url_without_query = a['href'].split('?')[0]
        all_laptop_urls.append("https://www.flipkart.com" +  url_without_query)
    
    if len(all_laptop_urls) >= MAX_LAPTOP_COUNT:
        break

    CURR_PAGE_NUMBER += 1
    time.sleep(2) 

# Getting Each Laptop Detail
laptop_array : list[Laptop]  = []

def get_laptop_specifications(tbody, spec_keys):
    specs = {}
    rows = tbody.find_all('tr', class_='WJdYP6 row')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True)
            if key in spec_keys:
                value = cells[1].get_text(strip=True)
                specs[key] = value
                if len(specs) == len(spec_keys):
                    break
    return specs

def save_laptop_as_markdown(base_directory, laptop: Laptop):
    laptop_directory = os.path.join(base_directory, laptop.id)
    os.makedirs(laptop_directory, exist_ok=True)

    # write features
    features_filename = os.path.join(laptop_directory, "features.md")
    with open(features_filename, 'w') as file:
        file.write(laptop.features_to_md_text())

    # write reviews
    for index, review in enumerate(laptop.reviews,start=1):
        review_filename = os.path.join(laptop_directory, f"review_{index}.md")
        with open(review_filename , 'w',encoding='utf-8') as file:
            file.write(laptop.review_to_md_text(review))

laptop_db = LaptopSQLiteDb(str(os.getenv('LAPTOP_DB_PATH')))

# Laptops that will be being scrapped
for url in all_laptop_urls:
    driver.get(url)
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")

    # Get reviews
    reviews = []
    all_reviews_div = soup.find('div', attrs={'class': constants.REVIEW_COUNT_DIV})
    #if review count is greater than three then go to reviews page
    if all_reviews_div:
        reviews_url = "https://www.flipkart.com" + all_reviews_div.parent.get('href')
        reviews_thread = WebDriverThread(target=get_laptop_reviews, args=(reviews_url,MAX_LAPTOP_REVIEW_PAGE_COUNT))
        reviews_thread.start()
        reviews = reviews_thread.join()
    #if review count is less than three then continue
    else:
        continue

    # Get features
    name = soup.find('span', attrs={'class': constants.NAME_SPAN}).string.split('-')[0].rstrip()

    processor_memory_features_tbody = soup.find('div', attrs={'class': constants.SPECIFICATIONS_PARENT_DIV}).contents[1].contents[1].contents[0]
    spec_keys = ['Processor Brand', 'Processor Name', 'RAM', 'Storage Type','SSD Capacity','Graphic Processor']
    specs = get_laptop_specifications(processor_memory_features_tbody, spec_keys)

    processor_brand = specs.get('Processor Brand',"Unknown")
    processor_name = specs.get('Processor Name',"Unknown")
    graphic_processor = specs.get('Graphic Processor',"Unknown")
    ram_capacity = specs.get('RAM',"Unknown")
    storage_type = specs.get('Storage Type',"Unknown")
    storage_capacity = specs.get('SSD Capacity',"Unknown")

    display_features_tbody = soup.find('div', attrs={'class': constants.SPECIFICATIONS_PARENT_DIV}).contents[4].contents[1].contents[0]
    spec_keys = ['Screen Size']
    specs = get_laptop_specifications(display_features_tbody, spec_keys)

    screen_size = specs.get('Screen Size',None)
    if screen_size:
        screen_size = screen_size.split('(')[1].split(')')[0]

    product_id = url.rpartition('/')[-1]
    
    # Add the laptop to the array
    laptop = Laptop(url=url,product_id=product_id,name=name,
                    processor_brand=processor_brand,processor_name=processor_name,ram_capacity=ram_capacity,
                    storage_type=storage_type,storage_capacity=storage_capacity,
                    screen_size=screen_size,reviews=reviews,graphic_processor=graphic_processor)
    
    #Add laptop to the database
    added_laptop_id = laptop_db.add_laptop(laptop)
    #TO DO: mongo db ekleme yapılacak

    if not added_laptop_id:
        continue 

    laptop.id = str(added_laptop_id)
    save_laptop_as_markdown(str(os.getenv('LAPTOP_MARKDOWNS_PATH')), laptop)
    laptop_array.append(laptop)

driver.quit()