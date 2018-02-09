import re
import requests
from bs4 import BeautifulSoup
import os
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import imghdr
from PIL import Image
import sys
import shutil

site = 'https://www.youtube.com/user/Matroix/videos'

browser = webdriver.Chrome("C:/Users/tulim/Documents/Code/web scrapers/Ali-A/chromedriver.exe")
browser.get(site)
# response = requests.get(site)

time.sleep(1)

no_of_pagedowns = 10
counter = 0
counter2 = 0

elem = browser.find_element_by_tag_name("body")

while no_of_pagedowns:
    try:
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        no_of_pagedowns -= 1
    except:
        continue


try:
    shutil.rmtree('jpg_images')
    os.mkdir('jpg_images')
except:
    print("jpg directory not present. Building...")
    os.mkdir('jpg_images')
try:
    shutil.rmtree('webp_images')
    os.mkdir('webp_images')
except:
    print("webp directory not present. Building...")
    os.mkdir('webp_images')

elem = elem.find_elements_by_tag_name('ytd-grid-video-renderer')
# soup = BeautifulSoup(response.text, 'html.parser')
# img_tags = elem.find_elements_by_tag_name('img')
# img_tags = elem.find_elements_by_xpath('.//img[@class = "style-scope yt-img-shadow"]')
# urls = [img['src'] for img in img_tags]style-scope yt-img-shadow

# for url in urls:
for item in elem:
    #would remove if using request
    url = item.find_element_by_xpath('.//img[@class = "style-scope yt-img-shadow"]')
    url = url.get_attribute('src')
    try:
        filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
    except:
        continue
    with open("webp_images/" + str(counter) + ".webp", 'wb') as f:
        if 'http' not in url:
            # sometimes an image source can be relative
            # if it is provide the base url which also happens
            # to be the site variable atm.
            url = '{}{}'.format(site, url)
        try:
            response = requests.get(url)
            f.write(response.content)
            image = Image.open("webp_images/" + str(counter) + ".webp").convert("RGB")
            if (image.size == (246,138) or image.size == (336,188)):
                image.save("jpg_images/" + str(counter) + ".jpg", "jpeg")
            else:
                counter -= 1
        except:
            counter += 1
            continue
    counter += 1


for item in elem:
    info_tags = item.find_element_by_xpath('.//span[@class = "style-scope ytd-grid-video-renderer"]')
    #would remove if using request
    text = info_tags.text
    print(counter2, " : ",text)
    with open("info.txt", 'wb') as f:
        try:
            f.write(text)
        except:
            counter2 += 1
            continue

char = input("=>")
browser.close()
