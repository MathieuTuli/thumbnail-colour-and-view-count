import re
import requests
import os
import time
import imghdr
import sys
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from PIL import Image
from collections import namedtuple
# from math import sqrtz

#global variables
Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))
rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

class LoadedBrowser:
    def __init__(self, site):
        self.browser = webdriver.Chrome("chromedriver.exe")
        print("\nOpening browser...")
        self.browser.get(site)
        print("Browser open.\n")
        time.sleep(1)
        self.body = self.browser.find_element_by_tag_name("body")

    def scroll(self):
        no_of_pagedowns = 10
        while no_of_pagedowns:
            try:
                self.body.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.2)
                no_of_pagedowns -= 1
            except:
                break

    def close(self):
        print("\nClosing browser...")
        self.browser.close()
        print("Browser closed.")

def reset_folders():
    while True:
        try:
            print("Trying to remove jpg directory...")
            shutil.rmtree('jpg_images')
            print("Removed jpg directory")
        except:
            print("jpg directory not present. Building...")
            try:
                os.mkdir('jpg_images')
                print("jpg directory built")
                break
            except:
                continue
        try:
            os.mkdir('jpg_images')
            print("jpg directory built")
            break
        except:
            print("Couldn't build jpg directory. Retrying...")

    while True:
        try:
            print("Trying to remove webp directory...")
            shutil.rmtree('webp_images')
            print("Removed webp directory")
        except:
            print("webp directory not present. Building...")
            try:
                os.mkdir('webp_images')
                print("webp directory built")
                break
            except:
                continue
        try:
            os.mkdir('webp_images')
            print("webp directory built")
            break
        except:
            print("Couldn't build webp directory. Retrying...")

def find_images(body, counter):
    #HTML element tag that defines the thumbal group
    elem = body.find_elements_by_tag_name('ytd-grid-video-renderer')

    for item in elem:
        #HTML path for the thumbnail image
        url = item.find_element_by_xpath('.//img[@class = "style-scope yt-img-shadow"]')
        url = url.get_attribute('src')

        #Basically, assure the image src is usable
        try:
            filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', url)
        except:
            counter += 1
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

                #the above saves the image as webp file (google chrome filetype)
                #convert that into a jpg
                image = Image.open("webp_images/" + str(counter) + ".webp").convert("RGB")
                if (image.size == (246,138) or image.size == (336,188)):
                    image.save("jpg_images/" + str(counter) + ".jpg", "jpeg")
                else:
                    counter -= 1
            except:
                counter += 1
                continue

        counter += 1

def find_views(body, counter):
    elem = body.find_elements_by_tag_name('ytd-grid-video-renderer')
    with open("views.txt", 'w') as f:
        for item in elem:
            #info tags are identical for views and for video time
            #tbh, this magically skips over the video time
            info_tags = item.find_element_by_xpath('.//span[@class = "style-scope ytd-grid-video-renderer"]')
            text = info_tags.text
            # print(counter, " : ",text)
            try:
                f.write(str(counter) + " : " + text + '\n')
            except:
                continue
            counter += 1
    return counter

if __name__ == '__main__':
    print("\n\n__START__\n")
    site = 'https://www.youtube.com/user/Matroix/videos'
    chromeBrowser = LoadedBrowser(site)

    errorCode = 0
    numVideos = 0
    try:
        print('Scrolling browser...')
        chromeBrowser.scroll()
        errorCode = 1
        print('Browser scrolled.\n\nResetting folders...')
        reset_folders()
        errorCode = 2
        print('Folders reset.\n\nFinding images...')
        find_images(chromeBrowser.body, 0)
        errorCode = 3
        print('Images found.\n\nFinding views...')
        numVideos = find_views(chromeBrowser.body, 0)
        errorCode = 4
        print('Views found.')
    except Exception as e:
        if errorCode == 0:
            print("\nScrolling failed.: ", e)
        elif errorCode == 1:
            print("\nResetting folders failed: ", e)
        elif errorCode == 2:
            print("\nFinding images failed: ", e)
        elif errorCode == 3:
            print("\nFinding views failed: ", e)

    chromeBrowser.close()

    # try:
    #     print('\nFinding most frequent colour...')
    #     most_frequent_colour(Image.open('jpg_images/0.jpg'))
    #     errorCode == 5
    #     print('Most frequent colour found.')
    # except Exception as e:
    #     if errorCode == 4:
    #         print("Finding most frequent colour failed: ", e)

    print("\n__END__")
