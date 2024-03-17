import re
import sys
import json
import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
import random


def scraper(board, fileName, numOfPages=1, whichPage=2):
    url = '/bbs/' + board + '/index.html'
    resp = requests.get('https://www.ptt.cc' + url)
    if resp.url.find('over18') > -1:
        print("\nThe board '%s' is admitted for over 18 only." % board)
    resp = enterAgeCheck(resp, url)
    soup = BeautifulSoup(resp.text, 'lxml')

    # Wait for a second if busy
    if (soup.title.text.find('Service Temporarily') > -1):
        print('\nService busy...\n')
        time.sleep(1)
    # Start scraping
    else:
        print('Start scraping...\n')
        # scraping start from given page 'whichPage'
        for page in range(whichPage-1):
            try:
                pageUp = soup.select('.btn-group-paging')[0].findAll('a')[1]
                url = pageUp.attrs['href']
                resp = requests.get('https://www.ptt.cc' + url)
                resp = enterAgeCheck(resp, url)
                soup = BeautifulSoup(resp.text, 'lxml')
            except Exception as e:
                print(e + 'Less pages than given number in %s' % board)
                break
        data = []
        # scrape page by page
        for page in range(numOfPages):
            print('Scraping page %d...' % (page + 1))

            links = soup.select('.title')
            for link in links:
                try:
                    url = link.find('a').attrs['href']
                except:
                    continue
                sample_data = linkParser(url)
                if sample_data:
                    data.append(sample_data)
                # delay the downloading speed
                time.sleep(0.1)

            try:
                pageUp = soup.select('.btn-group-paging')[0].findAll('a')[1]
                url = pageUp.attrs['href']
                resp = requests.get('https://www.ptt.cc' + url)
                resp = enterAgeCheck(resp, url)
                soup = BeautifulSoup(resp.text, 'lxml')
            except:
                print('Cannot scrape next page. May be the final page.')
                break
            # delay the downloading speed
            time.sleep(0.1)

        print('Done scraping.\n')
        return data


def enterAgeCheck(response, url):
    # check 'over18' string within the URL
    if response.url.find('over18') > -1:
        data_to_load = {
            'from': url,
            'yes': 'yes'
        }
        response = requests.post(response.url.split('?')[0], data=data_to_load)
        return response
    else:
        return response


def metaCheck(soup, class_tag, data_name, index, link):
    # check if meta data (topic, id, etc) is there
    try:
        data = soup.select(class_tag)[index].text
    except:
        print('Error in %s with no %s' % (link, data_name))
        data = data_name + '_missed'
    return data


def linkParser(url):
    ## Parsing data items from given link
    try:
        resp = requests.get('https://www.ptt.cc' + url)
        resp = enterAgeCheck(resp, url)
        soup = BeautifulSoup(resp.text, 'lxml')
    # requests.exceptions
    except Exception as e:
        print('Request error message:', e)
        return None
    # children = [c for c in soup.select('#main-content')[0].children]
    try:
        mainContent = soup.select('#main-content')[0]
    except:
        print('Error in %s with no contents.' % url)
        return None

    # Messages
    msgList = []
    for tag in mainContent.select('.push'):
        try:
            tagContent = [c for c in tag.children]
            msgList.append(tagContent[2].text)
        except:
            continue

    sample_data = {'messages': msgList}

    return sample_data


def dataStore(data, fileName, format='txt'):
    print('Saving data...')
    if format == 'txt':
        with open(fileName + '.txt', 'w', encoding='utf-8') as f:
            for item in data:
                for msg in item['messages']:
                    f.write("%s\n" % msg)
    elif format == 'json':
        with open(fileName + '.json', 'w') as f:
            json.dump(data, f)


def split_train_test_data(data, test_ratio=0.2):
    random.shuffle(data)
    split_index = int(len(data) * test_ratio)
    test_data = data[:split_index]
    train_data = data[split_index:]
    return train_data, test_data


if __name__ == "__main__":
    pttName = "Badminton"#Badminton Stock
    #pttName = str(sys.argv[1])

    try:
        numOfPages = int(sys.argv[2])
    except:
        numOfPages = 8

    try:
        whichPage = int(sys.argv[3])
    except:
        whichPage = 2

    fileName = 'train_' + pttName
    t0 = time.time()
    data = scraper(board=pttName, numOfPages=numOfPages, whichPage=whichPage, fileName=fileName)
    random.shuffle(data)
    
    train_data, test_data = split_train_test_data(data, test_ratio=0.2)
    dataStore(train_data, fileName)
    print('Scraping with elapsed time', time.time()-t0, 'seconds.')

    # Saving test data
    test_fileName = 'test_' + pttName
    dataStore(test_data, test_fileName)
    print(f'Random 20% of data saved to {test_fileName}.txt')
    
