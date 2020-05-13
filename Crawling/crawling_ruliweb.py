import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import re
from datetime import datetime as dtime

# 통합검색 URL : https://bbs.ruliweb.com/search?q=%EC%A3%BC52%EC%8B%9C%EA%B0%84&page=36

# 변수 설정
QUERY = "주52시간"
search_QUERY = urlencode({'q' : QUERY}, encoding = 'utf-8')
URL = f"https://bbs.ruliweb.com/search?{search_QUERY}"
POST_PAGES = 37
COMMENT_PAGES = 49

# 게시글 링크 가져오기 : 게시글 검색 결과, 댓글 검색 결과 모두에서 루리웹으로 시작하는 주소 가져 오기.
def get_posts(post_pages, comment_pages):
    global URL
    LINKS = set()
    for page in range(post_pages):
        url = f"{URL}&page={page+1}"
        print(url)
        req = requests.get(url)
        print(req.status_code) # 36개 나와야 함.
        soup = BeautifulSoup(req.text, 'lxml')
        links = soup.find_all('a', {'class':'title text_over'})
        for link in links[:15]:
            matched = re.match("https\:\/\/bbs\.ruliweb\.com", link['href'])
            if matched:
                LINKS.add(link['href'])
        time.sleep(3)

    print(f"게시판 검색 결과 총 {len(LINKS)}개의 글 링크를 찾았습니다.")

    for page in range(comment_pages):
        url = f"{URL}&c_page={page+1}"
        print(url)
        req = requests.get(url)
        print(req.status_code) # 49개 나와야 함.
        soup = BeautifulSoup(req.text, 'lxml')
        links = soup.find_all('a', {'class':'title text_over'})
        for link in links:
            matched = re.match("https\:\/\/bbs\.ruliweb\.com", link['href'])
            if matched:
                LINKS.add(link['href'])

    print(f"댓글까지 검색한 결과 총 {len(LINKS)}개의 글 링크를 찾았습니다.")

    LINKS = list(LINKS)

    # 게시글 링크 csv로 저장
    post_file = open(f"ruliweb_{QUERY}_inner_links.csv", mode='w', encoding='utf-8')
    writer = csv.writer(post_file)
    for LINK in LINKS:
        writer.writerow([LINK])
    post_file.close()

    return LINKS

# 한 페이지에서 정보 가져오기
def extract_info(url, wait_time=3, delay_time=2):

    driver = webdriver.Chrome("C:/Users/user/PycharmProjects/Scraping/chromedriver.exe")

    driver.implicitly_wait(wait_time)
    driver.get(url)
    html = driver.page_source
    time.sleep(delay_time)  # 강제 연결 종료 방지

    driver.quit()

    soup = BeautifulSoup(html, 'lxml')

    try:
        site = soup.find('img', {'class': 'ruliweb_icon'})['alt'].strip()
    except:
        site = "루리웹"

    try:
        title = soup.find('span', {'class': 'subject_text'}).get_text(strip=True)

        user_id = soup.find('strong', {'class': 'nick'}).get_text(strip=True)

        post_time = soup.find('span', {'class': 'regdate'}).get_text(strip=True).replace('.','-').replace('(','').replace(')','') # datetime 형식으로 바꾸기 위한 작업
        post_time = dtime.strptime(post_time, '%Y-%m-%d %H:%M:%S')

        post = soup.find('div', {'class': 'view_content'}).get_text(strip=True).replace('\n', '').replace('\r', '').replace('\t', '')

        view_cnt = int(soup.find('div', {'class': 'user_info'}).find_all('p')[4].get_text(strip=True).split()[-1].replace(',',''))

        recomm_cnt = int(soup.find('div', {'class': 'user_info'}).find('span', {'class':'like'}).get_text(strip=True).replace('\n', '').replace('\r', '').replace('\t', '').replace(',', ''))

        reply_cnt = int(soup.find('strong', {'class':'reply_count'}).get_text(strip=True).replace('[','').replace(']','').replace(',',''))

        reply_content = []
        if reply_cnt != 0:
            replies = soup.find_all('span', {'class': 'text'})
            for reply in replies:
                reply_content.append(reply.get_text(strip=True).replace('\n', '').replace('\r', '').replace('\t', ''))
        reply_content = '\n'.join(reply_content)

        print(url, " 완료")

    except:
        print(url, "삭제된 게시물이거나, 오류가 있습니다.")
        title = None
        user_id = None
        post_time = None
        post = None
        view_cnt = None
        recomm_cnt = None
        reply_cnt = None
        reply_content = None

    return {'site': site, 'title': title, 'user_id': user_id, 'post_time': post_time, 'post' : post, 'view_cnt': view_cnt, 'recomm_cnt': recomm_cnt, 'reply_cnt': reply_cnt, 'reply_content': reply_content}

# 모든 게시물 링크에 대해 정보 가져오는 함수 호출
def get_contents():
    global ruliweb_results
    post_links = get_posts(POST_PAGES, COMMENT_PAGES)
    for post_link in post_links:
        content = extract_info(post_link)
        append_to_file(ruliweb_results, content)
    return print("모든 작업이 완료되었습니다.")

# 저장 파일 만드는 함수
def save_to_file():
    global QUERY
    global PAGES
    file = open(f"ruliweb_{QUERY}.csv", mode='w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(['site', 'title', 'user_id', 'post_time', 'post', 'view_cnt', 'recomm_cnt', 'reply_cnt', 'reply_content'])
    file.close()
    return file

# 파일 열어서 쓰는 함수
def append_to_file(file_name, dictionary):
    file = open(f"ruliweb_{QUERY}.csv", mode='a', encoding='utf-8') # 덮어 쓰기
    writer = csv.writer(file)
    writer.writerow(list(dictionary.values()))
    file.close()
    return

# 함수 실행
ruliweb_results = save_to_file()
get_contents()

