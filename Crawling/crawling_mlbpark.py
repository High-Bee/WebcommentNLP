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
from datetime import datetime as dtime
import csv

# 변수 설정
QUERY = "주52시간"
search_QUERY = urlencode({'query': QUERY}, encoding='utf-8')
URL = f"http://mlbpark.donga.com/mp/b.php?select=sct&m=search&b=bullpen&select=sct&{search_QUERY}&x=0&y=0"


# 마지막 페이지까지 클릭
def go_to_last_page(url):
    options = Options()
    ua = UserAgent()
    userAgent = ua.random
    print(userAgent)
    options.add_argument(f'user-agent={userAgent}')
    driver = webdriver.Chrome(chrome_options=options,
                              executable_path='C:/Users/user/PycharmProjects/Scraping/chromedriver.exe')
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    while True:
        # class가 right인 버튼이 없을 때까지 계속 클릭
        try:
            time.sleep(5)
            element = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'right')))
            element.click()
            time.sleep(5)
        except TimeoutException:
            print("no pages left")
            break
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')
    driver.quit()
    return soup


# 마지막 페이지 번호 알아내기
def get_last_page(url):
    soup = go_to_last_page(url)
    pagination = soup.find('div', {'class': 'page'})
    pages = pagination.find_all("a")
    page_list = []
    for page in pages[1:]:
        page_list.append(int(page.get_text(strip=True)))
    max_page = page_list[-1]
    print(f"총 {max_page}개의 페이지가 있습니다.")
    return max_page


# 게시판 링크 모두 가져오기
def get_boards(page_num):
    boards = []
    for page in range(page_num):
        boards.append(
            f"http://mlbpark.donga.com/mp/b.php?p={30 * page + 1}&m=search&b=bullpen&{search_QUERY}&select=sct&user=")
    return boards


# 게시글 링크 가져오기
def get_posts():
    global QUERY
    global PAGES
    board_links = get_boards(PAGES)
    posts = []
    for board_link in board_links:
        # print(f"게시판 링크는 {board_link}")
        req = requests.get(board_link)
        print(req.status_code)  # 49개 나와야 함
        soup = BeautifulSoup(req.text, 'lxml')
        tds = soup.find_all('td', {'class': 't_left'})
        for td in tds:
            post = td.find('a', {'class': 'bullpenbox'})
            if post is not None:
                posts.append(post['href'])
    print(f"총 {len(posts)}개의 글 링크를 찾았습니다.")

    # 게시글 링크 csv로 저장
    post_file = open(f"MLBPARK_{QUERY}_{PAGES}pages_inner_links.csv", mode='w', encoding='utf-8')
    writer = csv.writer(post_file)
    for post in posts:
        writer.writerow([post])
    post_file.close()

    return posts


# 한 페이지에서 정보 가져오기
def extract_info(url, wait_time=3, delay_time=1):
    driver = webdriver.Chrome("C:/Users/user/PycharmProjects/Scraping/chromedriver.exe")

    driver.implicitly_wait(wait_time)
    driver.get(url)
    html = driver.page_source
    time.sleep(delay_time)  # 강제 연결 종료 방지

    driver.quit()

    soup = BeautifulSoup(html, 'lxml')

    site = soup.find('h1', {'class': 'logo'}).find('a').find('img')['title'].strip()

    title = soup.find('div', {'class': 'titles'}).get_text(strip=True)

    user_id = soup.find('span', {'class': 'nick'}).get_text(strip=True)

    post_time = soup.find('div', {'class': 'text3'}).find('span', {'class': 'val'}).get_text(
        strip=True) + ":00"  # datetime 형식으로 바꾸기 위해 초 추가.
    post_time = dtime.strptime(post_time, '%Y-%m-%d %H:%M:%S')

    post = soup.find('div', {'id': 'contentDetail'}).get_text(strip=True)

    view_cnt = int(
        soup.find('div', {'class': 'text2'}).find_all('span', {'class': 'val'})[1].get_text(strip=True).replace('\n',
                                                                                                                '').replace(
            '\r', '').replace(',', ''))

    recomm_cnt = int(
        soup.find('span', {'id': 'likeCnt'}).get_text(strip=True).replace('\n', '').replace('\r', '').replace(',', ''))

    reply_cnt = int(
        soup.find('span', {'id': 'replyCnt'}).get_text(strip=True).replace('\n', '').replace('\r', '').replace(',', ''))

    reply_content = []
    if reply_cnt != 0:
        replies = soup.find_all('span', {'class': 're_txt'})
        for reply in replies:
            reply_content.append(reply.get_text(strip=True).replace('\n', '').replace('\r', '').replace('\t', ''))
    reply_content = '\n'.join(reply_content)

    print(url, " 완료")

    return {'site': site, 'title': title, 'user_id': user_id, 'post_time': post_time, 'post': post,
            'view_cnt': view_cnt, 'recomm_cnt': recomm_cnt, 'reply_cnt': reply_cnt, 'reply_content': reply_content}


# 모든 게시물 링크에 대해 정보 가져오는 함수 호출
def get_contents():
    global mlbpark_results
    post_links = get_posts()
    for post_link in post_links:
        content = extract_info(post_link)
        append_to_file(mlbpark_results, content)
    return print("모든 작업이 완료되었습니다.")


# 저장 파일 만드는 함수
def save_to_file():
    global QUERY
    global PAGES
    file = open(f"MLBPARK_{QUERY}_{PAGES}pages_checkiferror.csv", mode='w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(
        ['site', 'title', 'user_id', 'post_time', 'post', 'view_cnt', 'recomm_cnt', 'reply_cnt', 'reply_content'])
    file.close()
    return file


# 파일 열어서 쓰는 함수
def append_to_file(file_name, dictionary):
    file = open(f"MLBPARK_{QUERY}_{PAGES}pages_jupyternotebook.csv", mode='a', encoding='utf-8')  # 덮어 쓰기
    writer = csv.writer(file)
    writer.writerow(list(dictionary.values()))
    file.close()
    return


# 함수 실행
PAGES = get_last_page(URL)
mlbpark_results = save_to_file()
get_contents()