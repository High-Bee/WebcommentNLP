# module import
import requests
import urllib.parse
from bs4 import BeautifulSoup
import re
from selenium import webdriver
import time
import csv
import os

# 변수 설정
QUERY = "주52시간"
START_DATE = "2019.12.01"
END_DATE = "2019.12.31"

search_QUERY = urllib.parse.urlencode({'query': QUERY}, encoding='utf-8')
start_QUERY = urllib.parse.urlencode({'ds': START_DATE}, encoding='utf-8')
end_QUERY = urllib.parse.urlencode({'de': END_DATE}, encoding='utf-8')
p_QUERY = urllib.parse.urlencode({'p': f"from{START_DATE.replace('.', '')}to{END_DATE.replace('.', '')}"}, encoding='utf-8')

URL = f"https://search.naver.com/search.naver?&where=news&{search_QUERY}&sm=tab_pge&sort=2&photo=0&field=0&reporter_article=&pd=3&{start_QUERY}&{end_QUERY}&docid=&nso=so:da,{p_QUERY},a:all&mynews=0"

LINK_PAT = "https:\/\/news\.naver\.com\/main\/read\.nhn\?"
search_PAGE = 326

# driver 설정
driver = webdriver.Chrome("C:/Users/user/PycharmProjects/Scraping/chromedriver.exe")


# 검색결과 내 링크 찾기 : news.naver.com으로 시작하는 모든 링크 반환
def get_news_links(page_num, link_pattern):
    links = []
    for page in range(page_num):
        print(f"Scrapping page : {page + 1}")  # 확인용
        req = requests.get(f"{URL}&start={10 * page + 1}")
        print(req.status_code)  # 확인용
        soup = BeautifulSoup(req.text, 'lxml')
        results = soup.find_all('a', {'href': re.compile(link_pattern)})
        for result in results:
            links.append(result['href'])
    print(f"총 {len(links)}개의 뉴스 링크를 찾았습니다.")  # 확인용
    return links


# 한 페이지 별로 필요한 정보 스크레이핑
def extract_info(url, wait_time=2, delay_time=0.5):
    driver.implicitly_wait(wait_time)
    driver.get(url)

    # 댓글 창 있으면 다 내리기
    while True:
        try:
            more_comments = driver.find_element_by_css_selector('a.u_cbox_btn_more')
            more_comments.click()
            time.sleep(delay_time)
        except:
            break

    # html 페이지 읽어오기
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')

    result = []

    try: # 연예 분야 뉴스 제외

        site = soup.find('h1').find("span").get_text(strip=True)  # 출처
        title = soup.find('h3', {'id': 'articleTitle'}).get_text(strip=True)  # 기사 제목
        article_time = soup.find('span', {'class': 't11'}).get_text(strip=True)  # 작성 시간

        press = soup.find('div', {'class': "press_logo"}).find('a').find('img')['title']  # 언론사

        total_com = soup.find("span", {"class": "u_cbox_info_txt"}).get_text(strip=True)  # 댓글 수
        total_com = int(total_com.replace('\n', '').replace('\t', '').replace('\r', '').replace(',', ''))

        if total_com == 0: # 댓글 없는 경우
            result = [{'site': site,
                       'title': title,
                       'article_time': article_time,
                       'press': press,
                       'total_comments': total_com,
                       'nickname': None,
                       'date': None,
                       'contents': None,
                       'recomm': None,
                       'unrecomm': None}]
        else:
            nicks = soup.find_all("span", {"class": "u_cbox_nick"})  # 댓글 작성자
            nicks = [nick.text for nick in nicks]

            dates = soup.find_all("span", {"class": "u_cbox_date"})  # 댓글 날짜
            dates = [date.text for date in dates]

            contents = soup.find_all("span", {"class": "u_cbox_contents"})  # 댓글 내용
            contents = [content.text for content in contents]

            recomms = soup.find_all("em", {"class": "u_cbox_cnt_recomm"})  # 공감 수
            recomms = [recomm.text for recomm in recomms]

            unrecomms = soup.find_all("em", {"class": "u_cbox_cnt_unrecomm"})  # 비공감수
            unrecomms = [unrecomm.text for unrecomm in unrecomms]

            for i in range(len(contents)):
                result.append({'site': site,
                               'title': title,
                               'article_time': article_time,
                               'press': press,
                               'total_comments': total_com,
                               'nickname': nicks[i],
                               'date': dates[i],
                               'contents': contents[i].replace('\r','').replace('\t','').replace('\n',''),
                               'recomm': recomms[i],
                               'unrecomm': unrecomms[i]})

    except: # 연예 분야 뉴스인 경우 AttributeError.
        result = [{'site': None,
                   'title': None,
                   'article_time': None,
                   'press': None,
                   'total_comments': None,
                   'nickname': None,
                   'date': None,
                   'contents': None,
                   'recomm': None,
                   'unrecomm': None}]

    return result


# 각 페이지 돌면서 스크레이핑
def extract_contents(links):
    for link in links:
        print(f"{link}&m_view=1")
        content = extract_info(f"{link}&m_view=1")
        append_to_file(content)
    return print("모든 작업이 완료되었습니다.")


# 파일 만드는 함수
def make_file():

    if os.path.exists(f"C:/Users/user/PycharmProjects/Scraping/news_comments_NAVER_{START_DATE}_{END_DATE}.csv"):
        raise NameError("동일한 파일이 존재합니다.")

    file = open(f"news_comments_NAVER_{START_DATE}_{END_DATE}.csv", mode="w", encoding="UTF-8")
    writer = csv.writer(file)
    writer.writerow(['site', 'title', 'article_time', 'press', 'total_comments', 'nickname', 'date', 'contents', 'recomm', 'unrecomm'])
    file.close()
    return


# 파일에 한 줄씩 덮어 쓰는 함수
def append_to_file(lst):
    global START_DATE
    global END_DATE
    file = open(f"news_comments_NAVER_{START_DATE}_{END_DATE}.csv", mode="a", encoding="UTF-8")
    writer = csv.writer(file)
    for result in lst:
        writer.writerow(list(result.values()))
    file.close()
    return


# main 함수
def main():
    global search_PAGE
    make_file()
    news_links = get_news_links(search_PAGE, LINK_PAT)
    result = extract_contents(news_links)
    driver.quit()
    return


# 함수 실행
main()