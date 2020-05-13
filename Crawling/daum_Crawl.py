#댓글이 있는 기사링크만 가져오기

import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import exceptions
import datetime

# selenium 옵션설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--diable-dev-shm-usage')
wd = "./chromedriver_win32/chromedriver"
driver_daum = webdriver.Chrome(wd,options=options)
driver_news = webdriver.Chrome(wd,options=options)

#링크저장 리스트 생성
news_link = []
news_num = 0
for j in range(132,1017):
    # 20170301 ~ 20191212 까지 순차 검색 
    day_delta = datetime.timedelta(days=1)
    start_date = datetime.date(2017,3,2)
    first_date = str(start_date+day_delta*j).replace('-','')
    print(first_date)
    
    url = f"https://search.daum.net/search?w=news&nil_search=btn&DA=STC&enc=utf8&cluster=y&cluster_page=1&q=%EC%A3%BC%2052%EC%8B%9C%EA%B0%84&sd={first_date}000000&ed={first_date}235959&period="
    driver_daum.get(url)
    
    for i in range(1,21):
        driver_daum.get(url+'1&p2=&p='+str(i))
        #뉴스사마다 다른 셀렉터 변수 예외 처리
        try:
            next_page = driver_daum.find_element_by_css_selector("a.ico_comm1.btn_page.btn_next").text
        except exceptions.NoSuchElementException as e:
            next_page = '1'
            pass
        except Exception as e: # 다른 예외 발생시 확인
            print(e)
    
        html = driver_daum.page_source
        dom = BeautifulSoup(html, "lxml")
        daum_link = dom.find_all('a','f_nb') #다음뉴스 링크 모음
        if len(daum_link) > 0:
            for link in daum_link:
                try:
                    driver_news.get(link['href'])
                    driver_news.implicitly_wait(3)
                    count_re = driver_news.find_element_by_xpath('//*[@id="alex-header"]/em').text 
                    if int(count_re) > 0:
                        news_link.append(link['href'])
                        news_num += 1
                except exceptions.InvalidArgumentException:
                    pass
                except Exception as e: # 다른 예외 발생시 확인
                    print(e, link['href'])
                except:
                    count_re = driver_news.find_element_by_xpath('//*[@id="alexCounter"]/span').text
                    if int(count_re) > 0:
                        news_link.append(link['href'])
                        news_num += 1
                        

        print('page :',i)        
        if next_page != '':
            break


    print(len(news_link))


#가져올 내용 저장시킬 데이터프레임 생성
ndf = pd.DataFrame({
    '작성자': [],
    '댓글시간': [],
    '댓글': [],
    '공감수': [],
    '비공감수': [],
    '답글수': [],
    '포털': [],
    '언론사': [],
    '기사제목': [],
    '기사시간':  []   
})

#댓글이 있는 뉴스기사의 필요 정보 가져오기

import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import exceptions
import datetime
import pandas as pd
import numpy as np

# selenium 옵션 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--diable-dev-shm-usage')
wd = "./chromedriver_win32/chromedriver"
driver_news = webdriver.Chrome(wd,options=options)

for link in a:
    driver_news.get(link)
    driver_news.implicitly_wait(4)
    pages = 0 # 댓글창이 확장될때 까지
    try:
        cnt = driver_news.find_element_by_xpath('//*[@id="alex-header"]/em').text
    except exceptions.NoSuchElementException:
        cnt = '10'
        print(cnt)
        pass
    if int(cnt) > 3:
        try:
            while True: # 댓글 페이지가 몇개인지 모르므로.
                more_view = driver_news.find_element_by_xpath('//*[@id="alex-area"]/div/div/div/div[3]/div[2]/a')
                more_view.click()
                time.sleep(1.5)
                print(pages, end=" ")
                # 댓글창이 무제한 늘어나는 버그 조건처리
                if int(cnt) >150 and pages > int(int(cnt)/10): 
                    break
                if pages > int(int(cnt)/2):
                    break
                if pages > 300:
                    break
                pages+=1
                
        except exceptions.ElementNotVisibleException as e: # 페이지 끝
            pass
        except Exception as e: # 다른 예외 발생시 확인
            print(e)

    html = driver_news.page_source
    soup = BeautifulSoup(html, "lxml")
    try:
        potal = '다음'
        press = soup.select_one("a.link_cp>img.thumb_g")['alt']
        title = soup.select_one('h3.tit_view').text
        news_time = soup.select('span.info_view > span.txt_info')[-1].text
        authors = soup.find_all('a', 'link_nick')
        replies_time = soup.find_all(attrs = {'class' : 'txt_date'})
        replies = soup.select('div.cmt_info')
        recom_counts = soup.find_all(attrs = {'class' : 'btn_g btn_recomm #like ?c_title=%EB%8C%93%EA%B8%80%EC%B0%AC%EC%84%B1'})
        oppose_counts = soup.find_all(attrs = {'class' : 'btn_g btn_oppose #dislike ?c_title=%EB%8C%93%EA%B8%80%EB%B0%98%EB%8C%80'})
        reply_count = soup.find_all(attrs = {'class' : 'reply_count #reply ?c_title=%EB%8B%B5%EA%B8%80'})

        print("기사 정보",press,title,news_time)
        print("리스트 길이",len(authors), len(replies_time),len(replies),len(recom_counts),len(oppose_counts),len(reply_count))

        df = pd.DataFrame({
            '작성자': [author.text for author in authors],
            '댓글시간': [reply_time.text for reply_time in replies_time],
            '댓글': [reply.text for reply in replies],
            '공감수': [recom.text for recom in recom_counts],
            '비공감수': [oppose.text for oppose in oppose_counts],
            '답글수': [count.text for count in reply_count],
            '포털': [potal]*len(authors),
            '언론사': [press]*len(authors),
            '기사제목': [title]*len(authors),
            '기사시간':  [news_time]*len(authors) 
        })
        # 수집한 내용 합치기
        ndf = pd.concat([ndf, df])
    except exceptions.ElementNotSelectableException:
        pass
    except exceptions.NoSuchElementException:
        pass
    except TypeError:
        pass
    
ndf.to_csv('./Data/.csv')