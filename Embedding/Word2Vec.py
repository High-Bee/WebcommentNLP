# 데이터 전처리 및 형태소 변환
# 전처리
import re
from tqdm import tqdm_notebook
def re_hangul(s):
    hangul = re.compile('[^ 가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
    result = hangul.sub('', str(s)) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
    return result

# 형태소 변환
from konlpy.tag import Mecab
mecab = Mecab()
def pre_pro(li):
    t_w_series = []
    for i in tqdm_notebook(li):
        t_w_series.append(re_hangul(i))
        
    mongchi_list = []
    for i in tqdm_notebook(t_w_series):
        mongchi_list.append(mecab.morphs(i))
        
    for idx,i in tqdm_notebook(enumerate(mongchi_list)):
        for j in i:
            if len(str(j)) == 1:
                mongchi_list[idx].remove(j)
    return mongchi_list
    
# model 파라메터값 지정
num_features = 200 # 문자 벡터 차원 수
min_word_count = 5 # 최소 문자 수
num_workers = 4 # 병렬 처리 스레드 수
context = 10 # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도수 Downsample

from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 모델 학습
model = Word2Vec(mongchi_list, 
                  workers=num_workers, 
                  size=num_features, 
                  min_count=min_word_count,
                  window=context,
                  iter=15,
                  sg=1)
model

# 학습이 완료 되면 필요없는 메모리 unload
model.init_sims(replace=True)
model_name = '200features_5minwords_10_pre2m_text'
model.save(model_name)