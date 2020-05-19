# 실효성 판단 데이터 로드
import pandas as pd

ws_df = pd.read_csv("./data/생산성안전.csv")
rh_df = pd.read_csv("./data/여가가정.csv")
me_df = pd.read_csv("./data/임금고용.csv")

w_series = ws_df[ws_df.id == "생산성"].news
s_series = ws_df[ws_df.id == "안전"].news
r_series = rh_df[rh_df.id == "여가"].news
h_series = rh_df[rh_df.id == "가정"].news
m_series = me_df[me_df.id == "임금"].news
e_series = me_df[me_df.id == "고용"].news

w_series = pre_pro(w_series)
s_series = pre_pro(s_series)
r_series = pre_pro(r_series)
h_series = pre_pro(h_series)
m_series = pre_pro(m_series)
e_series = pre_pro(e_series)

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

w_series = pre_pro(w_series)
s_series = pre_pro(s_series)
r_series = pre_pro(r_series)
h_series = pre_pro(h_series)
m_series = pre_pro(m_series)
e_series = pre_pro(e_series)

#모델 학습용 데이터셋 생성 완료
tagged_list = [w_series, s_series, r_series, h_series, m_series, e_series]

## 실효성 평가 Doc2Vec & 분류 Logic
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tagged_list)]
model = Doc2Vec(documents, dm=1 ,vector_size=100, window=10, seed=1234,
                min_count=4, workers=4, sample=1e-5, epochs=20)

# 학습이 완료 되면 필요없는 메모리 unload
model.init_sims(replace=True)
model_name = '100features_4minwords_doc2vec_text'
model.save(model_name)

# 학습 결과 dict생성
bf_dict = {'work':[],'safety':[],"rest":[],"home":[],"money":[],"emplo":[]}

# 분석 시작
for idx, text in tqdm_notebook(enumerate(bf_s_com_1)):
    inferred_v = d_model.infer_vector(text)
    # 기학습된 문서중에서 현재 벡터와 가장 유사한 벡터를 가지는 문서를 topn만큼 추출
    most_similar_docs = d_model.docvecs.most_similar([inferred_v], topn=1)
    # index와 유사도 조건 이상을 dict에 저장
    for index, similarity in most_similar_docs:
        if similarity > 0.9950:
            if index == 0:
                bf_dict["work"].append(idx)
            elif index == 1:
                bf_dict["safety"].append(idx)
            elif index == 2:
                bf_dict["rest"].append(idx)
            elif index == 3:
                bf_dict["home"].append(idx)
            elif index == 4:
                bf_dict["money"].append(idx)
            elif index == 5:
                bf_dict["emplo"].append(idx)
            
