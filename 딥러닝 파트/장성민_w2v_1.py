
# 1. Library, Read Data
# 2. Feature Making:
#        - 0. oversampling2: 
#                            nunique 사용X -> 고객들이 산 물품 그대로 모두 뽑아냄, 
#                            비복원추출 -> 복원추출
#                           많이 산 고객들이 오버샘플링이 더 많이 됨
#        - 1. EmbeddingVectozier_W2V: (중분류 np.std, np.var 대분류 np.max, np.mean) -> 1200개
#        - 2. Cosine_Similarity_between_gender_&_Products_W2V: (대, 중, 소 물품들 사이에 gender를 넣어서 학습,
#                                                    고객별 물품들 gender와의 코사인 유사도를 계산.
#                                                    이후 모든 코사인 유사도를 각 고객별로 평균을 내어 피쳐로 채택. -> 6개
# 3. Feature Preprocessing: SelectPercentile
# 4. Modeling: Tuning with BayesianOptimization
# 5. Ensemble: Voting, Stacking: Stacking submission 채택





# basic 

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50, 'display.max_rows', 200)
import os
#os.environ["PYTHONHASHSEED"] = "123"


# plot

import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
%matplotlib inline

import platform
your_os = platform.system()
if your_os == 'Linux':
    rc('font', family='NanumGothic')
elif your_os == 'Windows':
    ttf = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=ttf).get_name()
    rc('font', family=font_name)
elif your_os == 'Darwin':
    rc('font', family='AppleGothic')
rc('axes', unicode_minus=False)


# models
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
import shap
from scipy.stats.mstats import gmean
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from bayes_opt import BayesianOptimization
from sklearn.ensemble import VotingClassifier
from vecstack import StackingTransformer
from vecstack import stacking

from gensim.models import word2vec
from sklearn.pipeline import Pipeline
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

### Read data
df_train = pd.read_csv('X_train.csv', encoding='cp949')
df_test = pd.read_csv('X_test.csv', encoding='cp949')
y_train = pd.read_csv('y_train.csv').gender
IDtest = df_test.cust_id.unique()

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)


### Make corpus: corpus는 말뭉치라는 뜻임

# oversample2: unique사용 X, 비복원추출 -> 복원추출
# data는 X_train 또는 X_test가 될 것
# p_level은 대, 중, 소분류중에 하나 넣기('gds_grp_mclas_nm', 'gds_grp_nm', 'goods_id')
# n은 몇번 오버샘플링 할 것인지 -> 1일때 기본 + 오버샘플링, 2일때 기본 + 오버샘플링2
#                               -> 고객리스트가 10개인데 n값이 1이면 고객리스트를 20개(기본(10) + 오버샘플링(10))로 반환
#                               -> 고객리스트가 10개인데 n값이 2이면 고객리스트를 30개(기본(10) + 오버샘플링(20))로 반환

def oversample2(data, p_level, n=1, seed=0):    
    
    np.random.seed(seed)
    
    cust_ids = data['cust_id'].unique().tolist() 
    
    customerProducts = []
    
    for cust_id in cust_ids:
        
        productLst = data.query(f'cust_id=={cust_id}')[p_level].tolist()
        
        for j in range(n):
   
            productLst = list(np.append(productLst, np.random.choice(productLst, len(productLst) * n, replace=True)))
        
        customerProducts.append(productLst)
    
    return customerProducts

# 중분류 구매목록
print('중분류 구매목록 뽑는중')
X_train_corpus_nm = oversample2(df_train,'gds_grp_nm', 2)
X_test_corpus_nm = oversample2(df_test,'gds_grp_nm', 2)

# 중분류 구매목록
print('중분류 구매목록 뽑는중')
X_train_corpus_mclas_nm = oversample2(df_train,'gds_grp_mclas_nm', 2)
X_test_corpus_mclas_nm = oversample2(df_test,'gds_grp_mclas_nm', 2)

### Training the Word2Vec model
num_features = 300 # 단어 벡터 차원 수
min_word_count = 1 # 최소 단어 수
context = 10 # 학습 윈도우(인접한 단어 리스트) 크기

# 중분류
print('중분류 학습중')
w2v_nm = word2vec.Word2Vec(X_train_corpus_nm,                  # 학습시킬 단어 리스트
                        size=num_features,        # 단어 벡터의 차원 수
                        window=context,           # 주변 단어(window)는 안뒤로 몇개까지 볼 것인지
                        min_count=min_word_count, # 단어 리스트에서 출현 빈도가 몇번 미만인 단어는 분석에서 제외해라
                        workers=4,                # cpu는 쿼드코어를 써라 n_jobs=-1과 같음
                        sg = 1,                   # CBOW와 skip-gram중 후자를 선택
                        iter=7,
                        seed=0)

# 필요없는 메모리 unload
w2v_nm.init_sims(replace=True)



# 대분류
print('대분류 학습중')
w2v_mclas_nm = word2vec.Word2Vec(X_train_corpus_mclas_nm,                  # 학습시킬 단어 리스트
                        size=num_features,        # 단어 벡터의 차원 수
                        window=context,           # 주변 단어(window)는 안뒤로 몇개까지 볼 것인지
                        min_count=min_word_count, # 단어 리스트에서 출현 빈도가 몇번 미만인 단어는 분석에서 제외해라
                        workers=4,                # cpu는 쿼드코어를 써라 n_jobs=-1과 같음
                        sg = 1,                   # CBOW와 skip-gram중 후자를 선택
                        iter=7,
                        seed=0)

# 필요없는 메모리 unload
w2v_mclas_nm.init_sims(replace=True)

### Make features
# 구매상품에 해당하는 벡터의 평균/최소/최대 벡터를 feature로 만드는 전처리기(pipeline에서 사용 가능)
class EmbeddingVectorizer_nm(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = num_features
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.hstack([
                np.std([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.var([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),          
            ]) 
            for words in X
        ])

### Make features
# 구매상품에 해당하는 벡터의 평균/최소/최대 벡터를 feature로 만드는 전처리기(pipeline에서 사용 가능)
class EmbeddingVectorizer_mclas(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = num_features
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.hstack([
                np.max([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),               
            ]) 
            for words in X
        ])

# EmbeddingVectorizer 클래스를 통해서 피쳐뽑기(std, max, var -> 600개 피쳐나옴)

# 중분류
Vectorizer = EmbeddingVectorizer_nm(w2v_nm.wv)
Vectorizer.fit(X_train_corpus_nm, y_train)

X_train_nm = pd.DataFrame(Vectorizer.transform(X_train_corpus_nm))
X_test_nm = pd.DataFrame(Vectorizer.transform(X_test_corpus_nm))

display(X_train_nm)


# EmbeddingVectorizer 클래스를 통해서 피쳐뽑기(std, max, var -> 600개 피쳐나옴)

# 중분류
Vectorizer = EmbeddingVectorizer_mclas(w2v_mclas_nm.wv)
Vectorizer.fit(X_train_corpus_mclas_nm, y_train)

X_train_mclas = pd.DataFrame(Vectorizer.transform(X_train_corpus_mclas_nm))
X_test_mclas = pd.DataFrame(Vectorizer.transform(X_test_corpus_mclas_nm))

display(X_train_mclas)

X_train_mix =  pd.concat([X_train_nm, X_train_mclas], axis=1)
X_test_mix = pd.concat([X_test_nm, X_test_mclas], axis=1)



### Read data
# y_train에서 .gender 지워준거 말고 다 같음

df_train = pd.read_csv('X_train.csv', encoding='cp949')
df_test = pd.read_csv('X_test.csv', encoding='cp949')
y_train = pd.read_csv('y_train.csv')
IDtest = df_test.cust_id.unique()


y_train.gender = y_train.gender.astype(str)

tr = pd.merge(df_train, y_train, on='cust_id')
tr2 = pd.concat([df_train, df_test])
y_train = y_train.gender

skf = StratifiedKFold(n_splits=4 , shuffle=True, random_state=0)

# 말뭉치(corpus) 제작전 뽑기

before_corpus_goods = []
before_corpus_nm = []
before_corpus_mclas = []

for i in range(len(tr)):
    
    goods = tr.loc[i, 'goods_id']
    nm = tr.loc[i, 'gds_grp_nm']
    mclas = tr.loc[i, 'gds_grp_mclas_nm']
    gender = tr.loc[i, 'gender']
    
    before_corpus_goods.append([goods, gender])
    before_corpus_nm.append([nm, gender])
    before_corpus_mclas.append([mclas, gender])
    
# 말뭉치 만들 칼럼 넣기
tr['before_corpus_goods'] = before_corpus_goods
tr['before_corpus_nm'] = before_corpus_nm
tr['before_corpus_mclas'] = before_corpus_mclas

# 말뭉치 칼럼 제작, 말뭉치 단어 몇개들어가 있는지

corpus_df = pd.DataFrame(tr.groupby('cust_id')['before_corpus_goods'].agg(lambda x: [j for i in x for j in i]))
corpus_df['before_corpus_nm'] = tr.groupby('cust_id')['before_corpus_nm'].agg(lambda x: [j for i in x for j in i])
corpus_df['before_corpus_mclas'] = tr.groupby('cust_id')['before_corpus_mclas'].agg(lambda x: [j for i in x for j in i])

display(corpus_df)

corpus1 = corpus_df['before_corpus_goods']
corpus2 = corpus_df['before_corpus_nm']
corpus3 = corpus_df['before_corpus_mclas']


def oversample3(data, n=1, seed=0):    
    
    np.random.seed(seed)
    
    customerProducts = []
    
    for cor in data:
            
        cor = list(np.append(cor, np.random.choice(cor, len(cor) * n, replace=True)))
        
        customerProducts.append(cor)
    
    return customerProducts

corpus1_oversample = oversample3(corpus1, n=20, seed=0)
corpus2_oversample = oversample3(corpus2, n=20, seed=0)
corpus3_oversample = oversample3(corpus3, n=20, seed=0)

# W2V 학습

num_features = 20 # 문자 벡터 차원 수
min_word_count = 0 # 최소 문자 수
num_workers = 4 # 병렬 처리 스레드 수
context = 3 # 문자열 창 크기

print('첫번째 w2v모델 학습 진행')
wv_model1 = word2vec.Word2Vec(corpus1_oversample, 
                          workers=num_workers, 
                          size=num_features, 
                          min_count=min_word_count,
                          window=context)
print('두번째 w2v모델 학습 진행')
wv_model2 = word2vec.Word2Vec(corpus2_oversample, 
                          workers=num_workers, 
                          size=num_features, 
                          min_count=min_word_count,
                          window=context)
print('세번째 w2v모델 학습 진행')
wv_model3 = word2vec.Word2Vec(corpus3_oversample, 
                          workers=num_workers, 
                          size=num_features, 
                          min_count=min_word_count,
                          window=context)




## goods_id

def get_0_similarity_goods(product):
    try:
        sim = wv_model1.similarity('0', f'{product}')
    except:
        sim = np.nan
    return sim

def get_1_similarity_goods(product):
    try:
        sim = wv_model1.similarity('1', f'{product}')
    except:
        sim = np.nan
    return sim

## gds_grp_nm

def get_0_similarity_nm(product):
    try:
        sim = wv_model2.similarity('0', f'{product}')
    except:
        sim = np.nan
    return sim

def get_1_similarity_nm(product):
    try:
        sim = wv_model2.similarity('1', f'{product}')
    except:
        sim = np.nan
    return sim

## gds_grp_mclas_nm

def get_0_similarity_mclas(product):
    try:
        sim = wv_model3.similarity('0', f'{product}')
    except:
        sim = np.nan
    return sim

def get_1_similarity_mclas(product):
    try:
        sim = wv_model3.similarity('1', f'{product}')
    except:
        sim = np.nan
    return sim



tr2['goods_0_similarity'] = tr2['goods_id'].apply(get_0_similarity_goods)
tr2['goods_1_similarity'] = tr2['goods_id'].apply(get_1_similarity_goods)
tr2['nm_0_similarity'] = tr2['gds_grp_nm'].apply(get_0_similarity_nm)
tr2['nm_1_similarity'] = tr2['gds_grp_nm'].apply(get_1_similarity_nm)
tr2['mclas_0_similarity'] = tr2['gds_grp_mclas_nm'].apply(get_0_similarity_mclas)
tr2['mclas_1_similarity'] = tr2['gds_grp_mclas_nm'].apply(get_1_similarity_mclas)


train_test = pd.DataFrame({'cust_id' : range(5982)}).set_index('cust_id')

train_test['goods_0_similarity'] = tr2.groupby('cust_id')['goods_0_similarity'].apply(lambda x: np.nanmean(x))
train_test['goods_1_similarity'] = tr2.groupby('cust_id')['goods_1_similarity'].apply(lambda x: np.nanmean(x))
train_test['nm_0_similarity'] = tr2.groupby('cust_id')['nm_0_similarity'].apply(lambda x: np.nanmean(x))
train_test['nm_1_similarity'] = tr2.groupby('cust_id')['nm_1_similarity'].apply(lambda x: np.nanmean(x))
train_test['mclas_0_similarity'] = tr2.groupby('cust_id')['mclas_0_similarity'].apply(lambda x: np.nanmean(x))
train_test['mclas_1_similarity'] = tr2.groupby('cust_id')['mclas_1_similarity'].apply(lambda x: np.nanmean(x))


X_train_w2v = train_test[:3500]
X_test_w2v = train_test[3500:]


X_train_experiment = pd.concat([mix, X_train_w2v], axis=1)
X_train_experiment

X_test_experiment = pd.concat([X_test_mix, X_test_w2v], axis=1)
X_test_experiment = X_test_experiment.drop('cust_id', axis=1)
X_test_experiment

logreg = LogisticRegression(random_state=0, n_jobs=-1)
lgbm = LGBMClassifier(n_jobs=-1, random_state=0)
rf = RandomForestClassifier(random_state=0, n_jobs=-1)

models = [logreg, lgbm, rf]

for model in models:
    
    cv_scores = cross_val_score(model, X_train_experiment, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    print(model.__class__.__name__,cv_scores)
    print('최대성능 :', max(cv_scores))
    print('평균성능 :', np.mean(cv_scores))

X_train_experiment.to_csv('X_train_corpus_nm&mclas.csv', index=False, encoding='cp949')
X_test_experiment.to_csv('X_test_corpus_nm&mclas.csv', index=False, encoding='cp949')


##  Modeling

X_train = pd.read_csv('X_train_corpus_nm&mclas.csv', encoding='cp949')
y_train = pd.read_csv('y_train.csv').gender
X_test = pd.read_csv('X_test_corpus_nm&mclas.csv', encoding='cp949')

logreg = LogisticRegression(random_state=0, n_jobs=-1)
rf = RandomForestClassifier(random_state=0, n_jobs=-1)
gbm = GradientBoostingClassifier(random_state=0)
lgbm = LGBMClassifier(random_state=0 ,n_jobs=-1)

models = [logreg, rf, gbm, lgbm]

models = [logreg]


# 6개의 모델을 이용해서 가장 잘나온 p를 뽑을 것임
for model in models:
    
    cv_scores = []
    
    # 퍼센타일을 5~100프로 모두 살피기 <- 처음에만 100프로 찍고 이후 조절하기
    for percentile in tqdm(range(5,100)):
    
        X_new = SelectPercentile(percentile = percentile).fit_transform(X_train,y_train)
       
        # cross_val_score 4번의 평균값 (정수시 skf로 자동으로 들어간다)
        cv_score = cross_val_score(model, X_new, y_train, scoring='roc_auc', cv=skf).mean()
        
        cv_scores.append((percentile, cv_score))
        
    # 베스트 percentile과 점수 출력
    best_score = cv_scores[np.argmax([score for _, score in cv_scores])]
    print(model.__class__.__name__, best_score)
    
    # 모델별 percentile에 따른 성능 그림
    plt.plot([p for p,_ in cv_scores], [score for _, score in cv_scores])
    plt.xlabel('Percent of features')
    plt.legend(loc=0)
    plt.grid()

X_test = X_test.fillna({'goods_0_similarity':np.mean(X_test['goods_0_similarity']), 'goods_1_similarity':np.mean(X_test['goods_1_similarity'])})

select_p = SelectPercentile(percentile= 22).fit(X_train, y_train)
X_train = select_p.transform(X_train)
X_test = select_p.transform(X_test)

pd.DataFrame(X_train).to_csv('X_train_after_percentile_nm&mclas.csv',index=False,encoding='cp949')
pd.DataFrame(X_test).to_csv('X_test_after_percentile_nm&mclas.csv',index=False,encoding='cp949')

X_train = pd.read_csv('X_train_after_percentile_nm&mclas.csv', encoding='cp949')
X_test = pd.read_csv('X_test_after_percentile_nm&mclas.csv', encoding='cp949')

logreg = LogisticRegression(random_state=0, n_jobs=-1)
rf = RandomForestClassifier(random_state=0, n_jobs=-1)
gbm = GradientBoostingClassifier(random_state=0)
lgbm = LGBMClassifier(random_state=0, n_jobs=-1)

models = [logreg, rf, gbm, lgbm]

# 모델별로 총 8개의 모델로 기본성능을 보고 평균적인 성능을 예측한다.

for model in models:
    
    lucky_seed = [2016, 2533]
    
    cv_scores = []
    
    for rs in lucky_seed:
        
        skf = StratifiedKFold(n_splits=4 , shuffle=True, random_state=rs)
        
        scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv = skf)
        
        cv_scores.append(scores)
        
    print(f'{model.__class__.__name__}, \ncv성능들: {cv_scores}\n최고성능: {max([score for scoreArr in cv_scores for score in scoreArr])}\n평균성능: {np.mean(cv_scores)}\n')

##  Ensemble

# 튜닝 진행하며 미리 init_points=20, n_iter=10 으로 때려보고 파라미터 조절해가며 최고성능 더 끌어내기
# 파라미터 조절이후 지속적으로 안정적인 성능이 나오면 50,50 안때려도 됨. 50,50은 파라미터 튜닝 귀찮을때

BO_tuned_clfs = []

### Basian LR

# 하이퍼 파라미터 범위

pbounds = { 'C': (.05,.7),}


def logreg_opt(C):
    
    params = {
        'C' : C
    }

    logreg = LogisticRegression(**params, n_jobs=-1, random_state=50)
    
    skf = StratifiedKFold(n_splits=4 , shuffle=False, random_state=50)
    
    score = cross_val_score(logreg, X_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)
    
    return np.mean(score)


BO_logreg = BayesianOptimization(f = logreg_opt, pbounds = pbounds, random_state=0)

BO_logreg.maximize(init_points=40, n_iter=10)

# BO_rf.res  # 모든 성능 들어가있음
BO_logreg.max

max_params = BO_logreg.max['params']

max_params


logreg_clf = LogisticRegression(**max_params,  n_jobs=-1, random_state=50)

scores = cross_val_score(logreg_clf, X_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)

print(scores)
print(f'최대성능: {max(scores)}\n평균성능: {np.mean(scores)}')

BO_tuned_clfs.append((logreg_clf.__class__.__name__, logreg_clf, max(scores)))

### Basian RF

# 하이퍼 파라미터 범위

pbounds = { 'n_estimators': (200,260),
            'max_depth': (5,15), 
            'max_features': (0.8,0.95),
            'min_samples_leaf': (1, 5)}

def rf_opt(n_estimators, max_depth, max_features, min_samples_leaf):
    
    params = {
        'n_estimators' : int(round(n_estimators)),
        'max_depth' : int(round(max_depth)),
        'min_samples_leaf' : int(round(min_samples_leaf))
    }

    rf = RandomForestClassifier(**params, n_jobs=-1, random_state=50)
    
    skf = StratifiedKFold(n_splits=4 , shuffle=False, random_state=50)
    
    score = cross_val_score(rf, X_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)
    
    return np.mean(score)


BO_rf = BayesianOptimization(f = rf_opt, pbounds = pbounds, random_state=0)

BO_rf.maximize(init_points=20, n_iter=10)

# BO_rf.res  # 모든 성능 들어가있음
BO_rf.max

max_params = BO_rf.max['params']

max_params['n_estimators'] = int(round(max_params['n_estimators']))
max_params['max_depth'] = int(round(max_params['max_depth']))
max_params['min_samples_leaf'] = int(round(max_params['min_samples_leaf']))

max_params

rf_clf = RandomForestClassifier(**max_params,  n_jobs=-1, random_state=50)

scores = cross_val_score(rf_clf, X_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)

print(scores)
print(f'최대성능: {max(scores)}\n평균성능: {np.mean(scores)}')

BO_tuned_clfs.append((rf_clf.__class__.__name__, rf_clf, max(scores)))

### Basian LGBM

pbounds = { 'learning_rate': (0.05, 1.5),
            'n_estimators': (90, 150),
            'max_depth': (3,10),   
            'subsample': (0.8,0.95), 
            'colsample_bytree': (0.75,0.9),   
            'num_leaves': (2,10),
            'min_child_weight': (1, 7)}


def lgbm_opt(learning_rate, n_estimators, max_depth, subsample, colsample_bytree, num_leaves, min_child_weight):

    params = {
        'learning_rate': learning_rate,
        'n_estimators' : int(round(n_estimators)),
        'max_depth' : int(round(max_depth)),
        'subsample': subsample,
        'colsample_bytree' : colsample_bytree,
        'num_leaves' : int(round(num_leaves)),
        'min_child_weight' : min_child_weight,
        'n_jobs' : -1
    }
    
    lgbm = LGBMClassifier(**params)
    
    skf = StratifiedKFold(n_splits=4 , shuffle=False, random_state=50)

    score = cross_val_score(lgbm, X_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)
    
    return np.mean(score)

BO_lgbm = BayesianOptimization(f = lgbm_opt, pbounds = pbounds, random_state=1)    


BO_lgbm.maximize(init_points=20, n_iter=10)

# BO_rf.res  # 모든 성능 들어가있음
BO_lgbm.max

max_params = BO_lgbm.max['params']

max_params['n_estimators'] = int(round(max_params['n_estimators']))
max_params['max_depth'] = int(round(max_params['max_depth']))
max_params['num_leaves'] = int(round(max_params['num_leaves']))

max_params

lgbm_clf = LGBMClassifier(**max_params)

scores = cross_val_score(lgbm_clf, X_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)

print(scores)
print(f'최대성능: {max(scores)}\n평균성능: {np.mean(scores)}')

BO_tuned_clfs.append((lgbm_clf.__class__.__name__, lgbm_clf, max(scores)))

### Basian GB

pbounds = { 'learning_rate': (0.05, 1.5),
            'n_estimators': (50, 250),
            'max_depth': (3,10),   
            'subsample': (0.8,0.95), 
            'min_samples_split': (2,5),   
            'min_samples_leaf': (1,5)}


def gb_opt(learning_rate, n_estimators, max_depth, subsample, min_samples_split, min_samples_leaf):

    params = {
        'learning_rate': learning_rate,
        'n_estimators' : int(round(n_estimators)),
        'max_depth' : int(round(max_depth)),
        'subsample': subsample,
        'min_samples_split' : int(round(min_samples_split)),
        'min_samples_leaf' : int(round(min_samples_leaf))
    }
    
    gb = GradientBoostingClassifier(**params)
    
    skf = StratifiedKFold(n_splits=4 , shuffle=False, random_state=50)

    score = cross_val_score(gb, X_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)
    
    return np.mean(score)

BO_gb = BayesianOptimization(f = gb_opt, pbounds = pbounds, random_state=0)    


BO_gb.maximize(init_points=20, n_iter=10)

# BO_rf.res  # 모든 성능 들어가있음
BO_gb.max

max_params = BO_gb.max['params']

max_params['n_estimators'] = int(round(max_params['n_estimators']))
max_params['max_depth'] = int(round(max_params['max_depth']))
max_params['min_samples_leaf'] = int(round(max_params['min_samples_leaf']))
max_params['min_samples_split'] = int(round(max_params['min_samples_split']))

max_params

gb_clf = GradientBoostingClassifier(**max_params)

scores = cross_val_score(gb_clf, X_train, y_train, scoring='roc_auc', cv=4, n_jobs=-1)

print(scores)
print(f'최대성능: {max(scores)}\n평균성능: {np.mean(scores)}')

BO_tuned_clfs.append((gb_clf.__class__.__name__, gb_clf, max(scores)))

gb_clf = gbm = GradientBoostingClassifier(random_state=0)

scores = cross_val_score(gb_clf, X_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)

print(scores)
print(f'최대성능: {max(scores)}\n평균성능: {np.mean(scores)}')

BO_tuned_clfs.append((gb_clf.__class__.__name__, gb_clf, max(scores)))

BO_tuned_clfs

len(BO_tuned_clfs)


### *Ensemble Stacking*

stack_estimators = estimators

len(stack_estimators)

#- S_train, S_test transform

S_train, S_test = stacking(stack_estimators,
                           X_train, y_train, X_test,
                           regression=False, needs_proba=True, metric=None, n_folds=5, stratified=True, shuffle=True,
                           random_state=0, verbose=0)

# Stacking - Meta_Model: LogReg

# -> 피쳐별로 파라미터 범위 간단한 튜닝 필요할 수 있음

pbounds = { 'C': (0.05,0.8),}


def logreg_meta(C):
    
    params = {
        'C' : C
    }

    logreg = LogisticRegression(**params, n_jobs=-1, random_state=50)
    
    skf = StratifiedKFold(n_splits=4 , shuffle=False, random_state=50)
    
    score = cross_val_score(logreg, S_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)
    
    return np.mean(score)


BO_logreg = BayesianOptimization(f = logreg_meta, pbounds = pbounds, random_state=0)

BO_logreg.maximize(init_points=40, n_iter=20)


# BO_rf.res  # 모든 성능 들어가있음
BO_logreg.max

max_params = BO_logreg.max['params']

max_params


logreg_meta = LogisticRegression(**max_params,  n_jobs=-1, random_state=50)

scores = cross_val_score(logreg_meta, S_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)

print(scores)
print(f'최대성능: {max(scores)}\n평균성능: {np.mean(scores)}')

S_train = pd.DataFrame(S_train)

model = logreg_meta
  
for train_index, test_index in skf.split(S_train, y_train):
    
    x_tra, x_val = S_train.iloc[train_index], S_train.iloc[test_index]
    y_tra, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
    
    pred = model.fit(x_tra, y_tra).predict_proba(x_val)[:,1]
    
    print(roc_auc_score(y_val, pred))
        
print(model.__class__.__name__)
    
stack_prediction = model.predict_proba(S_test)[:,1]


# 서브미션 파일 이름 조정

# stacking emsemble result

pd.DataFrame({'cust_id':np.arange(3500,5982), 'gender':stack_prediction}).set_index('cust_id').to_csv('submission_nm&mclas_stack.csv', index=True, encoding='cp949')
