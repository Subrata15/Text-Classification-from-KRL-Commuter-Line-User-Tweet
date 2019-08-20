import tweepy
import pickle
import csv
import pandas as pd
import datetime

consumer_key = 'xxx'
consumer_secret = 'xxx'
access_token = 'xxx'
access_token_secret = 'xxx'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

hari_ini =datetime.date.today()
#print(hari_ini)

def get_data(c):
    global df_n, db
    text1 = []
    text2 = []
    text3 = []
    data_base = []
    #get 30 update tweet
    for tweet in tweepy.Cursor(api.user_timeline,id=c,
                           lang="id", tweet_mode='extended').items(30):
           text1.append(tweet.full_text)
           text2.append('Tweet' + c )
           text3.append(tweet.created_at)
           data_base.append(tweet)
    #print(text4)
    db = pd.DataFrame(data_base)
    d = {'col1': text2, 'col2': text1, 'text':text1, 'date':text3}
    df_n = pd.DataFrame(d)
    df_n['txt'] = df_n['col1']+' '+df_n['col2']
    df_n = df_n.drop(['col1', 'col2'], axis=1)
    return
get_data('@CurhatKRL')
df1 = df_n
get_data('@krlmania')
df2 = df_n
#db bisa di query kan ke mySQL

#combine df1 and df2
frames = [df1, df2]
df = pd.concat(frames).reset_index(drop=True)
df['date'] = df['date'].dt.date

#CLEANING TWEET
import re
import Sastrawi as sts
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
stop_word = StopWordRemoverFactory().get_stop_words()

more_stopword = ['yg', 'ajah','iya','mba', 'mas', 'kak', 'pak', 'pahi', 'mah', 'muehehe', 'men', 'kehfine', 'alhamdulilah', 'alhamdulillah', 
                 'nih','om', 'selamat', 'sama', 'sabar', 'gak', 'yak', 'semoga' 'bu', 'adik', 'omen', 'tumben', 'tp', 'sy', 'kmu', 'jg', 'kyk', 'dll']
d_sword = stop_word+more_stopword
dictionary = ArrayDictionary(d_sword)
swr = StopWordRemover(dictionary)

pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
pat3 = '(RT)'
combined_pat = r'|'.join((pat1, pat2, pat3))
df_t = df['text']

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    #filtered_words = [stop_word.remove(words)] 
    return (" ".join(words)).strip()

clean_tweet_texts = []
for i in range(len(df_t)):
    clean_tweet_texts.append(tweet_cleaner(df_t[i]))

clean_tweet = []
for i in range(len(clean_tweet_texts)):
    clean_tweet.append(swr.remove(clean_tweet_texts[i]))

#load vocabulary 
df['text'] = clean_tweet
v_pkl = 'df_text.pkl'
v_open = open(v_pkl, 'rb')
vocab = pickle.load(v_open)

#IMport TFid Vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
Tfidf = TfidfVectorizer()

#vector vocabulary
vector = Tfidf.fit_transform(vocab)

#vector prediski
vect_pred = Tfidf.transform(df['text'])

#load the pre tained model
s_model = 'model_NLP_Rute_1.pkl'
s_open = open(s_model, 'rb')
model =  pickle.load(s_open)

#prediction
label = model.predict(vect_pred)
df['label'] = label
#print(df.head())

index_nu = 0
for i in range(len(df['txt'])):
    #print the text tweet if the tweet is today
    if (df.iloc[index_nu][0]==hari_ini) and (df.iloc[index_nu][3]==1):
       print('Info Curhat Breaking Hari ini via : ')
       print(df.iloc[index_nu][2]) #optional 
    index_nu += 1
    #print('done!')

