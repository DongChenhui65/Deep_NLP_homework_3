import os
import math
import jieba
import logging
from nltk import FreqDist
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

corpus = []
res = []
split_words = []
datapath = "./jyxstxtqj_downcc.com"
filelist = os.listdir(datapath)


def delet_stopwords(word_list):
    stopwords_file_path = './/cn_stopwords.txt'
    stopword_file = open(stopwords_file_path, "r", encoding='utf-8')
    stop_words = stopword_file.read().split('\n')
    stopword_file.close()

    for word in stop_words:
        word_list = list(filter(lambda x: x != word, word_list))  # 删除所有的
    return word_list


for filename in filelist:
    corpus = []
    res = []
    filepath = datapath + '/' + filename
    with open(filepath, "r", encoding="gb18030") as file:
        filecontext = file.read()
        filecontext = filecontext.replace(
            "本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
        filecontext = filecontext.replace("本书来自www.cr173.com免费txt小说下载站", '')
        corpus.append(filecontext)
        file.close()

    for filecontext in corpus:
        filecontext = filecontext.replace('\n', '')  # 去除换行符
        filecontext = filecontext.replace(' ', '')  # 去除空格
        filecontext = filecontext.replace('\u3000', '')  # 去除全角空白符
        if filecontext != '\n':
            res.append(filecontext.strip())

    for line in res:
        for x in jieba.lcut(line):
            split_words.append(x)

split_words = delet_stopwords(split_words)
result = ' '.join(split_words)
with open('./split_words.txt', 'w', encoding="utf-8") as f2:
    f2.write(result)

sentences = LineSentence('./split_words.txt')
model = Word2Vec(sentences=sentences, vector_size=200, min_count=10, window=5, sg=1, workers=4, epochs=50)
model.save('./all_skip_gram.model')

