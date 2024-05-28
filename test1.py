from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# sentences = LineSentence('./split_words.txt')
# model = Word2Vec(sentences=sentences, vector_size=200, min_count=10, window=5, sg=1, workers=4, epochs=50)
# model.save('./all_skip_gram.model')

model = Word2Vec.load('./all_skip_gram.model')#加载模型，由generate_model.py生成

'''查询词之间的相似度'''
word1 = "杨过"
word2 = "小龙女"
similarity_score = model.wv.similarity(word1, word2)
print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")
# print(model.wv.most_similar('杨过', topn=10))

word1 = "东方不败"
word2 = "韦小宝"
similarity_score = model.wv.similarity(word1, word2)
print(f"词语 '{word1}' 和 '{word2}' 的相似度得分为：{similarity_score}")


# print(model.wv.most_similar('东方不败', topn=10))
# print(model.wv.most_similar('张无忌', topn=10))
# print(model.wv.similarity('张无忌', '周芷若'))
# print(model.wv.similarity('张无忌', '赵敏'))


# a = model.wv.index_to_key
# print(model.wv.index_to_key)


# vector2 = model.wv["杨过"]
# print(vector2)

