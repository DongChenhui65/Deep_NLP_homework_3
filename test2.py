from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
sentences = LineSentence('./split_words.txt')
model = Word2Vec(sentences=sentences, vector_size=200, min_count=10, window=5, sg=1, workers=4, epochs=10)
model.save('./all_skip_gram_10.model')
model = Word2Vec.load('./all_skip_gram_10.model')#加载模型，由generate_model.py生成

'''KMeans聚类'''
# 获取词汇表中的所有词语及其向量
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])

# 使用KMeans算法进行聚类
num_clusters = 10  # 设定要分的簇数
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)

# 获取每个词语对应的簇标签
labels = kmeans.labels_

plt.figure(figsize=(8, 8))
# 使用颜色映射
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, num_clusters))  # 生成10种不同的颜色
for i in range(num_clusters):
    # 获取 Cluster 1 的词向量和索引
    cluster_index = i
    cluster_indices = [index for index, label in enumerate(labels) if label == cluster_index]
    cluster_words = [words[index] for index in cluster_indices]
    cluster_vectors = word_vectors[cluster_indices]

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=0)
    if len(cluster_vectors) > 30:
        cluster_vectors_2d = tsne.fit_transform(cluster_vectors)
        # 绘制散点图
        plt.scatter(cluster_vectors_2d[:, 0], cluster_vectors_2d[:, 1], color=colors[i], s=1, alpha=0.5, label=f'Cluster {cluster_index}')
plt.legend()
plt.title('Word Clustering Visualization for Cluster')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# 打印每个簇中的词语
clusters = {}
for i in range(num_clusters):
    clusters[i] = []

for word, label in zip(words, labels):
    clusters[label].append(word)

for cluster_id, cluster_words in clusters.items():
    print(f"Cluster {cluster_id}: {', '.join(cluster_words)}")


