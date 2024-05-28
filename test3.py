# -*- coding: utf-8 -*-
import os
import jieba
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def conduct_para_and_get_para_vector(paragraph, Model):
    # 预处理
    paragraph = paragraph.replace('\n', '')  # 去除换行符
    paragraph = paragraph.replace(' ', '')  # 去除空格
    paragraph = paragraph.replace('\u3000', '')  # 去除全角空白符
    # 停用词表
    stopwords_file_path = './/cn_stopwords.txt'
    stopword_file = open(stopwords_file_path, "r", encoding='utf-8')
    stop_words = stopword_file.read().split('\n')
    stopword_file.close()
    # 分词并去除停用词
    words = [word for word in jieba.lcut(paragraph) if word not in stop_words]
    # 计算词向量
    word_vectors = [Model.wv[word] for word in words if word in Model.wv]
    # 取平均值作为段落向量
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(Model.vector_size)


model = Word2Vec.load('./all_skip_gram.model')  # 加载模型，由generate_model.py生成
paragraph1 = "两人在乡间躲了三日，听得四乡饥民聚众要攻漳州、厦门。这一来，只将张朝唐吓得满腔雄心，登化乌有，眼见危邦不可居，还是急速回家的为是。其时厦门已不能再去，主仆两人一商量，决定从陆路西赴广州，再乘海船出洋。两人买了两匹坐骑，胆战心惊，沿路打听，向广东而去。幸喜一路无事，经南靖、平和，来到三河坝，已是广东省境，再过梅县、水口，向西迤逦行来。张朝唐素闻广东是富庶之地，但沿途所见，尽是饥民，心想中华地大物博，百姓人人生死系于一线，渤泥只是海外小邦，男女老幼却是安居乐业，无忧无虑，不由得大是叹息，心想中国山川雄奇，眼见者百未得一，但如此朝不保夕，还是去渤泥椰子树下唱歌睡觉安乐得多了。这一日行经鸿图嶂，山道崎岖，天色渐晚，他心中焦急起来，催马急奔。一口气奔出十多里地，到了一个小市镇上，主仆两人大喜，想找个客店借宿，哪知道市镇上静悄悄的一个人影也无。张康下马，走到一家挂着“粤东客栈”招牌的客店之外，高声叫道：“喂，店家，店家！”店房靠山，山谷响应，只听见“喂，店家，店家”的回声，店里却毫无动静。正在这时，一阵北风吹来，猎猎作响，两人都感毛骨悚然。张朝唐拔出佩剑，闯进店去，只见院子内地下倒着两具尸首，流了一大滩黑血，苍蝇绕着尸首乱飞。腐臭扑鼻，看来死者已死去多日。张康一声大叫，转身逃出店去。张朝唐四下一瞧，到处箱笼散乱，门窗残破，似经盗匪洗劫。张康见主人不出来，一步一顿的又回进店去。张朝唐道：“到别处看看。”哪知又去了三家店铺，家家都是如此。有的女尸身子赤裸，显是曾遭强暴而后被杀。一座市镇之中，到处阴风惨惨，尸臭阵阵。两人再也不敢停留，急忙上马向西。主仆两人行了十几里，天色全黑，又饿又怕，正狼狈间，张康忽道：“公子，你瞧！”张朝唐顺着他手指看去，只见远处有一点火光，喜道：“咱们借宿去。"
paragraph2 = "　　令狐冲只拣荒僻的小路飞奔，到了一处无人的山野，显是离杭州城已远。他如此迅捷飞奔，停下来时竟既不疲累，也不气喘，比之受伤之前，似乎功力尚有胜过。他除下头上罩子，听到淙淙水声，口中正渴，当下循声过去，来到一条山溪之畔，正要俯身去捧水喝，水中映出一个人来，头发篷松，满脸污秽，神情甚是丑怪。令狐冲吃了一惊，随即哑然一笑，囚居数月，从不梳洗，自然是如此龌龊了，霎时间只觉全身奇痒，当下除去外袍，跳在溪水中好好洗了个澡，心想：“身上的老泥便没半担，也会有三十斤。”浑身上下擦洗干净，喝饱清水后，将头发挽在头顶，水中一照，已回复了本来面目，与那满脸浮肿的风二中已没半点相似之处。穿衣之际，觉得胸腹间气血不畅，当下在溪边行功片刻，便觉丹田中的内急已散入奇经八脉，丹田内又是如竹之空、似谷之虚，而全身振奋，说不出的畅快。他不知自己已练成了当世第一等厉害功夫，桃谷六仙和不戒和尚的七道真气，在少林寺疗伤时方生大师注入他体内的内力，固然已尽皆化为己有，而适才抓住黑白子的手腕，又已将他毕生修习的内功吸了过来贮入丹田，再散入奇经八脉，那便是又多了一个高手的功力，自是精神大振。"
vector1 = conduct_para_and_get_para_vector(paragraph1, model)
vector2 = conduct_para_and_get_para_vector(paragraph2, model)
# 计算段落相似度
if np.all(vector1 == 0) or np.all(vector2 == 0):
    paragraph_similarity_score = 0  # 若有一个段落无有效向量，则相似度为0
else:
    paragraph_similarity_score = cosine_similarity([vector1], [vector2])[0][0]
print(f"段落间的语义相似度：{paragraph_similarity_score}")
