import os
import time
import jieba
import math


#读标点
def read_punctuation_list(path):
    punctuation = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    punctuation.extend(['\n', '\u3000', '\u0020', '\u00A0'])
    return punctuation
#读停词
def read_stopwords_list(path):
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords


# 读取文件夹文件
def read_data(path):
    data_txt = []
    files = os.listdir(path)  # 返回指定的文件夹包含的文件列表
    for file in files:
        position = path + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符

        with open(position, 'r', encoding='gb18030') as f:
            data = f.read()
            data_txt.append(data)
        f.close()
    return data_txt, files


# 模型语料库预处理
def preprocess():
    sentences = []
    line = '' 
    path = "./novel"
    data_txt,filenames = read_data(path)

    for file in filenames:  # 遍历文件夹
        position = path + '\\' + file  # 构造绝对路径，"\\"，其中一个'\'为转义符

        stopwords = read_stopwords_list("./cn_stopwords.txt")
        punctuation= read_punctuation_list("./cn_punctuation.txt")      

        with open(position, "r", encoding='gb18030') as f:
            txt = f.read()
            txt = txt.replace('本书来自www.cr173.com免费txt小说下载站', '')
            txt = txt.replace('更多更新免费电子书请关注www.cr173.com', '')

            for w in txt:
                if w in punctuation and line != '\n' or w in stopwords and line != '\n':
                    if line.strip() != '':
                        sentences.append(line.strip())
                        line = ''
                elif w not in punctuation and w not in stopwords:
                    line += w
            with open("./sentence//"+file, "a",encoding='gb18030') as p:
                p.truncate(0)
                for sentence in sentences:
                    p.write(str(sentence)+'\n')

        sentences = []

#一元模型获取词频率
def get_tf(words):
    tf_dic = {}
    for i in range(len(words)-1):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1
        # print(tf_dic.get(w,0))
    return tf_dic

#二元模型获取词频率
def get_bigram_tf(words):
    tf_dic = {}
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1
        # print(tf_dic.get(w,0))
    return tf_dic

#三元元模型获取词频率
#非句子末尾二元词频统计
def get_bi_tf(words):
    tf_dic = {}
    for i in range(len(words)-2):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1
    return tf_dic    
# 三元模型词频统计
def get_trigram_tf( words):
    tf_dic = {}
    for i in range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1
    return tf_dic    


if __name__ == '__main__':
    before = time.time()

    # preprocess()
    
    path = "./sentence"
    data_txt,filenames = read_data(path)
    for file in filenames: 
        #计算词信息熵
        with open(path + '\\' + file, 'r', encoding='gb18030') as f:     
            split_words = []
            count = 0
            corpus = []
            words_len = 0
            #按行读取 并且分词
            for line in f:
                if line != '\n':
                    corpus.append(line.strip())
                    count += len(line.strip())
            for line in corpus:        
                for x in jieba.cut(line):
                    split_words.append(x)
                    words_len += 1

            words_tf = get_tf(split_words)
            bigram_tf= get_bigram_tf(split_words)
            bigram_tf2= get_bi_tf(split_words)
            trigram_tf = get_trigram_tf(split_words)

            print("语料库字数:", count)
            print("分词个数:", words_len)
            print("平均词长:", round(count/words_len, 3))

            bigram_len = sum([dic[1] for dic in bigram_tf.items()])
            trigram_len = sum([dic[1] for dic in trigram_tf.items()])

            print(file+":")
            
            entropy = [-(uni_word[1]/words_len)*math.log(uni_word[1]/words_len, 2) for uni_word in words_tf.items()]
            
            print("基于词的一元模型的中文信息熵为:", round(sum(entropy), 3), "比特/词")
            entropy = []
            for bi_word in bigram_tf.items():
                jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
                cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # 计算条件概率p(x|y)
                entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
            
            print("基于词的二元模型的中文信息熵为:", round(sum(entropy), 3), "比特/词")  
            entropy = []
            for tri_word in trigram_tf.items():
                jp_xy = tri_word[1] / trigram_len  # 计算联合概率p(x,y)
                cp_xy = tri_word[1] / bigram_tf2[tri_word[0][0]]  # 计算条件概率p(x|y)
                entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
            print("基于词的三元模型的中文信息熵为:", round(sum(entropy), 3), "比特/词")  

        #计算字信息熵
        with open(path + '\\' + file, 'r', encoding='gb18030') as f:
            split_words = []
            count = 0
            corpus = []
            word = []
            token_2gram =[]
            words_len = 0


            txt = f.read()
            for w in txt:
                if w != '\n':
                    word.append(w)
                    words_len+=1

            words_tf = get_tf(word)
            bigram_tf= get_bigram_tf(word)
            bigram_tf2 = get_bi_tf(word)
            trigram_tf = get_trigram_tf(word)

            entropy = [-(uni_word[1]/words_len)*math.log(uni_word[1]/words_len, 2) for uni_word in words_tf.items()]
            print("基于字的一元模型的中文信息熵为:", round(sum(entropy), 3), "比特/字")
            bigram_len = sum([dic[1] for dic in bigram_tf.items()])
            trigram_len = sum([dic[1] for dic in trigram_tf.items()])

            entropy = []
            for bi_word in bigram_tf.items():
                jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
                cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # 计算条件概率p(x|y)
                entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
            print("基于字的二元模型的中文信息熵为:", round(sum(entropy), 3), "比特/字")  

            entropy = []
            for tri_word in trigram_tf.items():
                jp_xy = tri_word[1] / trigram_len  # 计算联合概率p(x,y)
                cp_xy = tri_word[1] / bigram_tf2[tri_word[0][0]]  # 计算条件概率p(x|y)
                entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
            print("基于字的三元模型的中文信息熵为:", round(sum(entropy), 3), "比特/字")  
    after = time.time()
    print("运行时间:", round(after-before, 3), "s")
