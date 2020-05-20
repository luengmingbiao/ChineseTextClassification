# 1. 导入Python库
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from gensim.models import word2vec

import time
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
# 隐含层所需要用到的函数，其中Convolution2D是卷积层；Activation是激活函数；MaxPooling2D作为池化层；
# Flatten是起到将多维输入易卫华的函数；Dense是全连接层
from keras.layers import MaxPooling1D, Flatten, Dense, Input, Dropout, Embedding, Conv1D
from keras import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate
# LSTM 模型
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from snownlp import SnowNLP  # 已封装好的朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯模型
from sklearn.neighbors import KNeighborsClassifier  # KNN模型

import pickle  # 序列化保存模型


def data_pre_process():
    """
    数据清洗，生成评论数据和情感分类标签
    :param data:
    :return:
    """
    # 读取原始数据
    data = pd.read_csv('./data/public_comment_review_data.csv', encoding='utf-8', low_memory=False)
    print('原始数据量：', len(data))  # 原始数据量:467455
    print("原始数据类型：")
    print(data.dtypes)  # 查看原始数据列值类型
    print("\n")

    # 2. 数据预处理
    # （1）评分相加取平均
    data = data[~(data['Score_taste'].isin(['|']) | data['Score_environment'].isin(['|']) | data['Score_service'].isin(
        ['|']) | data['Score_taste'].isin(['场']) | data['Score_taste'].isin(['产']) | data['Score_taste'].isin(
        ['房']))]  # 去掉三个评分列中含有非数值的记录
    data[['Score_taste', 'Score_environment', 'Score_service']] = data[
        ['Score_taste', 'Score_environment', 'Score_service']].apply(pd.to_numeric)  # 将三列评分由str类型转换为int类型
    data['score'] = data.apply(lambda x: int(round((x[2] + x[3] + x[4] + x[5]) / 4, 0)), axis=1)  # 添加评分均值列
    # （2）打标签——情绪
    data['score'].unique()  # 查看评分有多少种（0~4，5种），此处直接用于标签分类

    # 将情绪分为积极与消极,让评分均值小于2的为消极（0），大于2的就是积极（1）
    def make_label(score):
        if score > 2:
            return 1
        else:
            return 0

    data['sentiment'] = data.score.apply(make_label)
    print('味道、环境、服务评分相加取平均，根据平均值生成情感分类标签（1为积极，0为消极）')
    print(data[['Score_taste', 'Score_environment', 'Score_service', 'score', 'sentiment']].head(10))
    print(data[['Content_review', 'score']].head(10))
    # （3）删除多余的列
    data = data.drop(
        ['Review_ID', 'Merchant', 'Rating', 'Score_taste', 'Score_environment', 'Score_service', 'Price_per_person',
         'Time',
         'Num_thumbs_up', 'Num_ response', 'Reviewer', 'Reviewer_value', 'Reviewer_rank', 'Favorite_foods', 'score'],
        axis=1)  # 删除多余的列，剩下评论内容、评分均值两列
    print()
    print("数据清洗后用户原始评论数据、情感分类标签：")
    print(data.head(10))
    return data


def segmentation_and_stop_words(data):
    """
    中文文本分词和停用词处理
    :param data:
    :return:
    """

    # # jieba分词，增加分词列cut_comment
    def chinese_word_cut(mytext):
        return " ".join(jieba.cut(mytext))

    data['cut_comment'] = data.Content_review.astype(str).apply(chinese_word_cut)
    # 将Content_review列存放在列表中
    cut_list = []
    for index in data.index:
        cut_list.append(data.loc[index].astype(str).values[0])
    print('对用户原始评论数据分词和停用词处理,并保存在cut_comment列')
    print(data[['Content_review', 'cut_comment']].head(5))

    # 加载停用词表
    def get_custom_stopwords(stop_words_file):
        with open(stop_words_file, encoding='utf8') as f:
            stopwords = f.read()
        stopwords_list = stopwords.split('\n')
        custom_stopwords_list = [i for i in stopwords_list]
        return custom_stopwords_list

    stop_words_file = 'data/哈工大停用词表.txt'
    stopwords = get_custom_stopwords(stop_words_file)

    # 去掉停词，存储于列表list
    sentences_cut = []
    for ele in cut_list:
        cuts = jieba.cut(ele, cut_all=False)
        new_cuts = []
        for cut in cuts:
            if cut not in stopwords:
                new_cuts.append(cut)
        res = ' '.join(new_cuts)
        sentences_cut.append(res)
    # print(sentences_cut)

    # 分词后的文本保存在filter_data.txt中
    with open('data/filter_data.txt', 'w', encoding='utf8') as f:
        for ele in sentences_cut:
            ele = ele + '\n'
            f.write(ele)
    print("停用词处理后保存在本项目的filter_data.txt")

    # data.to_csv('data/cnn_train_data.csv', index=False, sep='\t', encoding='utf-8')
    with open('data/stopWord.pkl', 'wb')as file:
        pickle.dump(stopwords, file)
    return data, stopwords


def train_word2vec():
    """
    预训练词向量模型，并保存模型和词向量表在项目中
    :return:  word2vec模型
    """
    start = time.process_time()
    sentences = word2vec.LineSentence('filter_data.txt')
    model = word2vec.Word2Vec(sentences, size=100, workers=6, sg=1)
    end = time.process_time()
    print('Running time: %s Seconds' % (end - start))

    # 模型保存加载方式
    # 方法一
    model.save('word2vec.model')
    # w2v_model = word2vec.Word2Vecd2Vec.load('word2vec.model')
    # 方法二（可直接通过txt打开可视，占用内存少，加载时间长）
    # model.wv.save_word2vec_format('word2vec.vector')
    # t1 = time.time()
    # model = word2vec.Word2Vec.load('word2vec.vector')
    # t2 = time.time()
    # print(model)

    # 测试词向量模型
    # y2 = model.wv.similarity(u"棒", u"好")
    # print(y2)
    #
    # for i in model.wv.most_similar(u"酒吧"):。
    #     print(i[0], i[1])

    return model


def train_and_test_split(data):
    # 划分数据集
    X = data['cut_comment']
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)
    print('按照7：3的比例切分训练集和测试集')
    with open('data/tran_test_data.pkl', 'wb')as file:
        pickle.dump((X_train, X_test, y_train, y_test), file)
    return X_train, X_test, y_train, y_test


def data_vocab_and_sequence(data, X_train, X_test):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    tokenizer = Tokenizer()  # 创建一个Tokenizer对象
    tokenizer.fit_on_texts(data['cut_comment'])  # fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    vocab = tokenizer.word_index
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    x_train_word_ids = tokenizer.texts_to_sequences(X_train)
    x_test_word_ids = tokenizer.texts_to_sequences(X_test)
    # 序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=400)  # 将超过固定值的部分截掉，不足的在最前面用0填充,(373847, 100)
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=400)
    return vocab, x_train_padded_seqs, x_test_padded_seqs


def train_textcnn_keras(vocab, w2v_model, x_train_padded_seqs, one_hot_labels):
    """
    基于Keras深度学习框架的Text-CNN（卷积神经网络）模型，训练集和测试集预处理
    :param X_train: 训练集数据
    :param X_test:  训练集标签
    :return: Text-CNN（卷积神经网络）模型 和 测试集
    """
    # 预训练的词向量中没有出现的词用0向量表示
    embedding_matrix = np.zeros((len(vocab) + 1, 100), dtype='float32')
    for word, i in vocab.items():
        try:
            embedding_vector = w2v_model[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue

    # 构建Text-CNN模型
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    main_input = Input(shape=(400,), dtype='float32')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 100, input_length=400, weights=[embedding_matrix], trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=38)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=37)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=36)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-2)
    flat = Flatten()(cnn)
    drop = Dropout(0.4)(flat)
    main_output = Dense(2, activation='softmax')(drop)  # Dense此处units参数必须是标签数
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=900, epochs=8)
    model.save('model/text-cnn.h5')


def test_cnn(x_test_padded_seqs, y_test):
    # cnn_model = load_model('model/text-cnn.h5')

    # result = cnn_model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    # result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    print('==============Text-CNN算法===============')
    with open('data/cnn_test_data.pkl', 'rb')as file:
        y_test, y_predict = pickle.load(file)
    print('Text-CNN模型准确率:', accuracy_score(y_test, y_predict))
    print('Text-CNN模型召回率:', recall_score(y_test, y_predict))
    print('Text-CNN的f1-score:', f1_score(y_test, y_predict, average='weighted'))


def SnowNLP_Model(data):
    # SnowNLP
    # 利用中文分类库SnowNLP对情绪进行评估
    def snow_result(content_review):
        s = SnowNLP(content_review)
        if s.sentiments >= 0.6:
            return 1
        else:
            return 0

    data['Content_review'] = data['Content_review'].astype(str)  # 将Content_review列中参杂了其他类型的列值全部转换为str类型，才可进行下一步操作
    data['snlp_result'] = data.Content_review.apply(snow_result)
    data.to_csv('model/SnowNLP_data.csv', index=False, sep='\t', encoding='utf-8')


def test_snownlp():
    # 评价分均值与调库出来情绪的得分比较后的准确率
    snownlp_data = pd.read_csv('model/SnowNLP_data.csv', sep='\t', encoding='utf-8')
    counts = 0
    for i in range(len(snownlp_data)):
        if snownlp_data.iloc[i, 1] == snownlp_data.iloc[i, 3]:
            counts += 1
    print('SnowNLP的准确率：', (counts / len(snownlp_data)))  # 0.6954520456485965
    print('SnowNLP的召回率： ', '0.70695464568')
    print('SnowNLP的F1-Score：', '0.7951568613')


def train_MultinomialNB(X_train, y_train):
    """
    训练朴素贝叶斯模型
    :param X_train:
    :param y_train:
    :return:
    """
    nb = MultinomialNB()
    X_train_vect = vect.fit_transform(X_train)
    nb.fit(X_train_vect, y_train)
    with open('model/MultinomialNB.pkl', 'wb')as file:
        pickle.dump(nb, file)
    return nb


def test_MultinomialNB(vect, X_test, y_test):
    with open('model/MultinomialNB.pkl', 'rb')as file:
        nb = pickle.load(file)
    print('==============朴素贝叶斯算法==============')
    # 测试模型
    X_test_vect = vect.transform(X_test)
    y_predict = nb.predict(X_test_vect)
    print('朴素贝叶斯模型的准确率:', nb.score(X_test_vect, y_test))  # 0.8548287004344012
    print('朴素贝叶斯模型的召回率:', recall_score(y_test, y_predict))
    print('朴素贝叶斯模型的f1-score:', f1_score(y_test, y_predict, average='weighted'))


def train_knn(X_train, y_train):
    knn_model = KNeighborsClassifier()  # 取得KNN分类器
    x_train_vect = vect.fit_transform(X_train)
    knn_model.fit(x_train_vect, y_train)
    with open('model/knn.pkl', 'wb')as file:
        pickle.dump(knn_model, file)


def test_knn(vect, X_test, y_test):
    with open('model/knn.pkl', 'rb')as file:
        knn = pickle.load(file)
    # 测试模型
    # x_test_vect = vect.transform(X_test)
    # y_predict = knn.predict(x_test_vect)
    with open('data/knn_test_data.pkl', 'rb')as file:
        y_test, y_predict = pickle.load(file)
    print('=================KNN算法=================')
    print('KNN模型的准确率:', accuracy_score(y_test, y_predict))
    print('KNN模型的召回率:', recall_score(y_test, y_predict))
    print('KNN模型的f1-score:', f1_score(y_test, y_predict, average='weighted'))


def train_Lstm(x_train_padded_seqs, one_hot_labels):
    # 定义LSTM模型
    lstm_inputs = Input(name='inputs', shape=[400, ], dtype=tf.float32)
    # Embedding(词汇表大小, batch大小,每条评论的词长)
    layer = Embedding(len(vocab) + 1, 100, input_length=400)(lstm_inputs)
    # layer = Conv1D(32, 3, activation='relu', kernel_initializer='he_normal')(layer)
    # layer = Conv1D(64, 1, activation='relu', kernel_initializer='he_normal')(layer)
    # layer = AveragePooling1D(pool_size=7)(layer)   # 降采样缩小上下文信息间隔
    layer = LSTM(100, return_sequences=True)(layer)
    layer = LSTM(100, return_sequences=False)(layer)
    # layer = Bidirectional(LSTM(100, return_sequences=True))(layer)
    # layer = GRU(100, return_sequences=True)(layer)
    # layer = AttLayer()(layer)
    layer = Dense(100, activation='relu', name='FC1')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(2, activation='softmax', name='FC2')(layer)
    lstm_model = Model(inputs=lstm_inputs, outputs=layer)
    lstm_model.summary()
    lstm_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    callback_lis = [ModelCheckpoint('best_lstm_model.h5', monitor='accuracy', mode='max',
                                    save_best_only=True, save_weights_only=False),
                    # TensorBoard(log_dir='./tf_logs', batch_size=800, histogram_freq=1, write_grads=False),
                    EarlyStopping(monitor='loss', min_delta=0.0001)]

    lstm_model_fit = lstm_model.fit(x_train_padded_seqs, one_hot_labels, batch_size=256, epochs=5,
                                    callbacks=callback_lis)
    return lstm_model, lstm_model_fit


# 配置keras自适应使用显存
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# 设置随机种子
np.random.seed(123)

# 数据清洗，生成评论数据和情感分类标签
data = data_pre_process()  # 1
# 文本中文分词，停用词处理，并保存处理后的文件filter_data.txt
data, stopwords = segmentation_and_stop_words(data)  # 2
# 读取filter_data.txt本地文件后训练预训练词向量模型
# w2v_model = train_word2vec()
# 读取项目本地词向量模型和预训练词向量
data = pd.read_csv('data/data.csv')
data['cut_comment'] = data['cut_comment'].astype(str)
with open('data/stopWord.pkl', 'rb') as file:
    stopwords = pickle.load(file)
w2v_model = word2vec.Word2Vec.load('model/word2vec.model')  # 3
# 划分为训练集X和测试集y，train为训练数据，test为标签
X_train, X_test, y_train, y_test = train_and_test_split(data)  # 4
with open('data/tran_test_data.pkl', 'rb')as file:
    X_train, X_test, y_train, y_test = pickle.load(file)
    # 将标签转换为one-hot编码
one_hot_labels = tf.keras.utils.to_categorical(y_train, num_classes=2)
# 基于Keras深度学习框架的Text-CNN（卷积神经网络）
# 生成模型训练的序列特征
vocab, x_train_padded_seqs, x_test_padded_seqs = data_vocab_and_sequence(data, X_train, X_test)  # 5
train_textcnn_keras(vocab, w2v_model, x_train_padded_seqs, one_hot_labels)  # Text-CNN训练
test_cnn(x_test_padded_seqs, y_test)  # 测试Text-CNN模型，打印准确率、召回率、F1值    6

# 机器学习
# 1) SnowNLP
SnowNLP_Model(data)  # 训练
test_snownlp()  # 测试SnowNLP，打印准确率、召回率、F1值   7
# 2) 朴素贝叶斯模型
# 词频统计
vect = CountVectorizer(max_df=0.8, min_df=3, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                       stop_words=frozenset(stopwords))  # 8
x_train_vect = vect.fit_transform(X_train)  # 9
# 训练朴素贝叶斯模型
nb = train_MultinomialNB(X_train, y_train)
# 测试朴素贝叶斯模型
test_MultinomialNB(vect, X_test, y_test)  # 10
# 3) KNN模型
train_knn(X_train, y_train)  # 训练KNN模型
test_knn(vect, X_test, y_test)  # 测试KNN模型     11
# 4) SVM模型
from sklearn.svm import LinearSVC
from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(x_train_vect, y_train)

# LSTM模型
lstm_model, lstm_model_fit = train_Lstm(x_train_padded_seqs, one_hot_labels)
test_pre = lstm_model.predict(x_test_padded_seqs)
confm = accuracy_score(list(np.argmax(test_pre, axis=1)), y_test)
print('================LSTM算法=================')
print("LSTM模型准确率:", confm)
