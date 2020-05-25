import gc
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Lambda, Dropout, Dense
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split

import keras.backend as K


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def seq_padding(X, padding=0):
    """
    让每条文本的长度相同，用0填充
    :param X:
    :param padding:
    :return:
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def train_and_test_split(data):
    # 划分数据集
    X = data['cut_comment']
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print('按照7：3的比例切分训练集和测试集')
    with open('data/tran_test_data.pkl', 'wb')as file:
        pickle.dump((X_train, X_test, y_train, y_test), file)
    return X_train, X_test, y_train, y_test


# data_generator只是一种为了节约内存的数据方式
class data_generator:
    def __init__(self, data, batch_size=8, shuffle=True):
        self.maxLen = 200
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:self.maxLen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []


def acc_top2(y_true, y_pred):
    """
    计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
    :param y_true:
    :param y_pred:
    :return:
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def build_bert_model(nClass):
    # 默认不对Bert模型进行调参
    # trainable设置True对Bert进行微调
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, trainable=True)
    inputs_1 = Input(name='inputs_1', shape=[None, ])
    inputs_2 = Input(name='inputs_2', shape=[None, ])

    layer = bert_model([inputs_1, inputs_2])
    layer = Lambda(lambda x: x[:, 0])(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(nClass, activation='softmax')(layer)
    model = Model(inputs=[inputs_1, inputs_2], outputs=layer)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy', acc_top2])
    model.summary()
    return model


# 交叉验证训练和测试模型
def run_cv(nfold, data, data_labels, data_test):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(data)
    train_model_pred = np.zeros((len(data), 2))
    test_model_pred = np.zeros((len(data_test), 2))

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid = data[train_fold, :], data[test_fold, :]

        model = build_bert_model(2)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5,
                                    patience=2)  # 当评价指标不在提升时，减少学习率
        checkpoint = ModelCheckpoint(str(i) + '.hdf5', monitor='val_acc', verbose=2, save_best_only=True, mode='max',
                                     save_weights_only=True)  # 保存最好的模型

        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=True)
        test_D = data_generator(data_test, shuffle=False)

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=5,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )

        train_model_pred[test_fold, :] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)

        del model
        gc.collect()  # 清理内存
        K.clear_session()  # clear_session就是清除一个session

    return train_model_pred, test_model_pred


# 配置keras自适应使用显存
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# 设置随机种子
np.random.seed(123)

path_prefix = "/home/master/数据/bert"
# 预训练模型目录
config_path = path_prefix + "/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = path_prefix + "/chinese_L-12_H-768_A-12/bert_model.ckpt"
dict_path = path_prefix + "/chinese_L-12_H-768_A-12/vocab.txt"

token_dict = {}
tokenizer = OurTokenizer(token_dict)

data = pd.read_csv('data/data.csv')
data['cut_comment'] = data['cut_comment'].astype(str)
X_train, X_test, y_train, y_test = train_and_test_split(data)
train_one_hot_labels = to_categorical(y_train, num_classes=2)
test_one_hot_labels = to_categorical(y_test, num_classes=2)
# 训练数据、测试数据和标签转化为模型输入格式
DATA_LIST = []
for i in range(len(X_train)):
    DATA_LIST.append((X_train[i], train_one_hot_labels[i]))
DATA_LIST = np.array(DATA_LIST)

DATA_LIST_TEST = []
for i in range(len(X_test)):
    DATA_LIST_TEST.append((X_test[i], test_one_hot_labels[i]))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)

train_model_pred, test_model_pred = run_cv(5, DATA_LIST, None, DATA_LIST_TEST)
test_pred = [np.argmax(x) for x in test_model_pred]

# keras-bert
print('================Bert算法=================')
print("Bert模型准确率:", accuracy_score(y_test, test_pred))
print('Bert模型的召回率:', recall_score(y_test, test_pred))
print('Bert模型的f1-score:', f1_score(y_test, test_pred, average='weighted'))
