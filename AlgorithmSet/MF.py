import numpy as np
import pandas as pd
import os
from tensorflow import keras # tensorflow == 2.X
import warnings
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# 进行推荐
def recommend(user_id, uel, mel, N):
    movies = uel[user_id-1] @ mel.T # -1是因为预处理后的用户id从0开始
    mids = np.argpartition(movies, -N)[-N:]
    return mids

if __name__ == "__main__":
    # ------ 读入数据 ------ #
    dataset = pd.read_csv("./ratings.csv", sep=",", names=["user_id", "item_id", "rating", "timestamp"])
    # 数据预处理，下标从0开始，去除缺失值使得值连续
    dataset.user_id = dataset.user_id.astype('category').cat.codes.values
    dataset.item_id = dataset.item_id.astype('category').cat.codes.values
    # 获取用户和项目列表
    user_arr = dataset.user_id.unique()
    movies_arr = dataset.item_id.unique()
    # 获取用户和项目数量
    n_users, n_movies = len(user_arr), len(movies_arr)  # 6040 3706
    n_latent_factors = 20

    # ------ 设置Keras参数 ------ #
    # 设置项目参数
    movie_input = keras.layers.Input(shape=[1], name='Item')
    movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
    # 设置用户参数
    user_input = keras.layers.Input(shape=[1], name='User')
    user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors, name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
    # 计算项目向量与用户张量的点乘
    prod = keras.layers.dot([movie_vec, user_vec], axes=1, name='DotProduct')
    # 创建用户-项目模型
    model = keras.Model([user_input, movie_input], prod)
    # 设置模型优化器、损失函数、测量指标
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

    # ------ 训练模型 ------ #
    # 训练用户-项目模型
    # verbose=0：不输出日志；verbose=1：输出每一个step的训练进度及日志；verbose=2：输出每个epochs的日志
    model.fit([dataset.user_id, dataset.item_id], dataset.rating, epochs=5, verbose=1)
    # 获得用户和项目的嵌入矩阵
    user_embedding_learnt = model.get_layer(name='User-Embedding').get_weights()[0]
    movie_embedding_learnt = model.get_layer(name='Movie-Embedding').get_weights()[0]

    # ------ 进行推荐 ------ #
    # 给用户1推荐top10
    user = 1
    topN = recommend(user_id=user, uel=user_embedding_learnt, mel=movie_embedding_learnt, N=10)
    temp_topN = topN.tolist()

    print("------ user ------")
    print(user)
    print("------ temp_topN ------")
    print(temp_topN)

    # 给所有用户推荐Top10
    # for each_user in tqdm(user_arr, total=len(user_arr)):
    #     topN = recommend(user_id=each_user, uel=user_embedding_learnt, mel=movie_embedding_learnt, N=10)
    #     temp_topN = topN.tolist()
    #     print("------ each_user ------")
    #     print(each_user)
    #     print("------ temp_topN ------")
    #     print(temp_topN)
