import math

class UserBasedCF:
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = []
        self.trainData = {}
        self.userSimMatrix = []

    def readData(self):
        """
        在Movielens数据集中读取数据
        """
        datalist = []
        for line in open(self.datafile):
            userid, itemid, record, _ = line.split(",") # 用逗号分割
            datalist.append((int(userid), int(itemid), int(record)))
        self.data = datalist

    def preprocessData(self):
        """
        把读入的数据转换为训练UCF模型需要的格式
        """
        traindata_list = {}
        # 存储格式：
        for user, item, record in self.data:
            traindata_list.setdefault(user, {})
            traindata_list[user][item] = record
        self.trainData = traindata_list

    def userSimilarity(self):
        """
        生成用户相似度矩阵
        """
        self.userSimMatrix = dict()
        # 物品用户倒排表
        item_users = dict()
        for u, item in self.trainData.items():
            for i in item.keys():
                item_users.setdefault(i, set())
                item_users[i].add(u)
        # 计算用户间同时评分的物品
        user_item_count = dict()
        count = dict()
        for item, users in item_users.items():
            for u in users:
                user_item_count.setdefault(u, 0)
                user_item_count[u] += 1
                for v in users:
                    if u == v : continue
                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1
        # 计算相似度矩阵
        for u, related_users in count.items():
            self.userSimMatrix.setdefault(u, dict())
            for v, cuv in related_users.items():
                self.userSimMatrix[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v] * 1.0)

    def recommend(self, user_id, k, N):
        '''
        给用户推荐K个与之相似用户喜欢的物品
        :param user: 用户id
        :param k: 近邻范围
        :param N: 推荐列表长度
        :return: 推荐列表
        '''
        rank = dict() # k个近邻用户的
        interacted_items = self.trainData.get(user_id, {}) # 当前用户已经交互过的item
        # 取最相似的k个用户的item
        # nbor_u是近邻用户的id，nbor_u_sim是近邻用户与当前用户的相似度
        for nbor_u, nbor_u_sim in sorted(self.userSimMatrix[user_id].items(), key=lambda x:x[1], reverse=True)[0:k]:
            for i, i_score in self.trainData[nbor_u].items(): # 取出所有近邻用户的item
                if i in interacted_items: # 不计入用户已经交互过的item
                    continue
                rank.setdefault(i, 0) # 初始化rank
                rank[i] += nbor_u_sim # 相似度求和，作为item的得分
        # 取出得分最高的N个item作为推荐列表
        return dict(sorted(rank.items(), key=lambda x:x[1], reverse=True)[0:N])

if __name__ == "__main__":
    ubcf = UserBasedCF('ratings.csv')
    ubcf.readData() # 读取数据
    ubcf.preprocessData() # 预处理数据
    ubcf.userSimilarity() # 计算用户相似度矩阵

    # ------ 为用户 i 产生推荐 ------ #
    i = 1
    topN = ubcf.recommend(i, k=3, N=10)  # 输出格式：item的id和评分
    topN_list = list(topN.keys())  # 只取对应的item的id

    print("------ i ------")
    print(i)
    print("------ topN_list ------")
    print(topN_list)

    # ------ 为全部用户产生推荐 ------ #
    # topN_list = {} # 存储为每一个用户推荐的列表
    # for each_user in ubcf.trainData:
    #     topN = ubcf.recommend(each_user, k=3, N=10) # item的id和评分
    #     topN_list[each_user] = list(topN.keys()) # 只取对应的item的id
    #
    #     print("------ topN_list[each_user] ------")
    #     print(topN_list[each_user])
    # print(topN_list)
