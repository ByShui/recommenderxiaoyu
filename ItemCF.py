import math

class ItemBasedCF:
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = []
        self.trainData = {}
        self.itemSimMatrix = []

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
        for user, item, record in self.data:
            traindata_list.setdefault(user, {})
            traindata_list[user][item] = record
        self.trainData = traindata_list

    def itemSimilarity(self):
        """
        生成用户相似度矩阵
        """
        # 物品相似度矩阵
        self.itemSimMatrix = dict()
        # 物品-物品矩阵 格式：物品ID1:{物品ID2:同时给两件物品评分的人数}
        item_item_matrix = dict()
        # 物品-用户矩阵 格式：物品ID:给物品评分的人数
        item_user_matrix = dict()

        for user, items in self.trainData.items():
            for itemId, source in items.items():
                item_user_matrix.setdefault(itemId, 0) # 初始化item_user_matrix[itemId]
                item_user_matrix[itemId] += 1 # 给物品打分的人数+1
                item_item_matrix.setdefault(itemId, {})
                for i in items.keys():
                    if i == itemId:
                        continue
                    item_item_matrix[itemId].setdefault(i, 0)
                    # 计算同时给两个物品打分的人数，并对活跃用户进行惩罚
                    item_item_matrix[itemId][i] += 1 / math.log(1 + len(items) * 1.0)
        # 将物品-物品矩阵转换为物品相似度矩阵
        for itemId, relatedItems in item_item_matrix.items():
            self.itemSimMatrix.setdefault(itemId, dict()) # 初始化self.itemSimMatrix[itemId]
            for relatedItemId, count in relatedItems.items():
                self.itemSimMatrix[itemId][relatedItemId] = count / math.sqrt(item_user_matrix[itemId] * item_user_matrix[relatedItemId])
            # 归一化
            sim_max = max(self.itemSimMatrix[itemId].values())
            for item in self.itemSimMatrix[itemId].keys():
                self.itemSimMatrix[itemId][item] /= sim_max

    def recommend(self, user_id, k, N):
        """给用户推荐物品列表
            Args:
                userId:用户ID
                k:取和物品j最相似的K个物品
                N:推荐N个物品
            Return:
                推荐列表
            """
        rank = dict()
        interacted_items = self.trainData.get(user_id, {}) # 当前用户已经交互过的item
        # 遍历用户评分的物品列表
        for itemId, score in interacted_items.items(): # 取出每一个当前用户交互过的item
            # 取出与物品itemId最相似的K个物品及其评分
            for i, sim_ij in sorted(self.itemSimMatrix[itemId].items(), key=lambda x: x[1], reverse=True)[0:k]:
                # 如果这个物品j已经被用户评分了，舍弃
                if i in interacted_items.keys():
                    continue
                # 对物品ItemID的评分*物品itemId与j的相似度 之和
                rank.setdefault(i, 0)
                rank[i] += score * sim_ij
        # 堆排序，推荐权重前N的物品
        return dict(sorted(rank.items(), key=lambda x:x[1], reverse=True)[0:N])


if __name__ == "__main__":
    ibcf = ItemBasedCF('ratings.csv')
    ibcf.readData() # 读取数据
    ibcf.preprocessData() # 预处理数据
    ibcf.itemSimilarity() # 计算物品相似度矩阵

    # ------ 为用户 i 产生推荐 ------ #
    i = 1
    topN = ibcf.recommend(i, k=3, N=10)  # 输出格式item的id和评分
    topN_list = list(topN.keys())  # 只取对应的item的id

    print("------ i ------")
    print(i)
    print("------ topN_list ------")
    print(topN_list)

    # ------ 为全部用户产生推荐 ------ #
    # topN_list = {} # 存储为每一个用户推荐的列表
    # for each_user in ibcf.trainData:
    #     topN = ibcf.recommend(each_user, k=3, N=10) # item的id和评分
    #     topN_list[each_user] = list(topN.keys()) # 只取对应的item的id
    #
    #     print("------ topN ------")
    #     print(topN)
    #     print("------ topN_list[each_user] ------")
    #     print(topN_list[each_user])
    # print("------ topN_list ------")
    # print(topN_list)
