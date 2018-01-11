import numpy as np
import operator


def create_data_set():
    """
    生成数据集
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    """
    基于欧氏距离公式计算距离
    :param in_x: 用于分类输入的向量
    :param data_set: 训练样本集
    :param labels: 标签向量
    :param k: 用于选择最近邻居的数目
    :return: 最近的邻居标签
    """
    # 返回每个维度中数组大小的元祖
    data_set_size = data_set.shape[0]
    # 将传入的数组重复 data_set_size 次，然后减去训练集，得到差值数组
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    # 将差值数组平方
    sq_diff_mat = diff_mat ** 2
    # 将差值的平方求和，用于欧氏距离公式计算
    sq_distances = sq_diff_mat.sum(axis=1)
    # 开方
    distances = sq_distances ** 0.5
    # 返回数组元素从小到大的索引
    sorted_dist = distances.argsort()

    class_count = {}
    for i in range(k):
        # 取出类型
        vote_label = labels[sorted_dist[i]]
        # 统计匹配类型的次数
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 字典排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    group, labels = create_data_set()
    # return 'B'
    res = classify0([0, 0], group, labels, 3)
    print(res)
