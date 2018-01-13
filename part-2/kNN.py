import numpy as np
import operator
import os
import matplotlib.pyplot as plt


def create_data_set():
    """
    生成数据集
    :return:
    """
    _group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    _labels = ['A', 'A', 'B', 'B']
    return _group, _labels


def classify0(in_x, data_set, labels, k):
    """
    基于欧氏距离公式计算距离
    :param in_x: 用于分类输入的向量
    :param data_set: 训练样本集
    :param labels: 标签向量
    :param k: 用于选择最近邻居的数目
    :return:
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


def file_to_matrix(file_path):
    """
    文件转换数据集
    :param file_path: 文件路径
    :return:
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        number_of_lines = len(lines)
        return_mat = np.zeros((number_of_lines, 3))
        class_label_vector = []
        index = 0
        for line in lines:
            line = line.strip()
            list_from_line = line.split('\t')
            return_mat[index, :] = list_from_line[0:3]
            class_label_vector.append(list_from_line[-1])
            index += 1
        return return_mat, class_label_vector


def auto_norm(data_set):
    """
    归一化特征值 （将特征值转换成0到1的区间）
    :param data_set:
    :return:
    """
    min_values = data_set.min(0)  # 获取每一列的最小值 0：列 1：行
    max_values = data_set.max(0)
    _ranges = max_values - min_values
    m = data_set.shape[0]  # 数据集的行数
    # 套公式： newValue = (oldValue - min) / (max - min)
    norm_data_set = data_set - np.tile(min_values, (m, 1))
    norm_data_set = norm_data_set / np.tile(_ranges, (m, 1))
    return norm_data_set, _ranges, min_values


def dating_class_test():
    """
    计算错误率
    :return:
    """
    mat, _labels = file_to_matrix('./datingTestSet.txt')
    norm_mat, ranges, min_values = auto_norm(mat)
    error_count = 0
    for i in range(norm_mat.shape[0]):
        classifier_res = classify0(norm_mat[i, :], norm_mat, _labels, 3)
        # print("the classifier came back with: %s, the real answer is： %s" % (classifier_res, _labels[i]))
        if classifier_res != _labels[i]:
            error_count += 1
    print("the total error rate is： %f" % (error_count / float(mat.shape[0])))


def img2vector(file_path):
    """
    转换图片为向量
    :param file_path:
    :return:
    """
    vector = np.zeros((1, 32 * 32))
    with open(file_path, 'r') as f:
        for i in range(32):
            line_str = f.readline()
            for j in range(32):
                vector[0, 32 * i + j] = int(line_str[j])
    return vector


def hand_writing_class_test():
    """
    手写数字识别
    :return:
    """
    _labels = []
    files = os.listdir('./trainingDigits')
    files_count = len(files)
    arr = np.zeros((files_count, 32 * 32))
    index = 0
    for file in files:
        v = img2vector('./trainingDigits/%s' % file)
        # 数组切片：arr_name[行操作, 列操作]
        arr[index, :] = np.array(v)
        _labels.append(file.split('_')[0])
        index += 1

    files = os.listdir('./testDigits')
    error_count = 0
    for file in files:
        v = img2vector('./testDigits/%s' % file)
        classifier_res = classify0(v, arr, _labels, 3)
        if classifier_res != file.split('_')[0]:
            error_count += 1
    print("the total error rate is： %f" % (error_count / float(len(files))))


if __name__ == '__main__':
    group, labels = create_data_set()
    # return 'B'
    res = classify0([0, 0], group, labels, 3)
    print(res)
    mat, labels = file_to_matrix('./datingTestSet2.txt')
    fig = plt.figure()  # 创建一个图形
    ax = fig.add_subplot(111)  # 添加子图 111：分割为1行一列选择第一块
    ax.scatter(mat[:, 0], mat[:, 1], 15.0 * np.array(labels).astype(np.int32), 15.0 * np.array(labels).astype(np.int32))
    # plt.show()
    dating_class_test()
    hand_writing_class_test()
