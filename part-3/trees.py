import math
import operator


def calc_shannon_ent(data_set):
    """
    计算香农熵 一个事件的信息量就是这个事件发生的概率的负对数
    :param data_set:
    :return:
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        # 香农熵公式
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def create_data_set():
    """
    模拟数据集
    :return:
    """
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    """
    分割数据集
    :param data_set:
    :param axis:
    :param value:
    :return:
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            # 相当于移除了 axis 的数据， 例如: 当feat_vec = [1, 2, 3] axis = 1 value = 2 时， 取出的结果为 [1, 3]
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    选择最好的数据集划分方式
    :param data_set:
    :return:
    """
    num_features = len(data_set[0]) - 1
    base_ent = calc_shannon_ent(data_set)
    bast_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):  # 循环 len(data_set[0]) - 1 次找出最佳的划分索引值
        feat_list = [x[i] for x in data_set]  # 取出第i列的所有值
        unique_values = set(feat_list)  # 去重
        new_entropy = 0.0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)  # 返回的是 data_set 中每个元素移除第i列后的列表 （如果第i列不等于value 则不返回）
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_ent - new_entropy
        if info_gain > bast_info_gain:  # 选出熵增最小的划分列 （也就是最能准确划分数据集的列）
            bast_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建树
    :param data_set:
    :param labels:
    :return:
    """
    class_list = [x[-1] for x in data_set]
    if class_list.count(class_list[0]) == len(class_list):  # 类别完全相同则停止继续划分
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    feat_values = [x[best_feat] for x in data_set]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


if __name__ == '__main__':
    data_set, labels = create_data_set()
    my_tree = create_tree(data_set, labels)
    print(my_tree)
