def load_data_set():
    """
    定义数据集
    :return:
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
    通过集合的方式去重复
    :param data_set:
    :return:
    """
    vocab_set = set([])  # create empty set
    for document in data_set:
        vocab_set = vocab_set | set(document)  # union of the two sets
    return list(vocab_set)


def set_of_words_to_vec(vocab_list, input_set):
    """
    标记输入集合在词库中是否出现
    :param vocab_list:
    :param input_set:
    :return:
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return return_vec


if __name__ == "__main__":
    list_of_posts, list_of_classes = load_data_set()
    my_list = create_vocab_list(list_of_posts)
    print(set_of_words_to_vec(my_list, list_of_posts[0]))
    pass
