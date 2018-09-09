from random import seed, randrange, random

#导入CSV文件
def loadDataSet(filename):
    dataSet = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if not line:
                continue
                
            lineArr = []
            for feature in line.split(','):
                str_feature = feature.strip()#strip() 方法移除字符串指定的字符返回移除后生成新的字符

                if str_feature.isalpha():#如果是字母，则是标签
                    lineArr.append(str_feature)
                else:#将其转换为float
                    lineArr.append(float(str_feature))
            dataSet.append(lineArr)
    return dataSet

#data_set split n folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset) / n_folds
    for i in range(n_folds):
        fold = list() #每次循环fold清零，防止重复导入dataset_split
        while len(fold) < fold_size:
            #有放回的随机采样，有一些样本被重复采样，从而在训练集中重复出现，有的却从未出现，用自助采样法，保证每棵决策树训练集的差异性
            index = randrange(len(dataset_copy))#将对应索引index的内容从dataset_copy导出，并将该内容从dataset_copy删除
            fold.append(dataset_copy[index])#有放回的方式
        dataset_split.append(fold)

    return dataset_split

#根据特征和特征值value分割数据集
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, class_values):
    gini = 0.0
    D = len(groups[0]) + len(groups[1])
    for class_value in class_values: #class_values = [0, 1]
        for group in groups: #groups = [left, right]
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += float(size)/ D * (proportion*(1.0 - proportion))
    return gini

def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset)) #class_values = [0, 1]
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)

    for index in features:#
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group):
    #Create a terminal node value
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])

    #check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        # node['left']是一个字典，形式为{'index': b_index, 'value': b_value, 'groups':b_groups}所以node是一个多层字典
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)

def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']

    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset)*ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()

    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)

    predictions = [bagging_predict(trees, row) for row in test]
    return predictions

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

if __name__ == '__main__':
    dataset = loadDataSet('D:\project\machine_learning_action\data\\7.RandomForest\sonar-all-data.txt')
    n_folds = 5
    max_depth = 20
    min_size = 1
    sample_size = 1.0
    n_features = 15
    for n_trees in [1, 10, 20, 30, 40, 50]:
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        seed(1)
        print('random = ', random())
        print('Trees: %d' %n_trees)
        print('Scores:%s' %scores)
        print("mean accuracy: %.3f%%"%(sum(scores)/float(len(scores))))
    #cross_validation_split(dataset, 5)
