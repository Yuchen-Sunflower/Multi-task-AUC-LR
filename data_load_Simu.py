import scipy.io as scio
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

data_dir = r'Simulated\Simulated_10.mat'

test_ratio = 0.2

def Simulated_split(dataFile):
    task_num = 100
    # instance_num = 1000
    # 100 tasks, each task has 1000 80-dim instances
    data = scio.loadmat(dataFile)['data']
    label = scio.loadmat(dataFile)['label']
    # print(len(data), len(label))

    training_data = [i for i in range(task_num)]
    testing_data = [i for i in range(task_num)]
    # validation_data = [i for i in range(task_num)]

    training_label = [i for i in range(task_num)]
    testing_label = [i for i in range(task_num)]
    # validation_label = [i for i in range(task_num)]

    for i in range(task_num):
        training_data[i], testing_data[i], training_label[i], testing_label[i] = train_test_split(data[i], label[i], test_size=test_ratio, random_state=2022)
        # training_data[i], validation_data[i], training_label[i], validation_label[i] = train_test_split(training_data[i], training_label[i], test_size=0.1, random_state=2022)
        training_label[i] = training_label[i].reshape((training_label[i].shape[0]), 1)
        testing_label[i] = testing_label[i].reshape((testing_label[i].shape[0]), 1)

    return training_data, testing_data, training_label, testing_label


def Simulated_split_valid(dataFile):  # k fold cross-validation
    task_num = 100
    cross_validation_num = 5
    # 100 tasks, each task has 1000 80-dim instances

    data_temp = scio.loadmat(dataFile)['data']
    label_temp = scio.loadmat(dataFile)['label']

    data = [i for i in range(task_num)]
    label = [i for i in range(task_num)]
    
    for i in range(task_num):
        data[i], _, label[i], _ = train_test_split(data_temp[i], label_temp[i], test_size=test_ratio, random_state=2022)
    
    training_data = []
    validation_data = []
    training_label = []
    validation_label = []

    ######################### The K-fold cross-validation used for random split ##########################
    for k in range(cross_validation_num):
        training_data.append([i for i in range(task_num)])
        validation_data.append([i for i in range(task_num)])

        training_label.append([i for i in range(task_num)])
        validation_label.append([i for i in range(task_num)])

    skf = KFold(n_splits=cross_validation_num, random_state=2022, shuffle=True)
    for i in range(task_num):
        k = 0
        for train_index, test_index in skf.split(data[i]):
            # print('TRAIN:', train_index, "TEST:", test_index)
            training_data[k][i] = np.array([data[i][j] for j in train_index])
            validation_data[k][i] = np.array([data[i][j] for j in test_index])
            training_label[k][i] = np.array([label[i][j] for j in train_index])
            validation_label[k][i] = np.array([label[i][j] for j in test_index])

            k += 1
    ######################################################################################################
    return training_data, validation_data, training_label, validation_label


if __name__ == '__main__':
    training_data, testing_data, training_label, testing_label = Simulated_split(data_dir)