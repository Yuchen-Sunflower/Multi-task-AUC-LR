import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold, train_test_split

files = [r'USPS\task1.mat', r'USPS\task2.mat', r'USPS\task3.mat', r'USPS\task4.mat', r'USPS\task5.mat']
cross_validation_num = 5
test_ratio = 0.2

def dight_split_valid(dataFile):  # k fold cross-validation
    data_temp = []
    label_temp = []
    task_num = 5

    for file in dataFile:
        data_temp.append(np.concatenate((scio.loadmat(file)['train_data'], scio.loadmat(file)['test_data']), axis=0))
        label_temp.append(np.concatenate((scio.loadmat(file)['train_label'], scio.loadmat(file)['test_label']), axis=0))
        for i in range(len(data_temp)):
            np.random.seed(7)
            np.random.shuffle(data_temp[i])
            np.random.seed(7)
            np.random.shuffle(label_temp[i])

    data = [i for i in range(task_num)]
    label = [i for i in range(task_num)]
    # 5 tasks, each task has N_m 180-dim instances
    for i in range(task_num):
        data[i], _, label[i], _ = train_test_split(data_temp[i], label_temp[i], test_size=test_ratio, random_state=2022)
    
    training_data = []
    validation_data = []

    training_label = []
    validation_label = []

    # -------------------------- The K-fold cross-validation used for random split --------------------------
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
    # -------------------------------------------------------------------------------------------------------
    return training_data, validation_data, training_label, validation_label


def dight_split(dataFile):
    # 5 tasks, each task has N_m 180-dim instances
    data = []
    label = []
    task_num = 5

    for file in dataFile:
        data.append(np.concatenate((scio.loadmat(file)['train_data'], scio.loadmat(file)['test_data']), axis=0))
        label.append(np.concatenate((scio.loadmat(file)['train_label'], scio.loadmat(file)['test_label']), axis=0))
        for i in range(len(data)):
            np.random.seed(7)
            np.random.shuffle(data[i])
            np.random.seed(7)
            np.random.shuffle(label[i])

    training_data = [i for i in range(task_num)]
    # validation_data = [i for i in range(task_num)]
    testing_data = [i for i in range(task_num)]

    training_label = [i for i in range(task_num)]
    # validation_label = [i for i in range(task_num)]
    testing_label = [i for i in range(task_num)]

    for i in range(task_num):
        training_data[i], testing_data[i], training_label[i], testing_label[i] = train_test_split(data[i], label[i], test_size=test_ratio, random_state=2022)
        # training_data[i], validation_data[i], training_label[i], validation_label[i] = train_test_split(training_data[i], training_label[i], test_size=0.1, random_state=2022)

    # scio.savemat('USPS_split.mat', {'USPS_train_data': np.array(training_data), 'USPS_train_label': np.array(training_label), "USPS_valid_data": np.array(validation_data), "USPS_valid_label": np.array(validation_label), \
    #  "USPS_test_data": np.array(testing_data), "USPS_test_label": np.array(testing_label)})
    # scio.savemat('USPS.mat', {'data': data, 'label': label})
    return training_data, testing_data, training_label, testing_label


if __name__ == "__main__":
    a, b, c, d, e, f = dight_split(files)
