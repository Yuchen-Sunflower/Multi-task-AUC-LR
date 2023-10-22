from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
# from model import inception_v3
import scipy.io as scio
import numpy as np

data_dir = 'LandmineData.mat'

test_ratio = 0.2
cross_validation_num = 5


def landmine_valid(dataFile):  # k fold cross-validation
    task_num = 29
    # 29 tasks, each task has N_m 9-dim instances
    data_temp = scio.loadmat(dataFile)
    data = [i for i in range(task_num)]
    label = [i for i in range(task_num)]
    for i in range(task_num):
        data[i], _, label[i], _ = train_test_split(data_temp['feature'][0][i], data_temp['label'][0][i], test_size=test_ratio, random_state=2022)
    
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
    #######################################################################################################
    return training_data, validation_data, training_label, validation_label


def landmine_test(dataFile):
    task_num = 29
    # 29 tasks, each task has N_m 9-dim instances
    data = scio.loadmat(dataFile)

    training_data = [i for i in range(task_num)]
    testing_data = [i for i in range(task_num)]
    # validation_data = [i for i in range(task_num)]

    # validation_label = [i for i in range(task_num)]
    training_label = [i for i in range(task_num)]
    testing_label = [i for i in range(task_num)]

    for i in range(task_num):
        training_data[i], testing_data[i], training_label[i], testing_label[i] = train_test_split(data['feature'][0][i], data['label'][0][i], test_size=test_ratio, random_state=2022)
        # training_data[i], validation_data[i], training_label[i], validation_label[i] = train_test_split(training_data[i], training_label[i], test_size=0.1, random_state=2022)
    
    # scio.savemat('landmine.mat', {'landmine_train_data': np.array(training_data), 'landmine_train_label': np.array(training_label), \
    # "landmine_valid_data": np.array(validation_data), "landmine_valid_label": np.array(validation_label), \
    # "landmine_test_data": np.array(testing_data), "landmine_test_label": np.array(testing_label)})
    # scio.savemat('Landmine.mat', {'data': data['feature'][0], 'label': data['label'][0]})
    return training_data, testing_data, training_label, testing_label


if __name__ == '__main__':
    # data = scio.loadmat(data_dir)
    landmine_test(data_dir)
