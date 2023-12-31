#!/usr/bin/python
# If you use this script in your studies, here is the reference to cite:
# G. Altay, "Tensorflow Based Deep Learning Model and Snakemake Workflow for Peptide-Protein Binding Predictions", bioRxiv, 2018.
# This software is provided with absolutely no warranty. 

import numpy as np
import pandas as pd
import random as rnd
import scipy.io as scio
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

test_ratio = 0.2
##################################################################
### all the possible sequence letters
allSequences = 'ACEDGFIHKMLNQPSRTWVY'  # [0, 1,...,19]
# Establish a mapping from letters to integers
char2int = dict((c, i) for i, c in enumerate(allSequences))
int2char = dict((i, c) for i, c in enumerate(allSequences))
##################################################################
###
def Pept_OneHotMap(peptideSeq):
    """ maps amino acid into its numerical index
    USAGE
    Pept_OneHotMap('A')
    array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    # integer encode input data
    integer_encoded = [char2int[char] for char in peptideSeq]
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
    	letter = [0 for _ in range(len(allSequences))]
    	letter[value] = 1
    	onehot_encoded.append(letter)
    return np.asarray(onehot_encoded)
    # print(len(integer_encoded))
    # return np.asarray(integer_encoded)

#####

def getdata_onehot(testdatafile):
    ### READ in test dataset
    """ Reads the test data file and extracts allele subtype,
            peptide length, and measurement type. Returns these information
            along with the peptide sequence and target values.
    """
    print("Test peptide name: ", testdatafile)
    import os, re
    test_set = os.path.join("./DATA", "test_data", testdatafile )
    print("test_set name: ", test_set)
    test_data = pd.read_csv(test_set, delim_whitespace=True)
    #test_data = pd.read_csv('./DATA/test_data/A0202',sep="\t")
    '''
    [77 rows x 16 columns]
    >>> test_data.columns
    Index(['Date', 'IEDB', 'Allele', 'Peptide_length', 'Measurement_type',
           'Peptide_seq', 'Measurement_value', 'NetMHCpan', 'SMM', 'ANN', 'ARB',
           'SMMPMBEC', 'IEDB_Consensus', 'NetMHCcons', 'PickPocket', 'mhcflurry'],
          dtype='object')
    '''
    import re
    peptide = re.search(r'[A-Z]\*\d{2}:\d{2}', test_data['Allele'][0]).group()
    print(peptide)
    peptide_length = len(test_data['Peptide_seq'][0])
    measurement_type = test_data['Measurement_type'][0]

    if measurement_type.lower() == 'binary':
        test_data['Measurement_value'] = np.where(test_data.Measurement_value == 1.0, 1, 0)
    else:
        test_data['Measurement_value'] = np.where(test_data.Measurement_value < 500.0, 1, 0)

    test_label = test_data.Measurement_value

    ### end of reading test dataset

    ### NOW, READ training dataset
    """ Reads the training data file and returns the sequences of peptides
        and target values
    """
    train_set = './DATA/train_data/proteins.txt'
    df = pd.read_csv(train_set, delim_whitespace=True, header=0)
    '''
    [141224 rows x 3 columns]>
    >>> df.columns
    Index(['Peptide', 'HLA', 'BindingCategory'], dtype='object')
    '''
    #df.columns = ['sequence', 'HLA', 'target']
    # build training matrix
    #df.shape #(141224, 3)
    df = df[df.HLA == peptide]
    #df.shape #(14736, 3)
    df = df[df['Peptide'].map(len) == peptide_length]
    # df.shape #(10549, 3)
    # remove any peptide with  unknown variables
    df = df[df.Peptide.str.contains('X') == False]
    df = df[df.Peptide.str.contains('B') == False]
    #df.shape  #(10547, 3)
    # remap target values to 1's and 0's
    df['BindingCategory'] = np.where(df.BindingCategory == 1, 1, 0)
    ###
    """ Reads the specified train and test files and return the
            relevant design and target matrix for the learning pipeline.
    """
    # map the training peptide sequences to their integer index
    featureMatrix = np.empty((0, peptide_length,len(allSequences)), int)
    for num in range(len(df.Peptide)):
        # print([Pept_OneHotMap(df.Peptide.iloc[num])])
        featureMatrix = np.append(featureMatrix, [Pept_OneHotMap(df.Peptide.iloc[num])], axis=0)

    # map the test peptide sequences to their integer index
    testMatrix = np.empty((0, peptide_length,len(allSequences)), int)
    for num in range(len(test_data.Peptide_seq)):
        testMatrix = np.append(testMatrix, [Pept_OneHotMap(test_data.Peptide_seq.iloc[num])], axis=0)
    ###
    trainlen = len(featureMatrix)
    testlen = len(testMatrix)
    ss1 = list(range(trainlen))
    rnd.shuffle(ss1)
    valsize = 20  #Validation set size is 20 for 3 validations dataset
    X_val1 = featureMatrix[ss1[0:valsize]]
    Y_val1 = df['BindingCategory'].iloc[ss1[0:valsize]]
    X_val2 = featureMatrix[ss1[valsize:(2*valsize)]]
    Y_val2 = df['BindingCategory'].iloc[ss1[valsize:(2*valsize)]]
    X_val3 = featureMatrix[ss1[(2*valsize):(3*valsize)]]
    Y_val3 = df['BindingCategory'].iloc[ss1[(2*valsize):(3*valsize)]]
    labelmatrix = df.BindingCategory
    featureMatrix = np.delete(featureMatrix, ss1[0:(3*valsize)], axis=0)
    labelmatrix = labelmatrix.drop(labelmatrix.index[ss1[0:(3*valsize)]])
    # combine training and test datasets
    datasets={}
    datasets['X_train'] = featureMatrix
    datasets['Y_train'] = labelmatrix.values #df.BindingCategory.as_matrix()
    datasets['X_test'] = testMatrix
    datasets['Y_test'] = test_data.Measurement_value.values
    datasets['X_val1'] = X_val1
    datasets['Y_val1'] = Y_val1.values
    datasets['X_val2'] = X_val2
    datasets['Y_val2'] = Y_val2.values
    datasets['X_val3'] = X_val3
    datasets['Y_val3'] = Y_val3.values
    return datasets


#getdata_onehot function will return labes as 1 or 0 as a vector. To convert them into onehot encoded two-class format use this function
#function to convert output labels, which are 1 or 0, to two class outputs as [1,0] or [0,1]
def binary2onehot(yy):
    yy2= np.zeros((len(yy),2), dtype=int) #yy2.shape #(10547, 2)
    for num in range(len(yy)):
        if yy[num]==1:
            yy2[num,0]=1
        else:
            yy2[num,1]=1
    return yy2
#
#This function helps getting and arranging minibatch indices for DL model input at each epoch
#If that batch size and data size do not match, it randomly samples indices from the previous big pool
# and append to the remaining indices. In that case, there is one more iteration to complete, as data size seems a bit bigger.
def getIndicesofMinibatchs(featuredata, featurelabels, batchsize_, isShuffle=True):
    # by default and always in deep learning apps, shuffle should be True
    # usage: x=read_data.getIndicesofMinibatchs(featuredata=data['X_train'],
    #                featurelabels=data['Y_train'], batchsize_=40, isShuffle=True)
    datalength=len(featuredata)
    if isShuffle==True:
        tmpindx = np.arange(datalength)
        np.random.shuffle(tmpindx)
    tmp = datalength % batchsize_
    if tmp !=0:
        tmp2 = np.random.choice(tmpindx[range(datalength-tmp)], (batchsize_ - tmp ))
        tmpindx=np.append(tmpindx,tmp2)
    return tmpindx


def mhc_split(path):
    import os
    data = []
    label = []

    for root, dirs, files in os.walk(path): 
        # print(files)  #当前路径下所有非目录子文件 
        for file in files:
            data_temp = getdata_onehot(file)
            # print(len(data['X_train']))
            data.append(data_temp['X_train'])
            label.append(np.expand_dims(data_temp['Y_train'], 1))
            
    task_num = len(data)
    X_tr = [i for i in range(task_num)]
    # X_va = [i for i in range(task_num)]
    X_te = [i for i in range(task_num)]

    Y_tr = [i for i in range(task_num)]
    # Y_va = [i for i in range(task_num)]
    Y_te = [i for i in range(task_num)]

    for i in range(task_num):
        X_tr[i], X_te[i], Y_tr[i], Y_te[i] = train_test_split(data[i], label[i], test_size=test_ratio, random_state=2022)
        # X_tr[i], X_va[i], Y_tr[i], Y_va[i] = train_test_split(X_tr[i], Y_tr[i], test_size=0.1, random_state=2022)

        X_tr[i] = X_tr[i].reshape((X_tr[i].shape[0], X_tr[i].shape[1] * X_tr[i].shape[2]))
        # X_va[i] = X_va[i].reshape((X_va[i].shape[0], X_va[i].shape[1] * X_va[i].shape[2]))
        X_te[i] = X_te[i].reshape((X_te[i].shape[0], X_te[i].shape[1] * X_te[i].shape[2]))

    # scio.savemat('MHCI_split.mat', {'MHCI_train_data': np.array(training_data), 'MHCI_train_label': np.array(training_label), "MHCI_valid_data": np.array(validation_data), "MHCI_valid_label": np.array(validation_label), \
    #  "MHCI_test_data": np.array(testing_data), "MHCI_test_label": np.array(testing_label)})
    # scio.savemat('MHC-I.mat', {'data': data, 'label': label})
    return X_tr, Y_tr, X_te, Y_te
    

def mhc_split_valid(path):  # k fold cross-validation
    task_num = 8
    cross_validation_num = 5
    import os
    data_temp = []
    label_temp = []
    for root, dirs, files in os.walk(path): 
        # print(files)  #当前路径下所有非目录子文件 
        for file in files:
            temp = getdata_onehot(file)
            # print(len(data['X_train']))
            data_temp.append(temp['X_train'])
            label_temp.append(np.expand_dims(temp['Y_train'], 1))

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

            training_data[k][i] = training_data[k][i].reshape((training_data[k][i].shape[0], training_data[k][i].shape[1] * training_data[k][i].shape[2]))
            validation_data[k][i] = validation_data[k][i].reshape((validation_data[k][i].shape[0], validation_data[k][i].shape[1] * validation_data[k][i].shape[2]))
            # print(np.array(training_label[k][i]).shape)
            k += 1
    #######################################################################################################
    return training_data, validation_data, training_label, validation_label


if __name__ == "__main__":
    training_data, training_label, testing_data, testing_label = mhc_split('./DATA/test_data')
    