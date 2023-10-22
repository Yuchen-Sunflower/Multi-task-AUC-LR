import torch
from torch import tensor, linalg, matmul
import data_load
import read_data
import data_load_dight
import data_load_Simu
import AUC
import nni
import time
import pandas as pd
import numpy as np
import scipy.io as scio

# ------------------------------ data address ----------------------------------
mhc_dir = './DATA/test_data'
landmine_dir = 'LandmineData.mat'
files = [r'USPS\task1.mat', r'USPS\task2.mat', r'USPS\task3.mat', r'USPS\task4.mat', r'USPS\task5.mat']
Simulated_10 = r'Simulated\Simulated_10.mat'
Simulated_40 = r'Simulated\Simulated_40.mat'
Simulated_50 = r'Simulated\Simulated_50.mat'
Simulated_80 = r'Simulated\Simulated_80.mat'
Wmat = r'Simulated\W.mat'
# W_best = scio.loadmat(Wmat)

K = 500  # '''Number of stages'''
feature_num = 80   # Number of feature

# ------------------------ dataset split and load -----------------------------
# 用training_data, testing_data, training_label, testing_label即可
# training_data, validation_data, training_label, validation_label = data_load.landmine_valid(dataFile=landmine_dir)
# training_data, testing_data, training_label, testing_label = data_load.landmine_test(dataFile=landmine_dir)

# training_data, validation_data, training_label, validation_label = read_data.mhc_split_valid(mhc_dir)
# training_data, training_label, testing_data, testing_label = read_data.mhc_split(mhc_dir)

# training_data, validation_data, training_label, validation_label = data_load_dight.dight_split_valid(files)
# training_data, testing_data, training_label, testing_label = data_load_dight.dight_split(files)

# Simulated_10, Simulated_40, Simulated_50, Simulated_80
# training_data, validation_data, training_label, validation_label = data_load_Simu.Simulated_split_valid(Simulated_80)
training_data, testing_data, training_label, testing_label = data_load_Simu.Simulated_split(Simulated_80)
# -----------------------------------------------------------------------------


def reAUC_valid(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data[0])  # ''' task number'''
    r = args['r']

    iter_time = []
    iter_acc = []

    # training
    # print('Start Training ==>')
    for n in range(data_load.cross_validation_num):
        # parameters initialization
        M = torch.randn((d, r), requires_grad=True, dtype=torch.float32)
        N = torch.randn((T, r), requires_grad=True, dtype=torch.float32)
        a = torch.zeros((T, 1), requires_grad=True, dtype=torch.float32)
        b = torch.zeros((T, 1), requires_grad=True, dtype=torch.float32)
        alpha = torch.zeros((T, 1), requires_grad=True, dtype=torch.float32)

        final_auc = 0
        pred = []
        label = []

        model = AUC.reAUC(M, N, a, b, alpha, args['lambda'])
        optimizer = torch.optim.SGD([
            {'params': model.M, 'lr': args['lr_m'], 'momentum': 0.9}, 
            {'params': model.N, 'lr': args['lr_n'], 'momentum': 0.9}, 
            {'params': model.a}, 
            {'params': model.b},
            {'params': model.alpha, 'lr': -args['lr_alpha']}
        ], lr=args['lr_v'])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                        mode='max',
        #                                                        factor=0.1,
        #                                                        patience=50,
        #                                                        threshold=1e-4)
        for k in range(1, K + 1):
            # AUC.adjust_learning_rate(optimizer, args['lr'], k) 
            start = time.process_time()
            loss = model(training_data[n], training_label[n])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.a, model.b, model.alpha = AUC.update_param(model.a, model.a.grad, model.b, model.b.grad,
                model.alpha, model.alpha.grad, args['lam_v'], args['lr_v'], args['lr_alpha'])

            with torch.no_grad():
                # loss = model(testing_data, testing_label) validation
                for i in range(T):
                    res = AUC.h(torch.matmul(model.M, model.N.t()).t()[i],
                                torch.FloatTensor(validation_data[n][i]).t()).squeeze(0).tolist()
                    for j in range(len(res)):
                        pred.append(res[j])
                        label.append(validation_label[n][i][j])

                test_auc = AUC.AUC(label, pred)
                # scheduler.step(test_auc)
                nni.report_intermediate_result(test_auc)
                print('{}-th Fold, Test AUC {}/{}: {:.3f}'.format(n, k, K, test_auc))
                final_auc = max(test_auc, final_auc)

                pred.clear()
                label.clear()
            end = time.process_time() 
            iter_time.append(end - start)        
        iter_acc.append(final_auc)

    print('avg best auc: {:.3f}'.format(sum(iter_acc) / len(iter_acc)))
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    nni.report_final_result(sum(iter_acc) / len(iter_acc))


def reAUC_test(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''
    r = args['r']

    # training
    # print('Start Training ==>')
    
    # parameters initialization
    M = torch.randn((d, r), requires_grad=True, dtype=torch.float32)
    N = torch.randn((T, r), requires_grad=True, dtype=torch.float32)
    a = torch.zeros((T, 1), requires_grad=True, dtype=torch.float32)
    b = torch.zeros((T, 1), requires_grad=True, dtype=torch.float32)
    alpha = torch.zeros((T, 1), requires_grad=True, dtype=torch.float32)

    final_auc = 0
    pred = []
    label = []
    iter_time = []
    data = []
    model = AUC.reAUC(M, N, a, b, alpha, args['lambda'])
    optimizer = torch.optim.SGD([
        {'params': model.M, 'lr': args['lr_m'], 'momentum': 0.9}, 
        {'params': model.N, 'lr': args['lr_n'], 'momentum': 0.9}, 
        {'params': model.a}, 
        {'params': model.b},
        {'params': model.alpha, 'lr': -args['lr_alpha']}
    ], lr=args['lr_v'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode='max',
    #                                                        factor=0.1,
    #                                                        patience=50,
    #                                                        threshold=1e-4)
    for k in range(1, K + 1):
        start = time.process_time()
        loss = model(training_data, training_label)
        data.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.a, model.b, model.alpha = AUC.update_param(model.a, model.a.grad, model.b, model.b.grad,
             model.alpha, model.alpha.grad, args['lam_v'], args['lr_v'], args['lr_alpha'])
        
        with torch.no_grad():
            # loss = model(testing_data, testing_label) validation
            for i in range(T):
                res = AUC.h(torch.matmul(model.M, model.N.t()).t()[i],
                            torch.FloatTensor(testing_data[i]).t()).squeeze(0).tolist()
                for j in range(len(res)):
                    pred.append(res[j])
                    label.append(testing_label[i][j])

            test_auc = AUC.AUC(label, pred)
            # scheduler.step(test_auc)
            print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
            final_auc = max(test_auc, final_auc)
            # print('Test loss: {:.3f}'.format(model(testing_data, testing_label).item()))
            # print('Current Best loss: {:.3f}'.format(best_loss))

            pred.clear()
            label.clear()
        end = time.process_time() 
        iter_time.append(end-start)

    print('best auc: {:.3f}'.format(final_auc))  # best auc
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))  # Average iteration time
    # print('E.W.2: {:.4f}'.format(linalg.norm(AUC.h(model.M, model.N.t()) - tensor(W_best['W']).t()) ** 2 / T))  # E.W
    # print('E.W.: {:.4f}'.format(linalg.norm(AUC.h(model.M, model.N.t()) - tensor(W_best['W']).t()) / T))  # E.W
    
    # scio.savemat('W_80.mat', {'W': torch.matmul(model.M, model.N.t()).tolist()})
    return final_auc, sum(iter_time) / len(iter_time)
    

def STL_CE_valid(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data[0])  # ''' task number'''
    iter_acc = []
    iter_time = []
    for n in range(data_load.cross_validation_num):
        w = torch.randn((d, 1), requires_grad=True, dtype=torch.float32)

        pred = []
        label = []
        final_auc = 0

        model = []
        optimizer = []

        for i in range(T):
            model.append(AUC.STL_CE(w))
            optimizer.append(torch.optim.SGD(model[i].parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay']))
        citerion = torch.nn.MSELoss()

        for k in range(1, K + 1):
            start = time.process_time()
            # training
            for i in range(T):
                outputs = model[i](training_data[n][i])
                loss = citerion(outputs, tensor(training_label[n][i], dtype=torch.float32))
                optimizer[i].zero_grad()
                loss.backward()
                optimizer[i].step()

            # testing
            with torch.no_grad():
                for i in range(T):
                    res = model[i](validation_data[n][i]).squeeze(0).tolist()  # validation
                    for j in range(len(res)):
                        pred.append(res[j])
                        label.append(validation_label[n][i][j])

                test_auc = AUC.AUC(label, pred)
                nni.report_intermediate_result(test_auc)
                print('{}-th Fold, Test AUC {}/{}: {:.3f}'.format(n, k, K, test_auc))
                final_auc = max(test_auc, final_auc)

                pred.clear()
                label.clear()

            end = time.process_time() 
            iter_time.append(end-start)
        iter_acc.append(final_auc)

    print('avg best auc: {:.3f}'.format(sum(iter_acc) / len(iter_acc)))
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    nni.report_final_result(sum(iter_acc) / len(iter_acc))


def STL_CE_test(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''
    
    w = torch.ones((d, 1), requires_grad=True, dtype=torch.float32)

    pred = []
    label = []
    final_auc = 0
    iter_time = []

    model = []
    optimizer = []

    for i in range(T):
        model.append(AUC.STL_CE(w))
        optimizer.append(torch.optim.SGD(model[i].parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay']))
    citerion = torch.nn.MSELoss()

    for k in range(1, K + 1):
        start = time.process_time()
        # training
        for i in range(T): 
            outputs = model[i](training_data[i])
            loss = citerion(outputs, tensor(training_label[i], dtype=torch.float32))
            optimizer[i].zero_grad()
            loss.backward()
            optimizer[i].step()

        # testing
        with torch.no_grad():
            for i in range(T):
                res = model[i](testing_data[i]).squeeze(0).tolist()  # validation
                for j in range(len(res)):
                    pred.append(res[j])
                    label.append(testing_label[i][j])

            test_auc = AUC.AUC(label, pred)
            print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
            final_auc = max(test_auc, final_auc)

            pred.clear()
            label.clear()

        end = time.process_time() 
        iter_time.append(end-start)

    print('best auc: {:.3f}'.format(test_auc))
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    return final_auc, sum(iter_time) / len(iter_time)


def MTL_CE_valid(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data[0])  # ''' task number'''
    iter_acc = []
    iter_time = []
    for n in range(data_load.cross_validation_num):
        W = torch.randn((d, T), requires_grad=True, dtype=torch.float32)

        pred = []
        label = []
        final_auc = 0

        model = AUC.MTL_CE(W)
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay'])
        citerion = torch.nn.MSELoss()

        for k in range(1, K + 1):
            start = time.process_time()
            # training
            outputs = model(training_data[n])
            for i, res in enumerate(outputs):
                loss = citerion(res.t().unsqueeze(1), tensor(training_label[n][i], dtype=torch.float32))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # testing
            with torch.no_grad():
                for i in range(T):  # validation
                    res = AUC.h(model.W.t()[i], torch.FloatTensor(validation_data[n][i]).t()).squeeze(0).tolist()
                    for j in range(len(res)):
                        pred.append(res[j])
                        label.append(validation_label[n][i][j])

                test_auc = AUC.AUC(label, pred)
                nni.report_intermediate_result(test_auc)
                print('{}-th Fold, Test AUC {}/{}: {:.3f}'.format(n, k, K, test_auc))
                final_auc = max(test_auc, final_auc)

                pred.clear()
                label.clear()

            end = time.process_time() 
            iter_time.append(end-start)
        iter_acc.append(final_auc)
        
    print('avg best auc: {:.3f}'.format(sum(iter_acc) / len(iter_acc)))
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    nni.report_final_result(sum(iter_acc) / len(iter_acc))


def MTL_CE_test(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''

    W = torch.randn((d, T), requires_grad=True, dtype=torch.float32)

    pred = []
    label = []
    iter_time = []
    final_auc = 0

    model = AUC.MTL_CE(W)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay'])
    citerion = torch.nn.MSELoss()

    for k in range(1, K + 1):
        start = time.process_time()
        # training
        outputs = model(training_data)
        for i, res in enumerate(outputs):
            loss = citerion(res.t().unsqueeze(1), tensor(training_label[i], dtype=torch.float32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # testing
        with torch.no_grad():
            for i in range(T):  # validation
                res = AUC.h(model.W.t()[i], torch.FloatTensor(testing_data[i]).t()).squeeze(0).tolist()
                for j in range(len(res)):
                    pred.append(res[j])
                    label.append(testing_label[i][j])

            test_auc = AUC.AUC(label, pred)
            print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
            final_auc = max(test_auc, final_auc)

            pred.clear()
            label.clear()

        end = time.process_time() 
        iter_time.append(end-start)
        
    print('best auc: {:.3f}'.format(final_auc))
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    # print('E.W.2: {:.4f}'.format(linalg.norm(model.W - tensor(W_best['W']).t()) ** 2 / T))  # E.W.2
    # print('E.W.: {:.4f}'.format(linalg.norm(model.W - tensor(W_best['W']).t()) / T))  # E.W.
    return final_auc, sum(iter_time) / len(iter_time)


def MTL_CE_LR_valid(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data[0])  # ''' task number'''
    r = args['r']

    iter_time = []
    iter_acc = []
    for n in range(data_load.cross_validation_num):
        M = torch.randn((d, r), requires_grad=True, dtype=torch.float32)
        N = torch.randn((T, r), requires_grad=True, dtype=torch.float32)

        pred = []
        label = []
        final_auc = 0

        model = AUC.MTL_CE_LR(M, N)
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)
        citerion = torch.nn.MSELoss()

        for k in range(1, K + 1):
            start = time.process_time()
            # training
            outputs = model(training_data)
            loss = torch.tensor([[0]], requires_grad=True, dtype=torch.float32)
            for i, res in enumerate(outputs):
                loss = loss + citerion(res.t().unsqueeze(1), tensor(training_label[n][i], dtype=torch.float32))
            
            loss = loss / T + args['lambda'] / 2 * (linalg.norm(model.M)**2 + linalg.norm(model.N)**2) - args['lambda'] * linalg.norm(matmul(model.M, model.N.t()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # testing
            with torch.no_grad():
                for i in range(T):
                    res = AUC.h(torch.matmul(model.M, model.N.t()).t()[i],  # validation
                                torch.FloatTensor(testing_data[n][i]).t()).squeeze(0).tolist()
                    for j in range(len(res)):
                        pred.append(res[j])
                        label.append(testing_label[n][i][j][0])

                test_auc = AUC.AUC(label, pred)
                nni.report_intermediate_result(test_auc)
                print('{}-th Fold, Test AUC {}/{}: {:.3f}'.format(n, k, K, test_auc))
                final_auc = max(test_auc, final_auc)

                pred.clear()
                label.clear()

            end = time.process_time() 
            iter_time.append(end-start)
        iter_acc.append(final_auc)
        
    print('avg best auc: {:.3f}'.format(sum(iter_acc) / len(iter_acc)))
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    nni.report_final_result(final_auc)
        

def MTL_CE_LR_test(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''
    r = args['r']

    M = torch.randn((d, r), requires_grad=True, dtype=torch.float32)
    N = torch.randn((T, r), requires_grad=True, dtype=torch.float32)

    pred = []
    label = []
    iter_time = []
    final_auc = 0

    model = AUC.MTL_CE_LR(M, N)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)
    citerion = torch.nn.MSELoss()

    for k in range(1, K + 1):
        start = time.process_time()
        # training
        outputs = model(training_data)
        loss = torch.tensor([[0]], requires_grad=True, dtype=torch.float32)
        for i, res in enumerate(outputs):
            loss = loss + citerion(res.t().unsqueeze(1), tensor(training_label[i], dtype=torch.float32))
        
        loss = loss / T + args['lambda'] / 2 * (linalg.norm(model.M)**2 + linalg.norm(model.N)**2) - args['lambda'] * linalg.norm(matmul(model.M, model.N.t()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # testing
        with torch.no_grad():
            for i in range(T):
                res = AUC.h(torch.matmul(model.M, model.N.t()).t()[i],  # validation
                            torch.FloatTensor(testing_data[i]).t()).squeeze(0).tolist()
                for j in range(len(res)):
                    pred.append(res[j])
                    label.append(testing_label[i][j])

            test_auc = AUC.AUC(label, pred)
            print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
            final_auc = max(test_auc, final_auc)

            pred.clear()
            label.clear()

        end = time.process_time() 
        iter_time.append(end-start)
        
    print('best auc: {:.3f}'.format(final_auc))
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    # print('E.W.2: {:.4f}'.format(linalg.norm(AUC.h(model.M, model.N.t()) - tensor(W_best['W']).t()) ** 2 / T))  # E.W
    # print('E.W.: {:.4f}'.format(linalg.norm(AUC.h(model.M, model.N.t()) - tensor(W_best['W']).t()) / T))  # E.W
    return final_auc, sum(iter_time) / len(iter_time)


def STL_AUC_valid(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''

    w = torch.randn((d, 1), requires_grad=True, dtype=torch.float32)

    pred = []
    label = []
    iter_time = []
    final_auc = 0

    pos_data = []
    pos_label = []
    neg_data = []
    neg_label = []

    model = []
    optimizer = []

    for n in range(data_load.cross_validation_num):
        for i in range(T):
            model.append(AUC.AUROC(w))
            optimizer.append(torch.optim.SGD(model[i].parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay']))

        for k in range(1, K + 1):
            start = time.process_time()
            # training
            for i in range(T):
                for j in range(len(training_data[i])):
                    if training_label[i][j][0] == 1:
                        pos_data.append(training_data[n][i][j])
                        pos_label.append(training_label[n][i][j])
                    else:
                        neg_data.append(training_data[n][i][j])
                        neg_label.append(training_label[n][i][j])

                pos_sampler = torch.utils.data.RandomSampler(data_source=pos_data, num_samples=25)
                neg_sampler = torch.utils.data.RandomSampler(data_source=neg_data, num_samples=25)

                outputs = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
                for p in pos_sampler:
                    for q in neg_sampler:
                        outputs = outputs + model[i](pos_data[p], neg_data[q])

                outputs = outputs / len(pos_sampler) / len(neg_sampler)
                optimizer[i].zero_grad()
                outputs.backward()
                optimizer[i].step()

                pos_data.clear()
                pos_label.clear()
                neg_data.clear()
                neg_label.clear()

            # testing
            with torch.no_grad():
                for i in range(T):
                    for j in range(len(validation_data[i])):  # validation
                        res = AUC.h(model[i].w.t(), torch.FloatTensor(validation_data[i][j]).t()).item()
                        pred.append(res)
                        label.append(validation_label[i][j])

                test_auc = AUC.AUC(label, pred)
                # nni.report_intermediate_result(test_auc)
                print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
                final_auc = max(test_auc, final_auc)

                pred.clear()
                label.clear()

            end = time.process_time() 
            iter_time.append(end-start)
        
    print('best auc: {:.3f}'.format(final_auc))
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    # nni.report_final_result(final_auc)


def STL_AUC_test(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''

    w = torch.randn((d, 1), requires_grad=True, dtype=torch.float32)

    pred = []
    label = []
    iter_time = []
    final_auc = 0

    pos_data = []
    pos_label = []
    neg_data = []
    neg_label = []

    model = []
    optimizer = []

    for i in range(T):
        model.append(AUC.AUROC(w))
        optimizer.append(torch.optim.SGD(model[i].parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay']))

    for k in range(1, K + 1):
        start = time.process_time()
        # training
        for i in range(T):
            for j in range(training_data[i].shape[0]):
                if training_label[i][j] == 1:
                    pos_data.append(training_data[i][j])
                    pos_label.append(training_label[i][j])
                else:
                    neg_data.append(training_data[i][j])
                    neg_label.append(training_label[i][j])

            pos_sampler = torch.utils.data.RandomSampler(data_source=pos_data, num_samples=15)
            neg_sampler = torch.utils.data.RandomSampler(data_source=neg_data, num_samples=15)

            outputs = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
            for p in pos_sampler:
                for q in neg_sampler:
                    outputs = outputs + model[i](pos_data[p], neg_data[q])

            outputs = outputs / len(pos_sampler) / len(neg_sampler)
            optimizer[i].zero_grad()
            outputs.backward()
            optimizer[i].step()

            pos_data.clear()
            pos_label.clear()
            neg_data.clear()
            neg_label.clear()

        # testing
        with torch.no_grad():
            for i in range(T):
                for j in range(testing_data[i].shape[0]):  # validation
                    res = AUC.h(model[i].w.t(), torch.FloatTensor(testing_data[i][j]).t()).item()
                    pred.append(res)
                    label.append(testing_label[i][j])

            test_auc = AUC.AUC(label, pred)
            # nni.report_intermediate_result(test_auc)
            print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
            final_auc = max(test_auc, final_auc)
            # print('Test loss: {:.3f}'.format(model(testing_data, testing_label).item()))
            # print('Current Best loss: {:.3f}'.format(best_loss))

            pred.clear()
            label.clear()

        end = time.process_time() 
        iter_time.append(end-start)
        
    print('best auc: {:.3f}'.format(final_auc))
    print('avg iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    # nni.report_final_result(final_auc)
    return final_auc, sum(iter_time) / len(iter_time)


def MTL_AUC_valid(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''

    W = torch.randn((d, T), requires_grad=True, dtype=torch.float32)

    pred = []
    label = []
    iter_time = []
    final_auc = 0

    pos_data = []
    pos_label = []
    neg_data = []
    neg_label = []

    model = AUC.MTL_AUC(W)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    for k in range(1, K + 1):
        start = time.process_time()
        loss = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
        # training
        for i in range(T):
            output = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
            for j in range(training_data[i].shape[0]):
                if training_label[i][j][0] == 1:
                    pos_data.append(training_data[i][j])
                    pos_label.append(training_label[i][j][0])
                else:
                    neg_data.append(training_data[i][j])
                    neg_label.append(training_label[i][j][0])

            pos_sampler = torch.utils.data.RandomSampler(data_source=pos_data, num_samples=25)
            neg_sampler = torch.utils.data.RandomSampler(data_source=neg_data, num_samples=25)

            for p in pos_sampler:
                for q in neg_sampler:
                    output = output + model(pos_data[p], neg_data[q], i)
                    
            loss = loss + output / len(pos_sampler) / len(neg_sampler)

            pos_data.clear()
            pos_label.clear()
            neg_data.clear()
            neg_label.clear()

        loss = loss / T
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # testing
        with torch.no_grad():  # validation
            for i in range(T):
                res = AUC.h(model.W.t()[i], torch.FloatTensor(testing_data[i]).t()).squeeze(0).tolist()
                for j in range(len(res)):
                    pred.append(res[j])
                    label.append(testing_label[i][j][0])

            test_auc = AUC.AUC(label, pred)
            # scheduler.step()
            # nni.report_intermediate_result(test_auc)
            print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
            final_auc = max(test_auc, final_auc)
            # print('Test loss: {:.3f}'.format(model(testing_data, testing_label).item()))
            # print('Current Best loss: {:.3f}'.format(best_loss))

            pred.clear()
            label.clear()

        end = time.process_time() 
        iter_time.append(end-start)
        
    print('best auc: {:.3f}'.format(final_auc))
    print('each iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    # nni.report_final_result(final_auc)


def MTL_AUC_test(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''

    W = torch.randn((d, T), requires_grad=True, dtype=torch.float32)

    pred = []
    label = []
    iter_time = []
    final_auc = 0

    pos_data = []
    pos_label = []
    neg_data = []
    neg_label = []

    model = AUC.MTL_AUC(W)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    for k in range(1, K + 1):
        start = time.process_time()
        loss = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
        # training
        for i in range(T):
            output = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
            for j in range(training_data[i].shape[0]):
                if training_label[i][j] == 1:
                    pos_data.append(training_data[i][j])
                    pos_label.append(training_label[i][j])
                else:
                    neg_data.append(training_data[i][j])
                    neg_label.append(training_label[i][j])

            pos_sampler = torch.utils.data.RandomSampler(data_source=pos_data, num_samples=15)
            neg_sampler = torch.utils.data.RandomSampler(data_source=neg_data, num_samples=15)

            for p in pos_sampler:
                for q in neg_sampler:
                    output = output + model(pos_data[p], neg_data[q], i)
                    
            loss = loss + output / len(pos_sampler) / len(neg_sampler)

            pos_data.clear()
            pos_label.clear()
            neg_data.clear()
            neg_label.clear()

        loss = loss / T
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # testing
        with torch.no_grad():  # validation
            for i in range(T):
                res = AUC.h(model.W.t()[i], torch.FloatTensor(testing_data[i]).t()).squeeze(0).tolist()
                for j in range(len(res)):
                    pred.append(res[j])
                    label.append(testing_label[i][j])

            test_auc = AUC.AUC(label, pred)
            # scheduler.step()
            # nni.report_intermediate_result(test_auc)
            print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
            final_auc = max(test_auc, final_auc)
            # print('Test loss: {:.3f}'.format(model(testing_data, testing_label).item()))
            # print('Current Best loss: {:.3f}'.format(best_loss))

            pred.clear()
            label.clear()

        end = time.process_time() 
        iter_time.append(end-start)
        
    print('best auc: {:.3f}'.format(final_auc))
    print('each iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    print('E.W.2: {:.4f}'.format(linalg.norm(model.W - tensor(W_best['W']).t()) ** 2 / T))  # E.W.2
    print('E.W.: {:.4f}'.format(linalg.norm(model.W - tensor(W_best['W']).t()) / T))  # E.W.
    # nni.report_final_result(final_auc)
    return final_auc, sum(iter_time) / len(iter_time)


def MTL_AUC_LR_valid(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''
    r = args['r']

    M = torch.randn((d, r), requires_grad=True, dtype=torch.float32)
    N = torch.randn((T, r), requires_grad=True, dtype=torch.float32)

    pred = []
    label = []
    iter_time = []
    final_auc = 0

    pos_data = []
    pos_label = []
    neg_data = []
    neg_label = []

    model = AUC.MTL_AUC_LR(M, N)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)

    for k in range(1, K + 1):
        start = time.process_time()
        loss = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
        # training
        for i in range(T):
            output = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
            for j in range(training_data[i].shape[0]):
                if training_label[i][j][0] == 1:
                    pos_data.append(training_data[i][j])
                    pos_label.append(training_label[i][j][0])
                else:
                    neg_data.append(training_data[i][j])
                    neg_label.append(training_label[i][j][0])

            pos_sampler = torch.utils.data.RandomSampler(data_source=pos_data, num_samples=25)
            neg_sampler = torch.utils.data.RandomSampler(data_source=neg_data, num_samples=25)

            for p in pos_sampler:
                for q in neg_sampler:
                    output = output + model(pos_data[p], neg_data[q], i)
                    
            loss = loss + output / len(pos_sampler) / len(neg_sampler)

            pos_data.clear()
            pos_label.clear()
            neg_data.clear()
            neg_label.clear()

        loss = loss / T + args['lambda'] / 2 * (linalg.norm(model.M) ** 2 + linalg.norm(model.N) ** 2) - args['lambda'] * linalg.norm(matmul(model.M, model.N.t()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # testing
        with torch.no_grad():
            for i in range(T):
                res = AUC.h(torch.matmul(model.M, model.N.t()).t()[i],  # validation
                            torch.FloatTensor(testing_data[i]).t()).squeeze(0).tolist()
                for j in range(len(res)):
                    pred.append(res[j])
                    label.append(testing_label[i][j][0])

            test_auc = AUC.AUC(label, pred)
            # nni.report_intermediate_result(test_auc)
            print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
            final_auc = max(test_auc, final_auc)
            # print('Test loss: {:.3f}'.format(model(testing_data, testing_label).item()))
            # print('Current Best loss: {:.3f}'.format(best_loss))

            pred.clear()
            label.clear()

        end = time.process_time() 
        iter_time.append(end-start)
        
    print('best auc: {:.3f}'.format(final_auc))
    print('each iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    # nni.report_final_result(final_auc)


def MTL_AUC_LR_test(args):
    d = feature_num  # '''feature numbers'''
    T = len(training_data)  # ''' task number'''
    r = args['r']

    M = torch.randn((d, r), requires_grad=True, dtype=torch.float32)
    N = torch.randn((T, r), requires_grad=True, dtype=torch.float32)

    pred = []
    label = []
    iter_time = []
    final_auc = 0

    pos_data = []
    pos_label = []
    neg_data = []
    neg_label = []

    model = AUC.MTL_AUC_LR(M, N)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)

    for k in range(1, K + 1):
        start = time.process_time()
        loss = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
        # training
        for i in range(T):
            output = tensor([[0.0]], requires_grad=True, dtype=torch.float32)
            for j in range(training_data[i].shape[0]):
                if training_label[i][j] == 1:
                    pos_data.append(training_data[i][j])
                    pos_label.append(training_label[i][j])
                else:
                    neg_data.append(training_data[i][j])
                    neg_label.append(training_label[i][j])

            pos_sampler = torch.utils.data.RandomSampler(data_source=pos_data, num_samples=15)
            neg_sampler = torch.utils.data.RandomSampler(data_source=neg_data, num_samples=15)

            for p in pos_sampler:
                for q in neg_sampler:
                    output = output + model(pos_data[p], neg_data[q], i)
                    
            loss = loss + output / len(pos_sampler) / len(neg_sampler)

            pos_data.clear()
            pos_label.clear()
            neg_data.clear()
            neg_label.clear()

        loss = loss / T + args['lambda'] / 2 * (linalg.norm(model.M) ** 2 + linalg.norm(model.N) ** 2) - args['lambda'] * linalg.norm(matmul(model.M, model.N.t()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # testing
        with torch.no_grad():
            for i in range(T):
                res = AUC.h(torch.matmul(model.M, model.N.t()).t()[i],  # validation
                            torch.FloatTensor(testing_data[i]).t()).squeeze(0).tolist()
                for j in range(len(res)):
                    pred.append(res[j])
                    label.append(testing_label[i][j])

            test_auc = AUC.AUC(label, pred)
            # nni.report_intermediate_result(test_auc)
            print('Test AUC {}/{}: {:.3f}'.format(k, K, test_auc))
            final_auc = max(test_auc, final_auc)
            # print('Test loss: {:.3f}'.format(model(testing_data, testing_label).item()))
            # print('Current Best loss: {:.3f}'.format(best_loss))

            pred.clear()
            label.clear()

        end = time.process_time() 
        iter_time.append(end-start)

    _, s, _ = torch.svd(torch.matmul(model.M, model.N.t()))
    print(s)
        
    print('best auc: {:.3f}'.format(final_auc))
    print('each iteration time: {:.4f}'.format(sum(iter_time) / len(iter_time)))
    print('E.W.2: {:.4f}'.format(linalg.norm(AUC.h(model.M, model.N.t()) - tensor(W_best['W']).t()) ** 2 / T))  # E.W
    print('E.W.: {:.4f}'.format(linalg.norm(AUC.h(model.M, model.N.t()) - tensor(W_best['W']).t()) / T))  # E.W
    # nni.report_final_result(final_auc)
    return final_auc, sum(iter_time) / len(iter_time)


if __name__ == "__main__":
    params = {
        'lr_m': 5e-2,
        'lr_n': 5e-2,
        'lr_v': 1e-2,
        'lr_alpha': 1e-2,
        'lambda': 1e-2,
        'lam_v': 1e-2,

        'r': 5,
        'lr': 1e-2,
        'weight_decay': 1e-5,
    }
    avg_time = []
    auc = []
    iter = 10
    # params = nni.get_next_parameter()

    #-------------------------------STL_MSE---------------------------------------
    # STL_CE_valid(params)
    # for i in range(iter):
    #     auc_i, time_i = STL_CE_test(params)
    #     auc.append(auc_i * 100)
    #     avg_time.append(time_i)
    # print('Avg auc: {:.1f}'.format(np.sum(auc) / iter))
    # print('Standard deviation: {:.2f}'.format(np.std(auc)))
    # print('Each iteration time: {:.4f}'.format(sum(avg_time) / len(avg_time)))
    #-----------------------------------------------------------------------------

    #-------------------------------MTL_MSE---------------------------------------
    # MTL_CE_valid(params)
    # for i in range(iter):
    #     auc_i, time_i = MTL_CE_test(params)
    #     auc.append(auc_i * 100)
    #     avg_time.append(time_i)
    # print('Avg auc: {:.1f}'.format(np.sum(auc) / iter))
    # print('Standard deviation: {:.2f}'.format(np.std(auc)))
    # print('Each iteration time: {:.4f}'.format(sum(avg_time) / len(avg_time)))
    #-----------------------------------------------------------------------------

    #-------------------------------MTL_MSE_LR------------------------------------
    # MTL_CE_LR_valid(params)
    # for i in range(iter):
    #     auc_i, time_i = MTL_CE_LR_test(params)
    #     auc.append(auc_i * 100)
    #     avg_time.append(time_i)
    # print('Avg auc: {:.1f}'.format(np.sum(auc) / iter))
    # print('Standard deviation: {:.2f}'.format(np.std(auc)))
    # print('Each iteration time: {:.4f}'.format(sum(avg_time) / len(avg_time)))
    #-----------------------------------------------------------------------------

    #-------------------------------MTAUC_FNNFN-----------------------------------
    # reAUC_valid(params)
    for i in range(iter):
        auc_i, time_i = reAUC_test(params)
        auc.append(auc_i * 100)
        avg_time.append(time_i)
    print('Avg auc: {:.1f}'.format(np.sum(auc) / iter))
    print('Standard deviation: {:.2f}'.format(np.std(auc)))
    print('Each iteration time: {:.4f}'.format(sum(avg_time) / len(avg_time)))
    #-----------------------------------------------------------------------------

    #---------------------------------STL_AUC-------------------------------------
    # STL_AUC_valid(params)
    # for i in range(iter):
    #     auc_i, time_i = STL_AUC_test(params)
    #     auc.append(auc_i * 100)
    #     avg_time.append(time_i)
    # print('Avg auc: {:.1f}'.format(np.sum(auc) / iter))
    # print('Standard deviation: {:.2f}'.format(np.std(auc)))
    # print('Each iteration time: {:.4f}'.format(sum(avg_time) / len(avg_time)))
    #-----------------------------------------------------------------------------

    #---------------------------------MTL_AUC-------------------------------------
    # MTL_AUC_valid(params)
    # for i in range(iter):
    #     auc_i, time_i = MTL_AUC_test(params)
    #     auc.append(auc_i * 100)
    #     avg_time.append(time_i)
    # print('Avg auc: {:.1f}'.format(np.sum(auc) / iter))
    # print('Standard deviation: {:.2f}'.format(np.std(auc)))
    # print('Each iteration time: {:.4f}'.format(sum(avg_time) / len(avg_time)))
    #-----------------------------------------------------------------------------

    #-------------------------------MTL_AUC_LR------------------------------------
    # MTL_AUC_LR_valid(params)
    # for i in range(iter):
    #     auc_i, time_i = MTL_AUC_LR_test(params)
    #     auc.append(auc_i * 100)
    #     avg_time.append(time_i)
    # print('Avg auc: {:.1f}'.format(np.sum(auc) / iter))
    # print('Standard deviation: {:.2f}'.format(np.std(auc)))
    # print('Each iteration time: {:.4f}'.format(sum(avg_time) / len(avg_time)))
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # pred = []
    # label = []
    # W = scio.loadmat('CMTL_W_Sim10.mat')
    # # W = tensor(W['W11'], dtype=torch.float32)
    # # W = tensor(W['W21'], dtype=torch.float32)
    # W = tensor(W['W'], dtype=torch.float32)
    # # print(W.shape)
    # for i in range(100):
    #     res = AUC.h(W.t()[i], torch.FloatTensor(testing_data[i]).t()).squeeze(0).tolist()
    #     for j in range(len(res)):
    #         pred.append(res[j])
    #         label.append(testing_label[i][j][0])

    # test_auc = AUC.AUC(label, pred)
    # print('Test AUC {:.3f}'.format(test_auc))