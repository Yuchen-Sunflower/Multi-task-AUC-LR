import torch
from torch import linalg, tensor, matmul, nn
from torch.nn import Parameter
from sklearn import metrics


def h(w: tensor, x: tensor):
    '''
    multi-tasks linear function
    :param W^(i)^T: parameter tensor 1 * d
    :param X: example tensor d * N_m
    :return: tensor result 1 * N_m

    d: feature number
    T: task number
    N_m: instance number in each tasks
    '''
    return matmul(w, x)


def adjust_learning_rate(optimizer, lr0, epoch):
    # lr = lr0 * (1 / 3 ** (epoch - 1))
    if epoch < 50:
        lr = lr0 / epoch
    # elif epoch >= 50 and epoch < 500:
    #     lr = lr0 / 10
    else:
        lr = lr0 / 50
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.param_groups[-1]['lr'] = -lr


def proximal_L2norm(X, cons):  # cons = coefficient of reg, X = grad_W
    Y = X / (1 + 2 * cons) 
    return Y


def update_param(an, grad_a, bn, grad_b, alphan, grad_alpha, lam, ita_v, ita_alpha):  # proximal gradient descent ascent
    a = proximal_L2norm(an - grad_a * ita_v, lam)
    b = proximal_L2norm(bn - grad_b * ita_v, lam)
    alpha = proximal_L2norm(alphan - grad_alpha * ita_alpha, lam)
    return Parameter(a), Parameter(b), Parameter(alpha)


def AUC(y, pred):  # AUC metric
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    return metrics.auc(fpr, tpr)


class STL_CE(nn.Module):  # STL-MSE-L2
    def __init__(self, w: tensor):
        super(STL_CE, self).__init__()
        self.w = Parameter(w)  # w = (d, 1) 

    def forward(self, x: list):  # x = (N_m, d)
        pred = h(self.w.t(), torch.FloatTensor(x).t()).t()  # pred = (1, N_m)

        return pred


class MTL_CE(nn.Module):  # MTL-MSE-L2
    def __init__(self, W: tensor) -> None:
        super(MTL_CE, self).__init__()
        self.W = Parameter(W)  # W = (d, T) 

    def forward(self, X: list):
        pred = []
        for i in range(self.W.shape[1]):
            X_i = torch.FloatTensor(X[i])
            pred_i = h(self.W.t()[i], X_i.t()).t()  # W^T_i * X_i = (1, d) * (d, N_m) = (1, N_m)
            pred.append(pred_i)  # T * (1, N_m)

        return pred


class MTL_CE_LR(nn.Module):  # MTL-MSE-FNNFN
    def __init__(self, M: tensor, N: tensor) -> None:
        super(MTL_CE_LR, self).__init__()
        self.M = Parameter(M)  # M = (d, r) 
        self.N = Parameter(N)  # N = (T, r)

    def forward(self, X: list):
        pred = []
        for i in range(self.N.shape[0]):
            X_i = torch.FloatTensor(X[i])
            pred_i = h(matmul(self.M, self.N.t()).t()[i], X_i.t()).t()  # W_i^T * X_i = (1, d) * (d, N_m) = (1, N_m)
            pred.append(pred_i)  # T * (1, N_m)

        return pred
        

class AUROC(nn.Module):  # STL-AUC-L2
    def __init__(self, w: tensor) -> None:
        super(AUROC, self).__init__()
        self.w = Parameter(w)  # w = (d, 1)

    def forward(self, data_pos: list, data_neg: list):
        pos = torch.FloatTensor(data_pos)
        neg = torch.FloatTensor(data_neg)

        objective = (1 - h(self.w.t(), (pos - neg).t())) ** 2

        return objective


class MTL_AUC(nn.Module):  # MTL-AUC-L2
    def __init__(self, W: tensor) -> None:
        super(MTL_AUC, self).__init__()
        self.W = Parameter(W)  # W = (d, T)

    def forward(self, pos: list, neg: list, i: int):
        pred = (1 - h(self.W.t()[i], torch.FloatTensor(pos).t()) + h(self.W.t()[i], torch.FloatTensor(neg).t())) ** 2

        return pred
        

class MTL_AUC_LR(nn.Module):  # MTL-AUC-FNNFN
    def __init__(self, M: tensor, N: tensor) -> None:
        super(MTL_AUC_LR, self).__init__()
        self.M = Parameter(M)  # M = (d, r)
        self.N = Parameter(N)  # N = (T, r)

    def forward(self, pos: list, neg: list, i: int):
        pred = (1 - h(matmul(self.M, self.N.t()).t()[i], torch.FloatTensor(pos).t()) + h(matmul(self.M, self.N.t()).t()[i], torch.FloatTensor(neg).t())) ** 2

        return pred


class reAUC(nn.Module):
    def __init__(self, M: tensor, N: tensor, a: tensor, b: tensor, alpha: tensor, lam: float):
        super(reAUC, self).__init__()
        self.M = Parameter(M)
        self.N = Parameter(N)
        self.a = Parameter(a)
        self.b = Parameter(b)
        self.alpha = Parameter(alpha)

        self.lam = lam

    def forward(self, X: list, Y: list):
        '''
            AUC Expression
            :param W: [d][T]
            :param M: [d][k]
            :param N: [T][k]
            :param a, b: [T][1] 
            :param alpha: [T][1]
            :param X: T tasks, i-th task is /// T * N_i * d////
            :param Y: T tasks' label, i-th task's label is /// T * N_i * 1 /// {0, 1}
            :return: expression F
        '''
        objective = 0
        T_pos, T_neg, p = 0, 0, 0

        # main reAUC
        for i in range(self.N.shape[0]):
            X_i = torch.FloatTensor(X[i]).t()
            Y_i = torch.FloatTensor(Y[i])
            # print(X_i.shape)

            y = Y[i]
            pos_i = sum(y).item()
            neg_i = sum(1 - y).item()

            T_pos = T_pos + pos_i
            T_neg = T_neg + neg_i
            p = T_pos / (T_pos + T_neg)

            if pos_i == 0:
                pos_i += 1
            if neg_i == 0:
                neg_i += 1

            objective = objective + (1 - p) / pos_i * h((h(h(self.M, self.N.t()).t()[i], X_i) - self.a[i][0]) ** 2, Y_i)
            objective = objective + p / neg_i * h((h(h(self.M, self.N.t()).t()[i], X_i) - self.b[i][0]) ** 2, (1 - Y_i))
            objective = objective + 2 * (1 + self.alpha[i][0]) * p / neg_i * h(h(h(self.M, self.N.t()).t()[i], X_i), (1 - Y_i))
            objective = objective - 2 * (1 + self.alpha[i][0]) * (1 - p) / pos_i * h(h(h(self.M, self.N.t()).t()[i], X_i), Y_i)

        objective = objective / self.N.shape[0]
        objective = objective - p * (1 - p) * h(self.alpha.t(), self.alpha)

        # regularizer NNFN
        objective = objective + self.lam / 2 * (linalg.norm(self.M)**2 + linalg.norm(self.N)**2)
        objective = objective - self.lam * linalg.norm(matmul(self.M, self.N.t()))

        return objective
    
def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss