import numpy as np
import torch


class FNNCore(torch.nn.Module):

    def __init__(self, D_in, H, D_out, momentum):
        """
        D_in: input dimension
        H: hidden layer dimension
        D_out: output dimension
        momentum: the proportion for current batch's stats
        """
        super(FNNCore, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

        # normalization layers
        if momentum==1:
            # BN
            self.norm1 = torch.nn.BatchNorm1d(H, track_running_stats=False)
            self.norm2 = torch.nn.BatchNorm1d(H, track_running_stats=False)
            self.norm3 = torch.nn.BatchNorm1d(D_out, track_running_stats=False)
        else:
            # DBN
            self.norm1 = torch.nn.BatchNorm1d(H, momentum=momentum)
            self.norm2 = torch.nn.BatchNorm1d(H, momentum=momentum)
            self.norm3 = torch.nn.BatchNorm1d(D_out, momentum=momentum)


    def forward(self, x):
        step1 = self.norm1(torch.nn.functional.relu(self.linear1(x)))
        step2 = self.norm2(torch.nn.functional.relu(self.linear2(step1)))
        step3 = self.norm3(torch.nn.functional.relu(self.linear3(step2)))
        return step3


    # def update_norm(self, momentum):
    #     self.norm1 = torch.nn.BatchNorm1d(self.H, momentum=momentum)
    #     self.norm2 = torch.nn.BatchNorm1d(self.H, momentum=momentum)
    #     self.norm3 = torch.nn.BatchNorm1d(self.D_out, momentum=momentum)



class FNN(object):

    def __init__(self, D_in, H, D_out, momentum, gpu=True):
        self.device = torch.device("cuda" if gpu else "cpu")
        self.model = FNNCore(D_in, H, D_out, momentum).to(self.device)
        self.loss_func = torch.nn.CrossEntropyLoss() # softmax is included here
        self.optimizer = torch.optim.Adagrad(self.model.parameters(),
            lr=1e-2, weight_decay=1e-3) # w_d is the L2 regularization coefficient


    def train(self, batch_data):
        x, y = batch_data[0].to(self.device), batch_data[1].to(self.device)
        y_pred = self.model.forward(x)
        self.loss = self.loss_func(y_pred, y) # averaging over batch is included in loss_func
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss


    def get_loss(self, batch_data):
        x, y = batch_data[0].to(self.device), batch_data[1].to(self.device)
        y_pred = self.model.forward(x)
        loss = self.loss_func(y_pred, y) # averaging over batch is included in loss_func
        return loss


    def accuracy(self, batch_data, binary=False):
        x, y = batch_data[0].to(self.device), batch_data[1].to(self.device)
        y_pred = torch.argmax(self.model.forward(x), dim=1)
        if binary:
            y = torch.eq(y, 0)
            y_pred = torch.eq(y_pred, 0)
        return float(torch.sum(torch.eq(y, y_pred)))/len(y)

    # def update_norm(self, momentum):
    #     self.model.update_norm(momentum)
