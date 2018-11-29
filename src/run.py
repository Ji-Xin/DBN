import numpy as np
import argparse
from datetime import datetime
import torch

from model import FNN
from dataset import Dataset


#################################### parse arguments
parser = argparse.ArgumentParser()

choice = ["mnist", "kdd"]
parser.add_argument("--dataset", dest="dataset", required=True, choices=choice,
    help="which dataset to use")
parser.add_argument("--data_dir", dest="data_dir", required=True, action="store",
    help="path of dataset")
parser.add_argument("--result_dir", dest="result_dir", required=True, action="store",
    help="path of results")
parser.add_argument("--momentum", dest="momentum", required=False, action="store", default=1,
    help="the value of diminishing momentum")
args = parser.parse_args()
dd = args.data_dir
ds = args.dataset #dataset name



###################################### build model
if ds=="mnist":
    D_in, H, D_out = 784, 300, 10
    num_epochs = 200
    batch_size = 100
else:
    D_in, H, D_out = 41, 40, 24 # tentative
    num_epochs = 100
    batch_size = 1000

model = FNN(D_in, H, D_out, momentum=float(args.momentum), gpu=True)




#################################### load data
train_set = Dataset(dd+"/"+ds+".train.x.pt", dd+"/"+ds+".train.y.pt")
train_size = len(train_set)
train_loader = train_set.getLoader(batch_size, shuffle=True)
test_set = Dataset(dd+"/"+ds+".test.x.pt", dd+"/"+ds+".test.y.pt")
test_size = len(test_set)
test_loader = test_set.getLoader(test_size, shuffle=True)


print("Finishing loading data")


################################### train model
test_err = []
test_acc = []

for epo in range(num_epochs):

    for test_batch_data in test_loader:
        acc = model.accuracy(test_batch_data)
        err = model.get_loss(test_batch_data)
        test_err.append(float(err))
        test_acc.append(float(acc))

    for batch_i, batch_data in enumerate(train_loader):
        # batch_i: from 0 to dataset_size/batch_size
        # batch_data: [[xdata * batch_size], [ydata * batch_size]]
        batch_x, batch_y = batch_data
        model.train(batch_data)
        
    print(datetime.now().strftime("%m-%d %H:%M:%S"), end='\t')
    print("Epoch {}\tTest_acc: {}".format(epo, acc))

save_flag = str(args.momentum)

np.save(args.result_dir+"/"+ds+"-testerr-"+save_flag+".npy", test_err)
np.save(args.result_dir+"/"+ds+"-testacc-"+save_flag+".npy", test_acc)
