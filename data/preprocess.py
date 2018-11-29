import numpy as np
import torch

def mnist():
    for fname in ["mnist.train", "mnist.test"]:
        with open(fname) as f:
            x = []
            y = []
            for line in f.readlines()[1:]:
                a = line.strip().split(',')
                y.append(int(a[0]))
                x.append(list(map(lambda x: float(x), a[1:])))
            torch.save(torch.FloatTensor(x), fname+".x.pt")
            torch.save(torch.LongTensor(y), fname+".y.pt")

dicts = [{}, {}, {}, {}]
stat = [0,0,0,0]

def get_dict(string, column, update):
    dic = dicts[column]
    count = len(dic)
    if string not in dic:
        if update:
            dic[string] = count
        else:
            if "UNKNOWN" not in dic:
                dic["UNKNOWN"] = count
            stat[column] += 1
            return dic["UNKNOWN"]

    return dic[string]


def kdd():
    for fname in ["kdd.train", "kdd.test"]:
        with open(fname) as f:
            x = []
            y = []
            for line in f.readlines():
                try:
                    a = line.strip().split(',')
                    tempx = []
                    for ind, value in enumerate(a[:-1]):
                        if ind in [1,2,3]:
                            tempx.append(get_dict(value, ind-1, "train" in fname))
                        else:
                            tempx.append(float(value))
                    x.append(tempx)
                    y.append(get_dict(a[-1][:-1], 3, "train" in fname))
                except ValueError:
                    print(line)
            torch.save(torch.FloatTensor(x), fname+".x.pt")
            torch.save(torch.LongTensor(y), fname+".y.pt")

mnist()
kdd()
print(dicts)
print(stat)
