import numpy as np
import matplotlib.pyplot as pt
# from matplotlib import rc

# rc('text', usetex=True)

m_range = ['1', '0.75', '0.5', '0.25', '0.1', '0.01', '0.001', '0']

def plot(dataset):
    if dataset=="mnist":
        epo = 200
    else:
        epo = 100
    x = np.arange(epo)
    y_train = []
    y_test = []
    for momentum in m_range:
        y_train.append(np.load("result/"+dataset+"-testerr-"+momentum+".npy"))
        y_test.append(np.load("result/"+dataset+"-testacc-"+momentum+".npy"))

    tr_min = min([min(yi) for yi in y_train])
    te_max = max([max(yi) for yi in y_test])

    for ind, yi in enumerate(y_train):
        if m_range[ind]=='1':
            pt.plot(x, yi, 'r', label="alpha="+m_range[ind], linewidth=1.3)
        else:
            pt.plot(x, yi, label="alpha="+m_range[ind], linewidth=0.5)
    # pt.title(dataset)
    pt.xlabel("# Epochs")
    pt.ylabel("Validation errors")
    pt.xlim(0, epo)
    if dataset=="kdd":
        pt.ylim(0.52, 0.7)
    else:
        pt.ylim(tr_min*0.8, 0.12)
    pt.legend()
    pt.savefig("figs/"+dataset+"_test_err.pdf",
        bbox_inches='tight')
    pt.close()

    for ind, yi in enumerate(y_test):
        if m_range[ind]=='1':
            pt.plot(x, yi, 'r', label="alpha="+m_range[ind], linewidth=1.3)
        else:
            pt.plot(x, yi, label="alpha="+m_range[ind], linewidth=0.5)
    # pt.title(dataset)
    pt.xlabel("# Epochs")
    pt.ylabel("Test accuracy")
    pt.xlim(0, epo)
    if dataset=="kdd":
        pt.ylim(0.88, 0.92)
    else:
        pt.ylim(te_max*0.99, te_max*1.005)
    pt.legend()
    pt.savefig("figs/"+dataset+"_test_accuracy.pdf",
        bbox_inches='tight')
    pt.close()


plot("mnist")
plot("kdd")
