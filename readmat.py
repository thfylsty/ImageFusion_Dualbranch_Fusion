import scipy.io as scio

path = 'result/H/add.mat'
data = scio.loadmat(path)
data = data['b'][0]
# data = data[:, 0, 0]
for d in data:

    print( "%.4f "%d,end="")