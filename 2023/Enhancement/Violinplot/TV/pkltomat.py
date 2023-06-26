# read file pkl and save as mat

import pickle
import scipy.io as sio
import numpy as np
listkey = []
value = []
# read pkl file
pkl_file = open(
    '2021/src/Classification/kvasir-capsule/TCFA/dict/blur_dict.pkl', 'rb')
data = pickle.load(pkl_file)
for key in data:
    # save key and value to mat file
    listkey.append(key)
    value.append(data[key].item())

# append key and value to a matrix
dictfinal = np.array([listkey, value])

print(dictfinal)

# save mat file
save_path = '2021/src/Classification/kvasir-capsule/TCFA/dict/blur_dict.mat'
sio.savemat(save_path, {'dictfinal': dictfinal})
