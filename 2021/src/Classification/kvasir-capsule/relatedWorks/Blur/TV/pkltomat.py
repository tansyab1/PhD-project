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
# open file csv to save data
with open('2021/src/Classification/kvasir-capsule/TCFA/dict/blur_dict.csv', 'w') as f:
    # write header
    f.write("key,value\n")
    for key in data:
        f.write("%s,%s\n" % (key, data[key].item()))


# for key in data:
#     # save key and value to mat file
#     listkey.append(key)
#     value.append(data[key].item())

# # save mat file
# save_path = '2021/src/Classification/kvasir-capsule/TCFA/dict/blur_dict.mat'
# sio.savemat(save_path, {'key': listkey, 'value': value})
