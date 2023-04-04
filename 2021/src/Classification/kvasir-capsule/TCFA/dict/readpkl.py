import pickle


with open('/home/nguyentansy/DATA/PhD-work/PhD-project/2021/src/Classification/kvasir-capsule/classification_experiments/dict/noise_dict.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)