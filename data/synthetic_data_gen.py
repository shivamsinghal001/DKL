import numpy as np
import sys
import pickle
import argparse
import torch
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import dump

def coverage(y, yhat, sigma):
    count_vec = [(x1 <= x2 + x3) & (x1 >= x2 - x3) for x1, x2, x3 in zip(y, yhat, sigma)]
    return sum(count_vec)/len(y)

def rand_data_gen(x):
    if(x < 0.10) :
        y = 0.3
    elif(x >= 0.10 and x < 0.20) :
        y = -0.4
    elif (x >= 0.20 and x < 0.35) :
        y = 0.7 * np.sin((10 * x) ** 2 + 3)
    elif (x >= 0.35 and x < 0.65):
        y = -0.7
    elif (x >= 0.65 and x < 0.85):
        y = 0.2 * np.sin((5 * x) ** 2 + 10)
    else :
        y = -0.4
    return y

def data_gen(x):
    y = (np.exp(-0.75 * x) * np.sin((np.pi * 3/ 4 * x ** 2 + np.pi) ** 2))
    if x >= 0.2 and x < 0.4:
      y = y ** 2
    elif x >= 0.4 and x < 0.8:
      y = -3 ** y
    else:
      y = -2 ** y
    return y

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Synthetic data generation')
  parser.add_argument('--data', type=int, default=4000,
                      help='Number of data samples')
  parser.add_argument('--test_prop', type=float, default=0.20,
                      help='Proportion of data to set aside for testing')
  parser.add_argument('--save', type=str, default="synthetic-data",
                      help='directory to save the data')
  args = parser.parse_args()
  N = args.data
  test_prop = args.test_prop
  
  n = round(test_prop * N)

  vf = np.vectorize(data_gen)
  X_train=np.random.random(size=(n, 1))
  X_train = np.sort(X_train.reshape(-1, )).reshape(-1, 1)

  noise_vec = np.array([np.random.normal(0, 0.2, size=1) for _ in X_train])
  y_train = vf(X_train) + noise_vec

  X_test=np.linspace(-.3, 1.3, N).reshape(-1, 1)
  y_test = vf(X_test)

  data_dict = {'X_train': X_train,
             'y_train': y_train,
             'X_test': X_test,
             'y_test': y_test}
  pickle.dump(data_dict, open(args.save+'.p', "wb" ))

  feature_scalar = StandardScaler().fit(X_train)
  label_scalar = StandardScaler().fit(y_train)
  dump(feature_scalar, 'feature_scalar.bin', compress=True)
  dump(label_scalar, 'label_scalar.bin', compress=True)
  
  scaled_X_train = feature_scalar.transform(X_train)
  scaled_y_train = label_scalar.transform(y_train)

  scaled_X_test = feature_scalar.transform(X_test)

  scaled_X_train = torch.from_numpy(scaled_X_train).float()
  scaled_y_train = torch.from_numpy(scaled_y_train).float()
  scaled_X_test = torch.from_numpy(scaled_X_test).float()
  scaled_data_dict = {'scaled_X_train': scaled_X_train,
                      'scaled_y_train': scaled_y_train,
                      'scaled_X_test': scaled_X_test}
  pickle.dump(scaled_data_dict, open(args.save+'-scaled.p', "wb" ))
                  


