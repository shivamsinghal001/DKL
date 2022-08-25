import pickle
import argparse
import joblib
from joblib import load
from sklearn import metrics
import numpy as np
import sys
import time
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel, ExpSineSquared
from sklearn import linear_model
from sklearn.utils import resample
import xgboost as xgb
import BagsOfNN
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle

viridis = cm.get_cmap('viridis', 256)

def coverage(y, yhat, sigma):
    count_vec = [(x1 <= x2 + x3) & (x1 >= x2 - x3) for x1, x2, x3 in zip(y, yhat, sigma)]
    return sum(count_vec)/len(y)

def plot_uncertainty(mean, std, X_train, X_test, y_train, y_test, title, name):
	plt.figure() 
	plt.plot(X_test.reshape(-1,), y_test) 
	plt.plot(X_test.reshape(-1,), mean)
	plt.plot(X_train.reshape(-1,), y_train, 'g.', markersize=6) 
	plt.fill_between(X_test.reshape(-1,), (mean-std).reshape(-1,),\
    	             (mean + std).reshape(-1,), color='lightblue', alpha=1)
	plt.legend(['ground truth','mean prediction','training data','uncertenty bound'])
	plt.title(title)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig(name)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--kernel_rank', type=int, default=2,
                      help='Rank of the kernel')
  parser.add_argument('--load_model', type=str, default='models/',
                      help='directory to load the model')
  parser.add_argument('--load_data', type=str, default='data/',
                      help='directory to load the data')
  parser.add_argument('--save_fig', type=str, default='output/',
                      help='directory to save the figures')
  args = parser.parse_args()

  data_dict = pickle.load(open(args.load_data + "synthetic-data.p", "rb" ) )
  X_train = data_dict['X_train']
  y_train = data_dict['y_train']
  X_test = data_dict['X_test']
  y_test = data_dict['y_test']

  data_dict_scaled = pickle.load(open(args.load_data + "synthetic-data-scaled.p", "rb" ))
  scaled_X_train = data_dict_scaled['scaled_X_train']
  scaled_y_train = data_dict_scaled['scaled_y_train']
  scaled_X_test = data_dict_scaled['scaled_X_test']

  feature_scalar = load(args.load_data + 'feature_scalar.bin')
  label_scalar = load(args.load_data + 'label_scalar.bin')

  R2 = {}
  explained_var = {}
  MSE = {}
  MAE = {}
  inference_times = {}
  coverage_metric = {}
  sizes = {}

  #DKL Evaluation
  dklnet = torch.load(args.load_model+'dklnet.h')
  dklnet.eval()
  phi,sigma = dklnet(scaled_X_test)
  R = phi.size()[1]

  plt.figure()
  for ii in range(R):
      plt.plot(X_test, phi.detach().numpy()[:,ii])
  plt.title('DEKL basis')
  plt.xlabel('x')
  plt.ylabel('Phi')
  plt.savefig(args.save_fig + 'DEKL_syn1.jpg')

  
  start_time = time.time()
  scaled_mean_lindkl,scaled_var_lindkl = dklnet.predict(scaled_X_train,scaled_y_train,scaled_X_test)
  mean_lindkl = label_scalar.inverse_transform(scaled_mean_lindkl.detach().numpy())
  std_lindkl = label_scalar.inverse_transform(np.sqrt(torch.diag(scaled_var_lindkl).detach()
                                           .numpy()).reshape(-1,1))
  std_lindkl = np.abs(std_lindkl)
  end_time = time.time()
  infer_time = end_time - start_time
  inference_times['DKL'] = infer_time

  mtx = (phi@phi.t()).detach().numpy()

  plot_uncertainty(mean_lindkl, std_lindkl, X_train, X_test, y_train, y_test, 'DKL', args.save_fig+'DKL_syn1_fit.jpg')

  min_val = min(X_test)[0]
  max_val = max(X_test)[0]
  plt.imshow(mtx,interpolation='none', 
            cmap=viridis,extent=[min_val, max_val,min_val, max_val]
          )

  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('DEKL kernel matrix')
  plt.savefig(args.save_fig + 'DEKL_syn1_heat_map1.jpg')

  R2['DKL'] = metrics.r2_score(y_test, mean_lindkl)
  explained_var['DKL'] = metrics.explained_variance_score(y_test, mean_lindkl)
  MSE['DKL'] = metrics.mean_squared_error(y_test, mean_lindkl)
  MAE['DKL'] = metrics.mean_absolute_error(y_test, mean_lindkl)
  coverage_metric['DKL'] = coverage(y_test.flatten(), 
                     mean_lindkl.flatten(), 
                     std_lindkl.flatten())
  sizes['DKL'] = sys.getsizeof(dklnet)
  print('DKL evaluation done!')

  #Lasso Evaluation
  lasso = linear_model.LassoCV().fit(scaled_X_train.detach().numpy(), scaled_y_train.detach().numpy().reshape(-1,))

  start_time = time.time()
  scaled_mean_LASSO = lasso.predict(scaled_X_test)
  mean_LASSO = label_scalar.inverse_transform(scaled_mean_LASSO) 
  end_time = time.time()
  infer_time = end_time - start_time
  inference_times['Lasso'] = infer_time

  R2['Lasso'] = metrics.r2_score(y_test, mean_LASSO)
  explained_var['Lasso'] = metrics.explained_variance_score(y_test, mean_LASSO)
  MSE['Lasso'] = metrics.mean_squared_error(y_test, mean_LASSO)
  MAE['Lasso'] = metrics.mean_absolute_error(y_test, mean_LASSO)
  coverage_metric['Lasso'] = coverage(y_test.flatten(), mean_LASSO.flatten(), 
                     np.std(y_train) * np.array([1.] * len(y_test)))
  sizes['Lasso'] = sys.getsizeof(lasso)
  print('Lasso evaluation done!')

  #Bags of NN Evaluation
  num_bag_nn = 10
  bag_of_nn = []
  n_samples = round(scaled_X_train.size()[0]/num_bag_nn)
  num_of_iterations = 10000

  for _ in range(num_bag_nn): 
      scaled_X_train_bt, scaled_y_train_bt = resample(scaled_X_train.numpy(), scaled_y_train.numpy(), n_samples=n_samples)
      scaled_X_train_bt = torch.from_numpy(scaled_X_train_bt).float()
      scaled_y_train_bt = torch.from_numpy(scaled_y_train_bt).float()
    
      nnet = BagsOfNN.BagOfNeuralNet(scaled_X_train_bt.shape[1], args.kernel_rank) 
      nnet.fit(scaled_X_train_bt, scaled_y_train_bt, num_of_iterations)
      bag_of_nn.append(nnet)

  y_pred_list = []
  start_time = time.time()
  for ii in range(num_bag_nn):
      scaled_y_pred = bag_of_nn[ii].predict(scaled_X_test)
      y_pred = label_scalar.inverse_transform(scaled_y_pred.detach().numpy())
      y_pred_list.append(y_pred)
  end_time = time.time()
  infer_time = end_time - start_time
  inference_times['bag'] = infer_time 

  mean_BNN = np.mean(y_pred_list, axis=0)
  std_BNN = np.std(y_pred_list, axis = 0)
  std_BNN[std_BNN <= 0.1 * np.std(y_train)] = 0.1 * np.std(y_train)

  plot_uncertainty(mean_BNN, std_BNN, X_train, X_test, y_train, y_test, 'Bags of NN', args.save_fig+'BagOfNN_syn1_fit.jpg')

  R2['bag'] = metrics.r2_score(y_test, mean_BNN)
  explained_var['bag'] = metrics.explained_variance_score(y_test, mean_BNN)
  MSE['bag'] = metrics.mean_squared_error(y_test, mean_BNN)
  MAE['bag'] = metrics.mean_absolute_error(y_test, mean_BNN)
  coverage_metric['bag'] = coverage(y_test.flatten(), mean_BNN.flatten(), std_BNN.flatten())
  sizes['bag'] = sys.getsizeof(bag_of_nn)
  print('Bags of NN evaluation done!')

  #Mixture of Gaussians
  k1 = 66.0**2 * RBF(length_scale=67.0)  
  k2 = 2.4**2 * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)  
  k3 = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78)
  k4 = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)  
  kernel_gpml = k1+k2+k3+k4

  gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0.000001, normalize_y=True)
  gp.fit(scaled_X_train,scaled_y_train)

  start_time = time.time()
  scaled_mean_pred_GP,scaled_std_GP=gp.predict(scaled_X_test, return_std = True)
  mean_GP = label_scalar.inverse_transform(scaled_mean_pred_GP)
  std_GP = scaled_std_GP*np.sqrt(label_scalar.var_) 
  end_time = time.time()
  infer_time = end_time - start_time
  inference_times['GP'] = infer_time

  plot_uncertainty(mean_GP, std_GP, X_train, X_test, y_train, y_test, 'GP', args.save_fig+'GP_syn1_fit.jpg')

  R2['GP'] = metrics.r2_score(y_test, mean_GP)
  explained_var['GP'] = metrics.explained_variance_score(y_test, mean_GP)
  MSE['GP'] = metrics.mean_squared_error(y_test, mean_GP)
  MAE['GP'] = metrics.mean_absolute_error(y_test, mean_GP)
  coverage_metric['GP'] = coverage(y_test.flatten(), mean_GP.flatten(), std_GP.flatten())
  sizes['GP'] = sys.getsizeof(GaussianProcessRegressor)
  print('GP evaluation done!')

  #XGBOOST
  num_bag_xgb = 10
  bag_of_xgb = []
  n_samples = round(scaled_X_train.size()[0]/num_bag_xgb)
  num_of_iterations = 10000

  for _ in range(num_bag_xgb): 
      scaled_X_train_bt, scaled_y_train_bt = resample(scaled_X_train.numpy(), 
                                                    scaled_y_train.numpy(), 
                                                    n_samples=n_samples)
    
      xgb_reg = xgb.XGBRegressor() 
      xgb_reg.fit(scaled_X_train_bt, scaled_y_train_bt)
      bag_of_xgb.append(xgb_reg)

  y_pred_list = []
  start_time = time.time()
  for ii in range(num_bag_xgb):
      scaled_y_pred = bag_of_xgb[ii].predict(scaled_X_test.numpy())
      y_pred = label_scalar.inverse_transform(scaled_y_pred)
      y_pred_list.append(y_pred)
  end_time = time.time()
  infer_time = end_time - start_time
    
  mean_XGB = np.mean(y_pred_list, axis=0)
  std_XGB = np.std(y_pred_list, axis = 0)

  plot_uncertainty(mean_XGB, std_XGB, X_train, X_test, y_train, y_test, 'XGB', args.save_fig+'XGB_syn1_fit.jpg')

  R2['XGB'] = metrics.r2_score(y_test, mean_XGB)
  explained_var['XGB'] = metrics.explained_variance_score(y_test, mean_XGB)
  MSE['XGB'] = metrics.mean_squared_error(y_test, mean_XGB)
  MAE['XGB'] = metrics.mean_absolute_error(y_test, mean_XGB)
  coverage_metric['XGB'] = coverage(y_test.flatten(), mean_XGB.flatten(), std_XGB.flatten())
  sizes['XGB'] = sys.getsizeof(bag_of_xgb)
  print('XGBoost evaluation done!')

  data = {'R2':R2, 'explained_var':explained_var, 'MSE':MSE, 'MAE':MAE, 'coverage':coverage_metric, 'sizes':sizes, 'inference_times':inference_times}
  pickle.dump(data, open(args.save_fig+'stats.p', "wb" ))
