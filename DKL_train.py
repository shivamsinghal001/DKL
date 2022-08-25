from data import synthetic_data_gen
import DKL
import argparse
import pickle
import torch
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--kernel_rank', type=int, default=2,
                      help='Rank of the kernel')
  parser.add_argument('--lamb_reg', type=float, default=3.0,
                      help='Weight for the regularity loss')
  parser.add_argument('--lamb_comp', type=float, default=1.0,
                      help='Weight for the complexity loss')
  parser.add_argument('--lamb_l2', type=float, default=3.0,
                      help='Weight for the l2 regularization loss')
  parser.add_argument('--num_iters', type=int, default=10000,
                      help='Number of iterations')
  parser.add_argument('--save', type=str, default='models/dklnet.h',
                      help='directory to save the model')
  parser.add_argument('--load_data', type=str, default='data/',
                      help='directory to load the data')
  args = parser.parse_args()

  data_dict = pickle.load(open(args.load_data + "synthetic-data.p", "rb" ))
  X_train = data_dict['X_train']
  y_train = data_dict['y_train']
  X_test = data_dict['X_test']
  y_test = data_dict['y_test']

  data_dict_scaled = pickle.load(open(args.load_data + "synthetic-data-scaled.p", "rb" ))
  scaled_X_train = data_dict_scaled['scaled_X_train']
  scaled_y_train = data_dict_scaled['scaled_y_train']
                                                                                                                                                                                                                                                                                                                                                                                                                                          
  sigma_init = y_train.max() - y_train.min()
  input_size = scaled_X_train.size()[1]

  dklnet = DKL.LinearDKL(args.kernel_rank, sigma_init, input_size)
  dklnet.fit(scaled_X_train, scaled_y_train, 
           args.num_iters,
           args.lamb_reg, args.lamb_comp,
           args.lamb_l2)
  
  dklnet.train()
  phi,sigma = dklnet(scaled_X_train)

  mtx = phi.t()@phi

  R = phi.size()[1]

  for ii in range(R):
      for jj in range(R):
          if ii == jj:
              continue
          mtx[ii,jj] /= torch.sqrt(mtx[ii,ii]*mtx[jj,jj])
  for ii in range(R):
      mtx[ii,ii] = 1 

  torch.save(dklnet, args.save)
    
