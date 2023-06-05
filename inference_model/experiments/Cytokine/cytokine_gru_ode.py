import sys
sys.path.append('/gpfs/ysm/home/of56')
sys.path.append('/gpfs/ysm/home/of56/gru_ode_bayes')
sys.path.append('/gpfs/ysm/home/of56/gru_ode_bayes/torchdiffeq')
import argparse
import gru_ode_bayes.data_utils as data_utils
import gru_ode_bayes
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd

#Define argument parse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str,help="Model to use",default="NNFO_gru_ode_bayes")
parser.add_argument('--dataset', type=str, help="Dataset CSV file", default="/home/of56/MGH/processed_data/patrej_gt.csv")
parser.add_argument('--jitter', type=float, help="Time jitter to add (to split joint observations)", default=0)
parser.add_argument('--seed', type=int, help="Seed for data split generation", default=432)
parser.add_argument('--full_gru_ode', action="store_true", default=True)
parser.add_argument('--solver', type=str, choices=["euler", "midpoint","dopri5"], default="euler")
parser.add_argument('--no_impute',action="store_true",default = True)
parser.add_argument('--demo', action = "store_true", default = False)

#Define model name/load device
args = parser.parse_args()
model_name = args.model_name
params_dict = dict()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load metadata
metadata = np.load(f"{args.dataset[:-4]}_metadata.npy",allow_pickle=True).item()
print(metadata)
delta_t = metadata["delta_t"]
T       = metadata["T"]

#Define model parameters
params_dict["input_size"]  = 114
params_dict["hidden_size"] = 100
params_dict["p_hidden"]    = 100
params_dict["prep_hidden"] = 100
params_dict["logvar"]      = True
params_dict["mixing"]      = 0.0001
params_dict["delta_t"]     = delta_t
params_dict["dataset"]     = args.dataset
params_dict["jitter"]      = args.jitter
#params_dict["gru_bayes"]   = "masked_mlp"
params_dict["full_gru_ode"] = args.full_gru_ode
params_dict["solver"]      = args.solver
params_dict["impute"]      = not args.no_impute
params_dict["T"]           = T

## the neural negative feedback with observation jumps
model = gru_ode_bayes.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size =params_dict["hidden_size"],p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
logvar = params_dict["logvar"], mixing = params_dict["mixing"], full_gru_ode = params_dict["full_gru_ode"],
solver = params_dict["solver"], impute = params_dict["impute"])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
epoch_max = 100

#Split into Train/Validation
train_idx, val_idx = train_test_split(np.arange(metadata["N"]),test_size=0.3, random_state=args.seed)
print(metadata["N"])
print("Length of Training ID's",len(train_idx))
print("Length of Validation ID:",len(val_idx))



#val_options = {"T_val": 4, "max_val_samples": 1} NOTE: have option to specify additional validation options
data_train = data_utils.ODE_Dataset(csv_file=args.dataset, idx=train_idx, jitter_time=args.jitter)
data_val   = data_utils.ODE_Dataset(csv_file=args.dataset, idx=val_idx, jitter_time=args.jitter)

#print(data_val.df_before.sort_values("Time"))
#print(data_val.df)
data_val.df.to_csv('processed_data/debug_val.csv',index=True) #Check Validation set:



# Match shuffled validation indeces to original indeces 

#print(data_val.df.index.values)
#print(data_val.df_before.sort_values("Time")["ID"].values)
mapID_val = np.array([data_val.df.index.values,data_val.df_before.sort_values("Time")["ID"].values])
#print(mapID)
np.save("mapID_val.npy",mapID_val)

mapID_train = np.array([data_train.df.index.values,data_train.df_before.sort_values("Time")["ID"].values])
np.save("mapID_train.npy",mapID_train)


#Put data through data loader (shuffle/specify batch size)
dl     = DataLoader(dataset=data_train, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=128,num_workers=1)
dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.custom_collate_fn, shuffle=False, batch_size=len(val_idx),num_workers=1)

"""
print(dir(dl))
print(dir(dl.dataset))
print(dl.dataset.df)
print(dl.dataset.df.shape)
"""

#NOTE: batch size corresponds to number of unique observations, NOT # of time points


#For loop for training
for epoch in range(epoch_max):
    model.train()

    for i,b in tqdm.tqdm(enumerate(dl)): 
        
        optimizer.zero_grad()
        times=b["times"]
        time_ptr=b["time_ptr"]
        X=b["X"].to(device)
        M=b["M"].to(device)
        obs_idx=b["obs_idx"]
        cov=b["cov"].to(device)
        y=b["y"] 
        
        
        hT, loss, _, _  = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov)
        loss.backward()
        optimizer.step()
        
        #print(obs_idx)
    print("Epoch",epoch)
    print("Loss",loss)
    

with torch.no_grad():
    model.eval()
    for i, b in enumerate(dl):

        times    = b["times"]
        time_ptr = b["time_ptr"]
        X        = b["X"].to(device)
        M        = b["M"].to(device)
        obs_idx  = b["obs_idx"]
        cov      = b["cov"].to(device)
        y = b["y"]
        
        #print(X)

        #X_val     = b["X_val"].to(device)
        #M_val     = b["M_val"].to(device)
        #times_val = b["times_val"]
        times_idx = b["index_val"]        
 
        
        #print(obs_idx)


        #Pass through trained model
        hT, loss, _, t_vec, p_vec, h_vec, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True)              
        t_vec = np.around(t_vec,str(delta_t)[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.

        #p_val     = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
        #m, v      = torch.chunk(p_val,2,dim=1)
        
        np.save("time.npy",t_vec)
        np.save("obs_id.npy",obs_idx.cpu())
        np.save("mean_cov.npy",p_vec.cpu())
        
        #print("END: Observation ID:",obs_idx)
        print(len(obs_idx))
        print(torch.max(obs_idx))
        print(p_vec.size())
        #print("END: length validation obs_idx",len(obs_idx))
        #print("Times",times_idx)
        print("Done!")


    