import h5py
import random
import torch
import sys
import os
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
REL_PATH = os.path.join(CURR_PATH, '..') 
ABS_PATH = os.path.abspath(REL_PATH)
print(ABS_PATH)
sys.path.append(ABS_PATH)
from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.eginet import SEGINet
from deeprank_gnn.eNeuralNet import NeuralNet
import glob
import os 
from multiprocessing import Process,Pool
import multiprocessing

# example of creating hdf5 file
def create_graph():
    pdb_path = ABS_PATH + "/example/data/"
    nproc = 20
    outfile = ABS_PATH + "/example/test.hdf5"

    GraphHDF5(
        pdb_path=pdb_path,
        graph_type="residue",
        outfile=outfile,
        nproc=nproc,
        tmpdir="./tmpdir",
    )


# example of adding target value to hdf5 file
def add_target(hdf5):
    hdf5_file = h5py.File(hdf5, "r+")
    for mol in hdf5_file.keys():
        bin_class = [1 if mol[-1] == 'p' else 0] 
        hdf5_file.create_dataset(f"/{mol}/score/bin_class", data=bin_class)
    hdf5_file.close()
    print(f"{hdf5} target added!")



# exmaple of using pretrained model to predict
def predict(model,N,lmax_h, pretrained_model):
    database_test = ABS_PATH + "/example/test.hdf5"
    gnn = SEGINet
    target = "bin_class"
    edge_attr = ["dist"]
    threshold = 0.5
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = pretrained_model.split('/')[-1]
    batch = model_name.split('_')[-4][1:]
    max_epoch = model_name.split('_')[-3][1:]
    epoch = model_name.split('_')[-1].split('.')[0]
    node_feature = ["type", "polarity", "bsa", "charge",'pos']
    model = NeuralNet(
        database_test,
        gnn,
        N,
        lmax_h,
        device_name=device_name,
        edge_feature=edge_attr,
        node_feature=node_feature,
        target=target,
        lr = 0.001,
        percent = [0.8,0.2],
        pretrained_model=pretrained_model,
        threshold=threshold,
    )
    print(N)
    model.test(database_test=ABS_PATH + "/example/test.hdf5",
               threshold=0.5,
               hdf5=ABS_PATH + f"/example/test_{epoch}_0.001_{batch}_{max_epoch}_{N}_{lmax_h}.hdf5")
            
def main(): 
    create_graph()
    add_target(ABS_PATH + "/example/test.hdf5")
    model = ABS_PATH +"/pretrained_models/tclass_ybin_class_b128_e150_lr0.001_128.pth.tar"

    predict("segnn",1,1,model)


if __name__ == "__main__":
    main()

