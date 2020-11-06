#in this file, we want to be able to plot the performance of individual experiment runs, track performance across
#runs, 
import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from celluloid import Camera


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="all")

json_dir_name = './results/'

def load_jsons(path):
    sparse = []
    dense = []
    dir = json_dir_name+path+'/'
    json_pattern = os.path.join(dir, '*.json')
    file_list = glob.glob(json_pattern)
    for file in file_list:
        if "False" in file:
            sparse.append(json.load(open(file,"rb")))
        else:
            dense.append(json.load(open(file,"rb")))
    return sparse,dense

#just compare performance of all sparse and all dense on this dataset
def compare_sparse_dense(sparse,dense):
    epochs = sparse[0]['epoch']
    sl = []
    for s in sparse:
        sl.append(s['test_acc'])
    smean = np.array(sl).transpose().mean(1)
    sstd = np.array(sl).transpose().std(1)

    dl = []
    for d in dense:
        dl.append(d['test_acc'])
    dmean = np.array(dl).transpose().mean(1)
    dstd = np.array(dl).transpose().std(1)
    plt.plot(epochs,smean)
    plt.fill_between(epochs, smean-sstd, smean+sstd, alpha = 0.5)
    plt.plot(epochs,dmean)
    plt.fill_between(epochs, dmean-dstd, dmean+dstd, alpha = 0.5)

    #plt.show()
    

#only run on sparse, meant to show how much graph changes with changing amounts of forward steps
def show_forward_effect(sparse):

    fig, ax = plt.subplots()
    #axes.set_xlim([0,10])
    #axes.set_ylim([0,0.3])

    cam = Camera(fig)

    for i in range(1,10):
        epochs = sparse[0]['epoch']
        sl = []
        for s in sparse:
          
            if s['forward_steps'] == i:
                sl.append(s['test_acc'])
        smean = np.array(sl).transpose().mean(1)
        sstd = np.array(sl).transpose().std(1)

        plt.plot(epochs,smean,color='blue')
        plt.fill_between(epochs, smean-sstd, smean+sstd, alpha = 0.5,color='blue')
        ax.text(0.5, 1.01, f"N forward steps = {i}", transform=ax.transAxes)
        #ax.set_title(f"N forward steps = {i}")
        #plt.show()
        cam.snap()
       
    anim = cam.animate()
    anim.save("forward_step_inc.gif")

#only run on sparse, meant to show how mean gradient changes for first iteration as n forward steps increases
def show_gradient_effects(sparse):

    fig, ax = plt.subplots()
    sl = [[] for _ in range(5)]
    for s in sparse:
        forward_steps = s['n_forward_steps']
        mean = float(s['gradient_mean'][0])
        sl[forward_steps].append(mean)
        
    smean = np.array(sl).transpose().mean(1)
    sstd = np.array(sl).transpose().std(1)

    plt.plot(epochs,smean,color='blue')
    plt.fill_between(epochs, smean-sstd, smean+sstd, alpha = 0.5,color='blue')
    #ax.text(0.5, 1.01, f"N forward steps = {i}", transform=ax.transAxes)
    ax.set_title(f"Mean gradient on first pass with N total forward passes")
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    dset = args.dataset
    s,d = load_jsons(dset)
    show_forward_effect(s)