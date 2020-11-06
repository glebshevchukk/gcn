from trainer import *
import argparse
from settings import *
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="all")

#experiment to see how changing the number of forward passes affects performance
def n_pass_exp(basic_config,spec_config,setting):
    dynamic = False
    loss = setting.loss
    train_dset = setting.train_dset
    test_dset = setting.test_dset

    for rep in range(basic_config['n_repeats']):
        for hid in range(len(basic_config['n_hidden'])):
            #for passes in range(1,basic_config['n_forward_steps']):
                #first run dense resnet
                #run_trainer(basic_config,spec_config,loss,train_dset,test_dset,dynamic,True,rep,hid)
                #trun gcn
            passes = 10
            run_trainer(basic_config,spec_config,loss,train_dset,test_dset,dynamic,False,rep,hid,passes)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset in valid_exps:
        exp = [args.dataset]
    else:
        exp = valid_exps.keys()
    for e in exp:
        with open("configs/basic.yml") as s:
            basic_config = yaml.load(s)
        with open("configs/"+e+".yml") as s:
            spec_config = yaml.load(s)
        setting = valid_exps[e]
        n_pass_exp(basic_config,spec_config,setting)