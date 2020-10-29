from trainer import *

def mnist_exp(basic_config,spec_config):
    dynamic = False
    loss = nn.NLLLoss()
    output_act = nn.LogSoftmax(dim=1)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    test_dset = datasets.MNIST('../data', train=False,
                       transform=transform)
    run_trainer(basic_config,spec_config,loss,output_act,train_dset,test_dset,transform,dynamic)

if __name__ == '__main__':
    with open("configs/basic.yml") as s:
        basic_config = yaml.load(s)
    with open("configs/mnist.yml") as s:
        mnist_config = yaml.load(s)

    mnist_exp(basic_config,mnist_config)