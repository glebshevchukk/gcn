from trainer import *


def cifar_exp(basic_config,spec_config):
    dynamic = True
    loss = nn.NLLLoss()
    output_act = nn.LogSoftmax(dim=1)
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dset = datasets.CIFAR10(root='./data/cifar', train=True,
                                            download=True, transform=transform)

    test_dset = datasets.CIFAR10(root='./data/cifar', train=False,
                                        download=True, transform=transform)
    run_trainer(basic_config,spec_config,loss,output_act,train_dset,test_dset,transform,dynamic)

if __name__ == '__main__':
    with open("configs/basic.yml") as s:
        basic_config = yaml.load(s)
    with open("configs/cifar.yml") as s:
        cifar_config = yaml.load(s)

    cifar_exp(basic_config,cifar_config)