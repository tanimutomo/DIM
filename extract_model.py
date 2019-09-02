import torch
import sys


def main(name, eval_mode, model_name):
    if eval_mode:
        name = eval_mode + '_' + name
    path = '/home/tanimu/.cortex/ckpt/DIM_CIFAR10_{}/' \
           'binaries/DIM_CIFAR10_{}_final.t7'.format(
                   name, name)
    nets = torch.load(path)['nets']

    if model_name == 'encoder':
        model = nets['Controller.encoder'].encoder
    elif model_name == 'conv_classifier':
        model = nets['conv[2](256,4,4).classifier']
    elif model_name == 'fc_classifier':
        model = nets['fc[4](1024,).classifier']
    elif model_name == 'glob_classifier':
        model = nets['glob[-1](64,).classifier']
    else:
        raise NotImplementedError

    torch.save(model.state_dict(),
               './models/{}_{}.pth'.format(name, model_name))


if __name__ == '__main__':
    name = sys.argv[1]
    eval_mode = sys.argv[2]
    model_name = sys.argv[3]
    main(name, eval_mode, model_name)
