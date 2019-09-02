import torch
import sys


def main(name):
    state = torch.load(
            '/home/tanimu/.cortex/ckpt/DIM_CIFAR10_{}/'
            'binaries/DIM_CIFAR10_{}_final.t7'.format(
                name, name)
            )
    nets = state['nets']
    encoder = nets['Controller.encoder']
    encoder = encoder.encoder
    torch.save(encoder.state_dict(), './models/{}.pth'.format(name))


if __name__ == '__main__':
    name = sys.argv[1]
    main(name)
