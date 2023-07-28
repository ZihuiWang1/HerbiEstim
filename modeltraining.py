import sys
sys.path.append('pix2pix')
from options.train_options import TrainOptions
from train import main

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    main(opt)
