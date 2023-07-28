import sys
import os
sys.path.append('pix2pix')
from options.test_options import TestOptions
from predict import main



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    
    img = os.listdir(os.path.join(opt.dataroot,'test'))
    if '.DS_Store' in img:
        img.remove('.DS_Store')
    opt.num_test = len(img)
    opt.netG = 'unet_256'
    opt.norm = 'batch'
    main(opt)
