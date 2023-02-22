from options.test_options import TestOptions
from data import create_dataset
import numpy as np
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, set_seed

if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    meters_tst = {stat: AverageMeter() for stat in model.loss_names}

    set_seed(opt.seed)

    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.testset_name, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    num_of_views = 128
    theta_list = np.linspace(0, np.pi * 2, num_of_views)
    for i, data in enumerate(dataset):
        M = opt.n_img_each_scene 
        N = len(theta_list) // M
        # if i < 3:
        #     continue
        for j in range(N):
            thetas = theta_list[j * M: (j + 1) * M]
            visualizer.reset()
            model.set_input(data, thetas)  # unpack data from data loader
            model.test()           # run inference: forward + compute_visuals

            visuals = model.get_current_visuals()
            visualizer.display_current_results(visuals, epoch=None, save_result=False)
            img_path = model.get_image_paths()

            img_path = [p.replace('.png', f'_{str(j).zfill(4)}.png') for p in img_path]
            print(img_path)
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)
        break

    webpage.save()