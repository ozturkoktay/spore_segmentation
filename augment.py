import Augmentor
from Augmentor import Pipeline


def augment_images():
    data_dir = "output/train/"
    sample = Augmentor.Pipeline(data_dir)
    Pipeline.set_seed(130)
    sample.resize(width=224, height=224, probability=1)
    sample.rotate90(probability=0.5)
    sample.rotate270(probability=0.5)
    sample.flip_left_right(probability=0.8)
    sample.flip_top_bottom(probability=0.3)
    sample.rotate(probability=1, max_left_rotation=2, max_right_rotation=5)
    sample.gaussian_distortion(probability=1, grid_width=4, grid_height=4, magnitude=2,
                               corner="bell", method="in", mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)
    sample.random_brightness(probability=1, min_factor=0.5, max_factor=1.5)
    sample.random_color(probability=1, min_factor=0.5, max_factor=2)
    sample.status()
    sample.sample(1000, multi_threaded=True)


augment_images()
