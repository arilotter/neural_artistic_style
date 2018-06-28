#!/usr/bin/env python

import sys
import os
import math
import argparse
import numpy as np
import scipy.misc
import scipy.ndimage as nd
import deeppy as dp
import PIL.Image
import random
import time

from matconvnet import vgg_net
from style_network import StyleNetwork


def weight_tuple(s):
    try:
        conv_idx, weight = map(float, s.split(','))
        return conv_idx, weight
    except:
        raise argparse.ArgumentTypeError('weights must by "int,float"')


def float_range(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0, 1]" % x)
    return x


def weight_array(weights):
    array = np.zeros(19)
    for idx, weight in weights:
        array[idx] = weight
    norm = np.sum(array)
    if norm > 0:
        array /= norm
    return array


def imread(path):
    return scipy.misc.imread(path).astype(dp.float_)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    PIL.Image.fromarray(img).save(path, quality=97)
    print "Wrote %s"%path


def to_bc01(img):
    return np.transpose(img, (2, 0, 1))[np.newaxis, ...]


def to_rgb(img):
    return np.transpose(img[0], (1, 2, 0))


def run():
    parser = argparse.ArgumentParser(
        description='Neural artistic style. Generates an image by combining '
                    'the subject from one image and the style from another.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--subject', required=True, type=str,
                        help='Subject image.')
    parser.add_argument('--style', required=True, type=str,
                        help='Style image.')
    parser.add_argument('--output', default='out.png', type=str,
                        help='Output image.')
    parser.add_argument('--init', default=None, type=str,
                        help='Initial image. Subject is chosen as default.')
    parser.add_argument('--init-noise', default=0.0, type=float_range,
                        help='Weight between [0, 1] to adjust the noise level '
                             'in the initial image.')
    parser.add_argument('--random-seed', default=None, type=int,
                        help='Random state.')
    parser.add_argument('--animation', default='animation', type=str,
                        help='Output animation directory.')
    parser.add_argument('--iterations', default=500, type=int,
                        help='Number of iterations to run.')
    parser.add_argument('--scales', default=4, type=int,
                        help='Number of scales to use.')
    parser.add_argument('--learn-rate', default=2.0, type=float,
                        help='Learning rate.')
    parser.add_argument('--smoothness', type=float, default=5e-8,
                        help='Weight of smoothing scheme.')
    parser.add_argument('--subject-weights', nargs='*', type=weight_tuple,
                        default=[(9, 1)],
                        help='List of subject weights (conv_idx,weight).')
    parser.add_argument('--style-weights', nargs='*', type=weight_tuple,
                        default=[(0, 1), (2, 1), (4, 1), (8, 1), (12, 1)],
                        help='List of style weights (conv_idx,weight).')
    parser.add_argument('--subject-ratio', type=float, default=2e-2,
                        help='Weight of subject relative to style.')
    parser.add_argument('--pool-method', default='avg', type=str,
                        choices=['max', 'avg'], help='Subsampling scheme.')
    parser.add_argument('--network', default='imagenet-vgg-verydeep-19.mat',
                        type=str, help='Network in MatConvNet format).')
    parser.add_argument('--outer_it', default=8000, type=int, help='Outer iterations.')
    parser.add_argument('--inner_it', default=3, type=int, help='Inner iterations.')
    parser.add_argument('--patch_size', default=512, type=int, help='Patchsize.')
    args = parser.parse_args()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    layers, pixel_mean = vgg_net(args.network, pool_method=args.pool_method)

    # Inputs
    style_img = imread(args.style) - pixel_mean
    subject_img = imread(args.subject) - pixel_mean
    print "style_img", style_img.shape
    print "subject_img", subject_img.shape
    if args.init is None: init_img = subject_img
    else: init_img = imread(args.init) - pixel_mean
    noise = np.random.normal(size=init_img.shape, scale=np.std(init_img)*1e-1)
    init_img = init_img * (1 - args.init_noise) + noise * args.init_noise

    # Setup network
    subject_weights = weight_array(args.subject_weights) * args.subject_ratio
    style_weights = weight_array(args.style_weights)
   
    max_patch_size = np.array([args.patch_size, args.patch_size])
    inner_iterations = 10

    for outer_it in range(0,args.outer_it):
      scale = 1.0
      for sc in range(0, args.scales - 1): 
        if random.random() < 0.25: scale*=2.0
      print "OUTER_IT: ", outer_it, "SCALE:", scale

      effective_patch_size = max_patch_size * scale
      patch_size = (int(np.min([style_img.shape[0], subject_img.shape[0], init_img.shape[0], effective_patch_size[0]])),
                    int(np.min([style_img.shape[1], subject_img.shape[1], init_img.shape[1], effective_patch_size[1]])))

      rel_x = random.random()
      rel_y = random.random()
      x = int(math.floor(rel_x*(style_img.shape[0]-patch_size[0])))
      y = int(math.floor(rel_y*(style_img.shape[1]-patch_size[1])))
      style_patch = style_img[x:x+patch_size[0], y:y+patch_size[1], :]
      
      x = int(math.floor(rel_x*(subject_img.shape[0]-patch_size[0])))
      y = int(math.floor(rel_y*(subject_img.shape[1]-patch_size[1])))
      subject_patch = subject_img[x:x+patch_size[0], y:y+patch_size[1], :]
      
      x = int(math.floor(rel_x*(init_img.shape[0]-patch_size[0])))
      y = int(math.floor(rel_y*(init_img.shape[1]-patch_size[1])))
      init_patch = init_img[x:x+patch_size[0], y:y+patch_size[1], :]

      style_patch_scaled = nd.zoom(style_patch, (1.0 / scale, 1.0 / scale, 1), order=1) 
      subject_patch_scaled = nd.zoom(subject_patch, (1.0 / scale, 1.0 / scale, 1), order=1) 
      init_patch_scaled = nd.zoom(init_patch, (1.0 / scale, 1.0 / scale, 1), order=1) 
      
      net = StyleNetwork(layers, 
                         to_bc01(init_patch_scaled), 
                         to_bc01(subject_patch_scaled),
                         to_bc01(style_patch_scaled), 
                         subject_weights, style_weights,
                         args.smoothness)
      params = net.params
      learn_rule = dp.Adam(learn_rate=args.learn_rate)
      learn_rule_states = [learn_rule.init_state(p) for p in params]
      for i in range(args.inner_it):
          cost = np.mean(net.update())
          for param, state in zip(params, learn_rule_states):
              learn_rule.step(param, state)
          print('Iteration: %i, cost: %.4f' % (i, cost))
     
      result_patch_scaled = to_rgb(net.image) - init_patch_scaled 
      result_patch = nd.zoom(result_patch_scaled, (scale, scale, 1), order=1)
      
      border_cut = 10
      init_img[x+border_cut:x+result_patch.shape[0]-border_cut, 
               y+border_cut:y+result_patch.shape[1]-border_cut, :] += result_patch[border_cut:-border_cut, border_cut:-border_cut]
      imsave("%s.jpg"%args.output , init_img + pixel_mean)

if __name__ == "__main__":
    run()
