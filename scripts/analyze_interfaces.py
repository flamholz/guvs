#!/usr/bin/python

import glob
import argparse
import pylab
import numpy as np
import matplotlib.pyplot as plt
import os
import unittest

from image import util
from image.extraction.boundary_finder import BoundaryFinder
from image.extraction.circle_finder import CircleFinder
from image.extraction.region_finder import RegionFinder
from jinja2 import Environment, FileSystemLoader
from os import path
from skimage.io import imread
from skimage import exposure
from skimage import filter as skfilter
from skimage.util import img_as_ubyte, img_as_bool
from skimage.morphology import watershed, disk

from sklearn import mixture


def Main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('image_directory',
                        help='Where to load images and bounding boxes from.')
    parser.add_argument('-o', '--out_directory',
                        help='Where to save output to.')
    
    args = parser.parse_args()
    
    glob_str = path.join(path.abspath(args.image_directory), '*.tif')
    image_fnames = glob.glob(glob_str)
    out_dir = path.abspath(args.out_directory)
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    
    results = []
    for img_fname in image_fnames:
        print 'Analyzing', img_fname
        cur_img = imread(img_fname)
        preprocessed_img = img_as_ubyte(exposure.rescale_intensity(cur_img))
        
        base_name, unused_ext = path.splitext(img_fname)
        bbox_glob_str = base_name + '_bbox_*'
        bbox_fnames = glob.glob(bbox_glob_str)
        
        for bbox_fname in bbox_fnames:
            bbox_base_name, _ = path.splitext(path.split(bbox_fname)[1])
            print '\tExamining bounding box', bbox_fname
            
            bbox = util.load_fiji_bbox(bbox_fname)
            min_r, min_c, max_r, max_c = bbox
            cropped_image = cur_img[min_r:max_r,min_c:max_c]
            cropped_preprocessed = preprocessed_img[min_r:max_r,min_c:max_c]
            
            # Save the cropped image
            cropped_img_name = path.join(out_dir, '%s_cropped.png' % bbox_base_name)
            util.save_image(cropped_image, cropped_img_name, title='Raw Image')
            preprocessed_img_name = path.join(out_dir, '%s_denoised.png' % bbox_base_name)
            util.save_image(cropped_preprocessed, preprocessed_img_name, title='Preprocessed Image')

            intensities = cropped_image.flatten().copy()
            g = mixture.GMM(n_components=3, covariance_type='tied')
            g.fit(intensities)
            
            xs = np.arange(intensities.min() / 2, intensities.max())
            log_pdf, responsibilities = g.score_samples(xs)
            pdf = np.exp(log_pdf)
            pdf_individual = responsibilities * pdf[:, np.newaxis]
            
            pylab.figure(figsize=(10,10))
            pylab.plot(xs, pdf, lw=3, label='pdf')
            pylab.plot(xs, pdf_individual, linestyle='--', lw=3, label='individual pdfs')
            pylab.hist(intensities, normed=True, bins=1000, color='gray', alpha=0.3, label='data')
            pylab.legend()
            intensity_fit_fname = path.join(out_dir, '%s_intensity_fit.png' % bbox_base_name)
            pylab.savefig(intensity_fit_fname)
            pylab.close()
            
            predictions = g.predict(cropped_image.flatten().tolist())
            interface_mask = predictions.reshape(cropped_preprocessed.shape)
            means = sorted(g.means_[:, 0].tolist())
            bg_idx = np.argmin(g.means_)
            interface_idx = np.argmax(g.means_)
            idxs = range(3)
            idxs.remove(bg_idx)
            idxs.remove(interface_idx)
            boundary_idx = idxs[0]
            
            bg_vals = cropped_image[np.where(interface_mask == bg_idx)]
            boundary_vals = cropped_image[np.where(interface_mask == boundary_idx)]
            interface_vals = cropped_image[np.where(interface_mask == interface_idx)]

            bg_mean = bg_vals.mean()
            boundary_mean = boundary_vals.mean()
            interface_mean = interface_vals.mean()
            
            bg_std = bg_vals.std()
            boundary_std = boundary_vals.std()
            interface_std = interface_vals.std()
            
            pylab.figure()
            pylab.bar(np.arange(3) + 0.15, [bg_mean, boundary_mean, interface_mean],
                      yerr=[bg_std, boundary_std, interface_std],
                      width=0.7,
                      color='y')
            pylab.xticks(np.arange(3) + 0.5, ['bg', 'boundary', 'interface'])
            intensity_img_name = path.join(out_dir, '%s_intensities.png' % bbox_base_name)
            pylab.savefig(intensity_img_name)
            pylab.close()

            interface_img_name = path.join(out_dir, '%s_interfaces.png' % bbox_base_name)
            util.save_image(interface_mask, interface_img_name)

            result_data = {'name': bbox_base_name,
                           'cropped_image': cropped_img_name,
                           'preprocessed_image': preprocessed_img_name,
                           'intensity_fit_plot': intensity_fit_fname,
                           'intensities_plot': intensity_img_name,
                           'interface_image': interface_img_name}
            results.append(result_data)
    
    html_fname = path.join(out_dir, 'index.html')
    with open(html_fname, 'w') as out_f:
        template_env = Environment(loader=FileSystemLoader('templates'))
        t = template_env.get_template('guv_interfaces.html')
        print 'Writing summary HTML to', html_fname
        out_f.write(t.render(results=results))
  
if __name__ == '__main__':
    Main()
