#!/usr/bin/python

import argparse
import numpy as np
import os
import pandas as pd
import pylab
import matplotlib.pyplot as plt

from matplotlib import animation
from os import path
from scipy import ndimage
from scipy import optimize
from scipy import signal
from skimage import draw
from skimage.io import ImageCollection
from skimage.filter import rank
from skimage import measure
from skimage import segmentation
from skimage.morphology import disk

from sklearn import mixture
from sklearn.externals import joblib

from image import guv_utils
from image.guv_utils import subtract_bg, to_fold_change, find_largest_region
from image.util import load_fiji_bbox, load_fiji_line, image_datetime, get_duration
from image.util import map_line_to_bbox

def mean_min_max(x):
    return (np.mean(x), np.min(x), np.max(x))
        

parser = argparse.ArgumentParser(description='Analyze GUV videos')
parser.add_argument('-b', '--base_directory',
                    default='/Users/flamholz/Documents/Rotations/Fletcher/atto_390_dGFP_4_5_14/videos/merge_2_1/',
                    help='Location of image and other files.')
args = parser.parse_args()

# Paths to files
base_dir = args.base_directory
bbox_path = path.join(base_dir, 'bbox.txt')
line_path = path.join(base_dir, 'line.txt')
blue_gmm_path = path.join(base_dir, 'blue_gmm.pkl')
green_gmm_path = path.join(base_dir, 'green_gmm.pkl')
raw_blue_channel_path = path.join(base_dir, 'blue_channel/*.TIF')
blue_channel_path = path.join(base_dir, 'blue_channel_mean/*.tif')
green_channel_path = path.join(base_dir, 'green_channel_mean/*.tif')
output_path = path.join(base_dir, 'analysis')

# Check various stuff exists
assert path.exists(bbox_path)
assert path.exists(blue_gmm_path)
assert path.exists(green_gmm_path)
if not path.exists(output_path):
    os.makedirs(output_path)

bbox = load_fiji_bbox(bbox_path)
min_r, min_c, max_r, max_c = bbox
line_endpoints = load_fiji_line(line_path)
line_start, line_end = map_line_to_bbox(line_endpoints, bbox)

print 'Loading mixture models'
blue_gmm = joblib.load(blue_gmm_path)
green_gmm = joblib.load(green_gmm_path)

print 'Loading images'
blue_collection = ImageCollection(blue_channel_path)
green_collection = ImageCollection(green_channel_path)

print 'Calculating timing'
raw_blue_collection = ImageCollection(raw_blue_channel_path)
total_secs = get_duration(raw_blue_collection)
secs_per_frame = total_secs / float(len(raw_blue_collection.files))
print '\tTotal experiment duration: %.1f seconds' % total_secs
print '\t%.2f seconds per frame' % secs_per_frame

# Now analyze the video.
enrichment_at_interface = []
fluorescence_not_at_interface = []
count_at_interface = []
count_not_interface = []
interface_imgs = []
intensity_imgs = []
total_intensity = []
linescan_enrichments = []
linescans = []
blue_max = 0
green_max = 0
for i, (blue_channel, green_channel) in enumerate(zip(blue_collection, green_collection)):
    total_intensity.append(green_channel.flatten().sum())
    green_channel = green_channel[min_r:max_r, min_c:max_c]
    blue_channel = blue_channel[min_r:max_r, min_c:max_c]
    
    green_channel_nobg = subtract_bg(green_channel, green_gmm)
    blue_channel_nobg = subtract_bg(blue_channel, blue_gmm)
    
    scan_enrichment, scan = guv_utils.calc_linescan_enrichment(green_channel_nobg, line_start, line_end)
    linescan_enrichments.append(scan_enrichment)
    linescans.append(scan)
    
    blue_fold_change = to_fold_change(blue_channel_nobg)
    green_fold_change = to_fold_change(green_channel_nobg)
    counts_at = []
    counts_not_at = []
    enrichments_at = []
    fluor_not_at = []
    for thresh in np.arange(1.7, 2.3, 0.1):
        # Positions where both membrane & gfp dye are at least thresh-fold enriched.
        interface_positions = np.where(np.logical_and(blue_fold_change > thresh,
                                                      green_fold_change > thresh))
        
        non_zero_blue_below = np.logical_and(blue_channel_nobg > 0,
                                             blue_fold_change <= thresh)
        # Positions with above-background signal of GFP and membrane dye that are 
        # nonetheless below the threshold.
        non_interface_positions = np.where(non_zero_blue_below)
        
        interface_img = np.zeros(blue_channel.shape)
        interface_img[interface_positions] = 0.75
        largest_region = find_largest_region(interface_img)
        largest_region_size = largest_region.area
        largest_region_idxs = (largest_region.coords[:,0], largest_region.coords[:,1])
        interface_enrichment = (green_channel_nobg[largest_region_idxs].mean() /
                                green_channel_nobg[non_interface_positions].mean())
        
        counts_at.append(largest_region_size)
        total_positions = len(non_interface_positions[0]) + len(interface_positions[0])
        counts_not_at.append(float(total_positions - largest_region_size))
        enrichments_at.append(interface_enrichment)
        fluor_not_at.append(green_channel_nobg[non_interface_positions].mean())
        
        if thresh == 2.0:
            interface_img[non_interface_positions] = 0.25
            interface_img[largest_region_idxs] = 1.0
            interface_imgs.append(interface_img)
            intensity_imgs.append((blue_channel_nobg, green_channel_nobg))
            blue_max = max(blue_max, blue_channel_nobg.max())
            green_max = max(green_max, green_channel_nobg.max())
            
    # TODO: Wilcoxon rank sum test for non-parametric comparison of means.
    count_at_interface.append(counts_at)
    count_not_interface.append(counts_not_at)
    enrichment_at_interface.append(mean_min_max(enrichments_at))
    fluorescence_not_at_interface.append(mean_min_max(fluor_not_at))
    
enrichment_at_interface = np.array(enrichment_at_interface)
fluorescence_not_at_interface = np.array(fluorescence_not_at_interface)
count_at_interface = np.vstack(count_at_interface)
count_not_interface = np.vstack(count_not_interface)

# Video of interface intensity and extracted interface.
fig = pylab.figure(figsize=(20,10))
artists = []
for i, (interface_img, (blue_img, green_img)) in enumerate(zip(interface_imgs, intensity_imgs)):
    t_secs = secs_per_frame * i
    frame_artists = []
    pylab.subplot(131)
    pylab.axis('off')
    pylab.title('Atto390', fontsize=22)
    frame_artists.append(pylab.imshow(blue_img, cmap=plt.cm.jet, interpolation='nearest', vmin=0.0, vmax=blue_max))
    pylab.subplot(132)
    pylab.axis('off')
    pylab.title('dGFP', fontsize=22)
    frame_artists.append(pylab.imshow(green_img, cmap=plt.cm.jet, interpolation='nearest', vmin=0.0, vmax=green_max))
    pylab.subplot(133)
    pylab.axis('off')
    pylab.title('Interface', fontsize=22)
    frame_artists.append(pylab.imshow(interface_img, cmap=plt.cm.jet, interpolation='nearest', vmin=0.0, vmax=1.0))
    #frame_artists.append(pylab.colorbar())
    max_y, max_x = interface_img.shape
    frame_artists.append(pylab.text(max_x-40, max_y + 20, '%.1f s' % t_secs, fontsize=22, color='k'))
    artists.append(frame_artists)
    
ms_per_frame = int(secs_per_frame * 1000.0 / 20)
ani = animation.ArtistAnimation(fig, artists, interval=ms_per_frame, blit=True,
    repeat_delay=1000)
output_video_path = path.join(output_path, 'membrane_intensity.mp4')
ani.save(output_video_path)

# Size of the interface in pixels.
pylab.figure()
xs = np.arange(count_at_interface.shape[0]) * secs_per_frame
yerr = [np.min(count_at_interface, axis=1), np.max(count_at_interface, axis=1)]
mean_count = np.mean(count_at_interface, axis=1)
pylab.errorbar(xs, mean_count, yerr=yerr, fmt='r.')
pylab.plot(xs, pd.rolling_mean(mean_count, 5), 'k--', lw=3)
pylab.title('Absolute Interface Size')
pylab.xlabel('Time (Seconds)')
pylab.ylabel('Number of Pixels at Interface')
output_fig_path = path.join(output_path, 'interface_size.png')
pylab.savefig(output_fig_path)

# Size of the interface as a fraction of the total membrane in frame.
pylab.figure()
frac = np.divide(count_at_interface, count_at_interface + count_not_interface)
pct = 100*frac
mean_pct = np.mean(pct, axis=1)
yerr = [np.min(pct, axis=1), np.max(pct, axis=1)]
pylab.errorbar(xs, mean_pct, yerr=yerr, fmt='g.')
pylab.plot(xs, pd.rolling_mean(mean_pct, 5), 'k--', lw=3)
pylab.title('Relative Interface Size')
pylab.xlabel('Time (Seconds)')
pylab.ylabel('% of Pixels at Interface')
output_fig_path = path.join(output_path, 'relative_interface_size.png')
pylab.savefig(output_fig_path)

# Raw fluorescence at non-interface
pylab.figure()
fl_not_at_pct = 100.0 * fluorescence_not_at_interface / fluorescence_not_at_interface[:,0].max()
yerr = [fl_not_at_pct[:,0] - fl_not_at_pct[:,1],
        fl_not_at_pct[:,2] - fl_not_at_pct[:,0]]
pylab.errorbar(xs, fl_not_at_pct[:,0], fmt='b.', yerr=yerr)
pylab.xlabel('Time (Seconds)')
pylab.ylabel('% Max GFP Fluoresence')
output_fig_path = path.join(output_path, 'non_interface_fluorescence.png')
pylab.savefig(output_fig_path)

# Raw fluorescence enrichment at interface
pylab.figure()
yerr = [enrichment_at_interface[:,0] - enrichment_at_interface[:,1],
        enrichment_at_interface[:,2] - enrichment_at_interface[:,0]]
pylab.errorbar(xs, enrichment_at_interface[:,0], fmt='b.',
               yerr=yerr, label="measured")
pylab.xlabel('Time (Seconds)')
pylab.ylabel('GFP Fluoresence Enrichment')
output_fig_path = path.join(output_path, 'interface_enrichment.png')
pylab.savefig(output_fig_path)

# Intensity enrichment at interface relative to non-interface membrane as a function of time.
def saturating(t, fmax, kt, f0):
    return fmax * (1.0 - np.exp(-t / kt)) + f0
enrichments = enrichment_at_interface[:,0]
median_enrichments = pd.rolling_median(enrichments, 5)
median_enrichments[:5] = enrichments[:5]  # remove nans from windowing
dispersion = np.abs(np.log2(enrichments / median_enrichments))  # fold deviation from median
reasonable_enrichments = np.where(dispersion < 0.6)
filtered_enrichments = enrichments[reasonable_enrichments]
filtered_xs = xs[reasonable_enrichments]
filtered_mins = enrichment_at_interface[reasonable_enrichments,1]
filtered_maxs = enrichment_at_interface[reasonable_enrichments,2]
error_window = np.abs(filtered_mins - filtered_maxs).flatten()
popt, pcov = optimize.curve_fit(saturating, filtered_xs, filtered_enrichments, sigma=1.0/error_window)
fmax, kt, f0 = popt
max_enrichment = (fmax + f0)
half_max = f0 + (fmax / 2.0)
t_half = (kt) * np.log(2)
predicted_ys = saturating(filtered_xs, fmax, kt, f0)
residuals = filtered_enrichments - predicted_ys
r = np.corrcoef(filtered_enrichments, predicted_ys)[0,1]
r2 = r**2

pylab.figure(figsize=(20,10))
#pylab.subplot(121)
yerr = [(filtered_enrichments - filtered_mins).flatten(),
        (filtered_maxs - filtered_enrichments).flatten()]
pylab.errorbar(filtered_xs, filtered_enrichments, fmt='b.',
               yerr=yerr, label="measured")
pylab.plot(filtered_xs, predicted_ys, 'g--', lw=2, label="fit ($R^2 = %.2f$)" % r2)
pylab.plot(xs, np.ones(xs.size) * (fmax + f0), 'r--', lw=2, label="$f_{max} = %.2f$" % max_enrichment)
pylab.plot(xs, np.ones(xs.size) * f0, 'c--', lw=2, label="$f_0 = %.2f$" % f0)
pylab.plot([t_half, t_half], [0, half_max], 'k--', lw=2)
pylab.plot([0, t_half], [half_max, half_max], 'k--', lw=2, label="$t_{1/2} = %.2f$" % t_half)
pylab.xlabel('Time (Seconds)')
pylab.ylabel('GFP fluoresence enrichment interface')
pylab.title('$f_t = %.1f (1 - \exp(-t / %.1f)) + %.1f$' % (fmax, kt, f0))
pylab.legend(loc=4)
"""
pylab.subplot(122)
pylab.plot(filtered_xs, residuals, 'rx')
pylab.plot(xs, np.ones(xs.size) * residuals.mean())
pylab.plot(filtered_xs, pd.rolling_mean(residuals, 5), 'k-', lw=2)
pylab.xlabel('Time (Seconds)')
pylab.ylabel('Absolute Residual')
pylab.title('Residuals')
"""

output_fig_path = path.join(output_path, 'interface_enrichment_fit.png')
pylab.savefig(output_fig_path)

# Plot a single linescan
pylab.figure()
for idx in [180, 140, 70, 20, 10]:
    pylab.plot(np.arange(len(linescans[idx])), linescans[idx], '-',
               label='t = %.1f s' % (idx*secs_per_frame), lw=2)
    peaks = signal.argrelmax(linescans[idx])
    pylab.plot(peaks[0], linescans[idx][peaks], '^')
pylab.xlabel('Linear Position (Pixels)', fontsize=18)
pylab.ylabel('Fluoresence Above Background (AU)', fontsize=18)
pylab.legend()
output_fig_path = path.join(output_path, 'single_linescan.png')
pylab.savefig(output_fig_path)

# Do the same fit to a linescan
popt, pcov = optimize.curve_fit(saturating, xs, linescan_enrichments)
fmax, kt, f0 = popt
max_enrichment = (fmax + f0)
half_max = f0 + (fmax / 2.0)
t_half = (kt) * np.log(2)
predicted_ys = saturating(xs, fmax, kt, f0)
residuals = linescan_enrichments - predicted_ys
r = np.corrcoef(linescan_enrichments, predicted_ys)[0,1]
r2 = r**2

# Line scan enrichment at interface
pylab.figure()
pylab.plot(xs, linescan_enrichments, 'b.', label='measured')
pylab.plot(xs, predicted_ys, 'g--', lw=2, label="fit ($R^2 = %.2f$)" % r2)
pylab.plot(xs, np.ones(xs.size) * (fmax + f0), 'r--', lw=2, label="$f_{max} = %.2f$" % max_enrichment)
pylab.plot(xs, np.ones(xs.size) * f0, 'c--', lw=2, label="$f_0 = %.2f$" % f0)
pylab.plot([t_half, t_half], [0, half_max], 'k--', lw=2)
pylab.plot([0, t_half], [half_max, half_max], 'k--', lw=2, label="$t_{1/2} = %.2f$" % t_half)
pylab.title('$f_t = %.1f (1 - \exp(-t / %.1f)) + %.1f$' % (fmax, kt, f0))
pylab.xlabel('Time (Seconds)')
pylab.ylabel('Relative Intensity of Highest Peak (AU)')
pylab.legend(loc=4)
output_fig_path = path.join(output_path, 'linescan_enrichment.png')
pylab.savefig(output_fig_path)

pylab.figure()
pylab.plot(xs, dispersion)
pylab.title('Dispersion from Median over Time')
pylab.xlabel('Time (Seconds)')
pylab.ylabel('dispersion (abs(log2(val/(window median))))')
output_fig_path = path.join(output_path, 'enrichment_dispersion_from_median.png')
pylab.savefig(output_fig_path)


pct_max_intensity = np.array(total_intensity)
pct_max_intensity = 100.0 * pct_max_intensity / pct_max_intensity.max()

pylab.figure()
pylab.title('Total dGFP Intensity in FOV')
pylab.plot(xs, pct_max_intensity, 'b.')
pylab.plot(xs, pd.rolling_mean(pct_max_intensity, 5), 'k-', lw=2)
pylab.xlabel('Time (Seconds)')
pylab.ylabel('% Maximum dGFP Fluoresence Intensity')
output_fig_path = path.join(output_path, 'total_fluoresence_in_time.png')
pylab.savefig(output_fig_path)

pylab.close('all')