#!/usr/bin/python

import argparse
import shutil
import os
import glob

from os import path

parser = argparse.ArgumentParser(description='Analyze GUV videos')
parser.add_argument('-b', '--base_directory',
                    default='/Users/flamholz/Documents/Rotations/Fletcher/final_push/dGFP_ChChCh_atto390/take11',
                    help='Location of microscope images.')
args = parser.parse_args()

is_thumb = lambda x: 'thumb' not in x.lower()

green_glob = glob.glob(path.join(args.base_directory, '*491*_t*.TIF'))
green_no_thumbs = filter(is_thumb, green_glob)

if green_no_thumbs:
    print 'Moving', len(green_no_thumbs), 'green channel files'
    green_channel_dir = path.join(args.base_directory, 'green_channel')
    if not path.exists(green_channel_dir):
        os.makedirs(green_channel_dir)
    for fname in green_no_thumbs:
        shutil.move(fname, green_channel_dir)

blue_glob = glob.glob(path.join(args.base_directory, '*405*_t*.TIF'))
blue_no_thumbs = filter(is_thumb, blue_glob)

if blue_no_thumbs:
    print 'Moving', len(blue_no_thumbs), 'blue channel files'
    blue_channel_dir = path.join(args.base_directory, 'blue_channel')
    if not path.exists(blue_channel_dir):
        os.makedirs(blue_channel_dir)
    for fname in blue_no_thumbs:
        shutil.move(fname, blue_channel_dir)

# May be the 3rd or 4th channel acquired depending on situation.
red_glob = glob.glob(path.join(args.base_directory, '*w4561*_t*.TIF'))
red_no_thumbs = filter(is_thumb, red_glob)

if red_no_thumbs:
    print 'Moving', len(red_no_thumbs), 'red channel files'
    red_channel_dir = path.join(args.base_directory, 'red_channel')
    if not path.exists(red_channel_dir):
        os.makedirs(red_channel_dir)
    for fname in red_no_thumbs:
        shutil.move(fname, red_channel_dir)