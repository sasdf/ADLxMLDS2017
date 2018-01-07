#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument(
        'dataFolder',
        metavar='<dataFolder>', type=str, nargs='?',
        help='path to the data folder',
        default='/tmp2/sasdf/adl/hw4/data')
        #  default='../data')

parser.add_argument(
        'outputFile',
        metavar='<outputFile>', type=str, nargs='?',
        help='Place the output into <outputFile>',
        default='output/out')


FLAGS = parser.parse_args()
