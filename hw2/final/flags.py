import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument(
        'dataFolder',
        metavar='<dataFolder>', type=str, nargs='?',
        help='path to the data folder',
        default='/tmp2/sasdf/adl/hw2/data')
        #  default='../data')

parser.add_argument(
        'outputFile',
        metavar='<outputFile>', type=str, nargs='?',
        help='Place the output into <outputFile>',
        default='output/out')


parser.add_argument(
        'outputFilePeer',
        metavar='<outputFilePeer>', type=str, nargs='?',
        help='Place the output into <outputFile>',
        default='output/outpeer')


FLAGS = parser.parse_args()
