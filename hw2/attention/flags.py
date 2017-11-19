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


FLAGS = parser.parse_args()
