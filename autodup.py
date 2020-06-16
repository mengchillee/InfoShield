#!/usr/bin/env python3
# Author: Catalina Vajiac
# Purpose:
# Usage:

import os, sys

from autodupcoarse import *
from autodupfine import *


def usage(code):
    print('Usage: {} [filename]'.format(os.path.basename(sys.argv[0])))
    exit(code)


if __name__ == '__main__':
    if len(sys.argv) not in [2, 4]:
        usage(1)

    coarse = AutoDupCoarse(*sys.argv[1:])
    coarse.clustering()
    print(coarse.final_data_filename)
    run_autodupfine(coarse.final_data_filename)
