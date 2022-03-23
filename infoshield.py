#!/usr/bin/env python3
# Author: Catalina Vajiac
# Purpose:
# Usage:

import os, sys

from infoshieldcoarse import *
from infoshieldfine import *


def usage(code):
	print('Usage: {} [filename]'.format(os.path.basename(sys.argv[0])))
	exit(code)


if __name__ == '__main__':
	if len(sys.argv) not in [2, 4]:
		usage(1)

	coarse = InfoShieldCoarse(*sys.argv[1:])
	coarse.clustering()
	print(coarse.final_data_filename)
	run_infoshieldfine(coarse.final_data_filename, *sys.argv[2:])
