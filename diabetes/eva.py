########################################################
# Description: the main process to control the evaluations 
# Author: Jamie Zhu <jimzhu@GitHub>
# License: MIT
# Last updated: 2016/05/04
########################################################

import numpy as np
import time
import multiprocessing
from  main import *

# #======================================================#
# # Function to evalute the approach at all settings
# #======================================================#
# def execute(matrix, parallelMode=False):
#     # loop over each density and each round
#     if parallelMode: # run on multiple processes
#         pool = multiprocessing.Pool()
#         for den in para['density']: 
#             for roundId in xrange(para['rounds']):
#                 pool.apply_async(executeOneSetting, (matrix, den, roundId, para))
#         pool.close()
#         pool.join()
#     else: # run on single processes
#         for den in para['density']:
#             for roundId in xrange(para['rounds']):
#                 executeOneSetting(matrix, den, roundId, para)
#     # summarize the dumped results
#     evallib.summarizeResult(para)

# def hello(label):
#     for v in range(10):
#         time.sleep(1)
#         print label,v

if __name__=="__main__":
    pool = multiprocessing.Pool(processes=3)
    pca_size=[9,11,13,15]
    # print layers
    for pca in pca_size: 
        # print "pca_size%d"%n
        pool.apply_async(execute, (pca,(20,40)))

    # layers=[(10,10),(20,20),(30,30),(40,40),(50,50)]
    # layers=[(50,10),(40,20),(30,30),(20,40),(10,50)]
    # # print layers
    # for layer in layers: 
    #     # print "pca_size%d"%n
    #     pool.apply_async(execute, (7,layer))
    # alphas=[1e-5,1e-6,1e-7]
    # for alpha in alphas: 
    #     # print "pca_size%d"%n
    #     pool.apply_async(execute, (7,(10,50),alpha))

    pool.close()
    pool.join()
    print "over"