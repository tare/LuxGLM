#!/usr/bin/env python

import argparse
import sys
import os

import logging
from hashlib import md5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LuxRep')
    parser.add_argument('-d', '--data', action='store', dest='data_file', type=str, required=True, help='file containing noncontrol cytosine data')
    parser.add_argument('-s', '--split_size', action='store', dest='split_size', type=int, required=True, help='number of cytosine in split table')
    parser.add_argument('-o', '--outfolder', action='store', dest='outfolder', type=str, required=False, default='%s/results'%os.getcwd(), help='directory containing split tables with full pathname')
    parser.add_argument('-v','--version',action='version',version='%(prog)s 0.666')
    options = parser.parse_args()

    if not os.path.exists(options.outfolder): os.makedirs(options.outfolder)

    size = options.split_size

    lines = open(options.data_file,'r').readlines()
    header = lines[0].strip()
    chrom, flag = lines[1].split(':')[0], 0
    counter, chunk = 0, 0

    for line in lines[1:]:
        counter += 1
        if chrom != line.split(':')[0] and counter != 1:
            fo.close() 
            counter = 1
            chunk += 1 
            outfile = '%s/counts_%s.tab'%(options.outfolder,chunk)
            fo = open(outfile,'w')
            print >> fo, header
            print >> fo, line.strip()
            chrom = line.split(':')[0]
            continue
        if counter == 1:
            chunk += 1 
            outfile = '%s/counts_%s.tab'%(options.outfolder,chunk)
            fo = open(outfile,'w')
            print >> fo, header
        print >> fo, line.strip()
        if counter == size:
            fo.close() 
            counter = 0

    if counter < size: fo.close()
    print 'split %s into %s tables (%s loci) in %s'%(options.data_file,chunk,size,options.outfolder)
    
