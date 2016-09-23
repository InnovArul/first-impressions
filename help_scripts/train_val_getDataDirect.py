
# coding: utf-8

import urllib2,os
import utils

currdir = os.path.dirname(os.path.abspath(__file__)) + '/'
datapath = os.path.dirname(os.path.abspath(__file__)) + '/../data/'
trainzipfolder = 'trainzip/'
validationzipfolder = 'validationzip/'
utils.mkdirs(datapath + trainzipfolder)
utils.mkdirs(datapath + validationzipfolder)

def getDataDirect(fsubset):
    with open(fsubset, 'r') as f:
        b = 1
        subset = ''
        lines = f.read().splitlines()
        for url in lines:
            file_name = ''
            if b >= 1 and b <= 75 and subset != 'validation':
                file_name = datapath + trainzipfolder
                subset = 'training'
            elif b >= 76:
                file_name = datapath + validationzipfolder
                subset = 'validation'
                b = 1
            print "Downloading Batch "+str(b) 
            print "From URL "+url
            if len(str(b)) > 1:
                file_name = file_name+subset+"80_"+str(b)+".zip"
            else:
                file_name = file_name+subset+"80_0"+str(b)+".zip"
                
            if os.path.isfile(file_name):
                print "File "+file_name+" already exists. Skipping ..."
            else:
                u = urllib2.urlopen(url)
                print file_name
                batchfile = open(file_name, 'wb')      
                meta = u.info()
                file_size = int(meta.getheaders("Content-Length")[0])
                print "Downloading: %s Bytes: %s" % (file_name, file_size)

                file_size_dl = 0
                block_sz = 8192
                while True:
                    buffer = u.read(block_sz)
                    if not buffer:
                        break

                    file_size_dl += len(buffer)
                    batchfile.write(buffer)
                    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                    status = status + chr(8)*(len(status)+1)
                    print status,

                batchfile.close() 
            b += 1        
    f.close()


def main():
    getDataDirect(currdir + 'train_val.txt')

if __name__ == "__main__":
    main()
