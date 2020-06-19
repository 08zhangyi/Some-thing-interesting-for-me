import sys
import os
sys.path.append('/home/RNA-seq/')
from pathvariables import tophat_dir, index_dir, cufflinks_dir

os.system('tophat -o ' + tophat_dir + ' ' + index_dir + 'sample.fastq')
lag_file = open('dummy.txt', 'w')
lag_file.write('tophat completed')
lag_file.close()
if os.path.exists('/home/RNA-seq/dummy.txt'):
    os.system('cufflinks -o ' + cufflinks_dir + ' '
              + tophat_dir + '/accepted_hits.bam')