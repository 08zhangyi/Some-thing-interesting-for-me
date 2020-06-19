from tcoffeevariables import tcoffeeout
import sys
import os

sys.path.append('pathmodules/')
cmd = 't_coffee -in="file.fasta" 杛un_name="' + tcoffeeout + 'tcoffe.aln" -output=clustalw'
os.system(cmd)
