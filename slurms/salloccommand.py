import os
import sys
os.system(f'salloc -A m4272 -C cpu -q debug -t 00:{sys.argv[1]}:00 -N 1')
