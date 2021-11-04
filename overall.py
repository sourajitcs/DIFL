import subprocess
import sys
'''
directory = sys.argv[1]

while True:
    training = subprocess.run(['python3', 'cont_0_noninput.py', directory])
    training = subprocess.run(['python3', 'testing_0_noninput.py', directory])
'''

num = 4
while num < 10:
    training = subprocess.run(['python3', '0_difl.py', '../datasets/us', '../datasets/china', '10000'])
    num += 1

num = 4
while num < 10:
    training = subprocess.run(['python3', '0_difl.py', '../datasets/us', '../datasets/india', '10000'])
    num += 1


