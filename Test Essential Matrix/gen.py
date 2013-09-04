

import random
import sys


if (len(sys.argv) < 2):
    print "debe ingresar el numero de puntos"
    sys.exit()

amount = int(sys.argv[1])

for x in xrange(0, amount):
    x = random.randint(-20, 5)
    y = random.randint(-20, 20)
    z = random.randint(3, 20)
    print "%d %d %d" % (x,y,z)
    
   
