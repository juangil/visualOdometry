


import sys
from math import sqrt
from numpy import *


def dot_product(a, b):
    ret = 0
    for x in range(0, 3):
        ret += a[x] * b[x]
    return ret
    
def cross_product(a, b):
    ret = [a[1] * b[2] - a[2] * b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
    return ret
    
def norm(a):
    return sqrt((a[0]*a[0]) + (a[1]*a[1]) + (a[2]*a[2]))
    
def sumv(a,b):
    ret = [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    return ret
    
def res(a,b):
    ret = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    return ret
    
def multbyscalar(a,b):
    ret = [a[0]*b, a[1]*b, a[2]*b]
    return ret
    
def normalize(vector): 
    return multbyscalar(vector, 1.0 / norm(vector))
    

    
    

argumentos = sys.argv

#se asume la distancia focal de 1

# se asume que el centro de proyeccion de la primera camara esta en (0,0) y que el image plane es z = 1

if (len(argumentos) < 3):
    print "pasar el archivo de los puntos y de las camara"
    sys.exit()
    
pointsFile = open(argumentos[1])
points = []

for l in pointsFile:
    p = l.split()
    p = map(float, p)
    points.append(p)

if (len(points) < 8):
    print "hay menos de 8 puntos en el dataset"
    sys.exit()
    
    
# projectando los puntos sobre la primera camara.
# centro de proyeccion = (0,0,0)
# vector del image plane = (0,0, 1)
# vector del +y = (0, 1, 0)


# de aca en adelante se asume que el vector +x esta en +z cross +y  ((1,0,0) para la camara 1)

    
projection_on_first_camera = []

for p in points:
    if (p[2] < 1.5):
        print "los puntos debe estar en frente de la camara 1"
        sys.exit()
    np = [p[0] / p[2], p[1] / p[2], 1]
    projection_on_first_camera.append(np)

projection_on_second_camera = []
    
#procesando la segunda camara

cameraFile = open(argumentos[2])

camera_parameters = cameraFile.readline().split()

# Formato del archivo de las camaras.  Centro = (x, y, z) , +z =(x, y, z) , +y = (x, y, z)

centerOfProjection = [float(camera_parameters[0]), float(camera_parameters[1]), float(camera_parameters[2])]
plusZ = [float(camera_parameters[3]), float(camera_parameters[4]), float(camera_parameters[5])]
plusY = [float(camera_parameters[6]), float(camera_parameters[7]), float(camera_parameters[8])]

plusX = cross_product(plusY, plusZ)

plusZ = normalize(plusZ)
plusY = normalize(plusY)
plusX = normalize(plusX)

EPS = 1e-10

if ( (dot(plusZ, plusY)>EPS) or (dot(plusZ, plusX)>EPS) or (dot(plusY, plusX)>EPS) ):
    print "Los vectores base del sistema de coordenadas de la camara 2 no son ortogonales"
    
# projectando los puntos sobre la segunda camara.

projection_on_second_camera = []

for p in points:
    t = res(p, centerOfProjection)
    pp = [dot(t, plusX), dot(t, plusY), dot(t, plusZ)]
    print pp
    if (pp[2] < 1.5):
        print "los puntos debe estar en frente de la camara 2"
        sys.exit()
    np = [pp[0] / pp[2], pp[1] / pp[2], 1]
    projection_on_second_camera.append(np)
    
    
print "============"    
    
print projection_on_second_camera







    




