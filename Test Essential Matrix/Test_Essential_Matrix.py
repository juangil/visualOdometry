


import sys
from math import sqrt
from numpy import *

EPS = 1e-10

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
    
    
###static point intersectionbtwlines(point p1,point p2,point p3,point p4){
	###	point p2_p1=p2.sub(p1);
	###	point p4_p3=p4.sub(p3);
	###	double den=p2_p1.cross(p4_p3);
	###	if (Math.abs(den)<eps)//the lines are parallel or coincident
		###	return null;
	###	point p1_p3=p1.sub(p3);
	###	double num=p4_p3.cross(p1_p3);
	###	double ua=num/den;
	###	return p2_p1.multbyscalar(ua).add(p1);
###	}
	
def aux_inter(p1, p2, p3, p4):
    p2_p1 = [p2[0] - p1[0], p2[1] - p1[1]]
    p4_p3 = [p4[0] - p3[0], p4[1] - p3[1]]
    #print p2_p1, p4_p3
    den = p2_p1[0] * p4_p3[1] - p2_p1[1] * p4_p3[0]
    if(abs(den) < EPS):## the lines are parallel or coincident
        return None
    p1_p3 = [p1[0] - p3[0], p1[1] - p3[1]]
    num = p4_p3[0] * p1_p3[1] - p4_p3[1] * p1_p3[0]
    ua = num/den
    return ua
    
def get_intersection(p1, p2, p3, p4):
    pp1 = [p1[0], p1[1]]
    pp2 = [p2[0], p2[1]]
    pp3 = [p3[0], p3[1]]
    pp4 = [p4[0], p4[1]]
    intersection = aux_inter(pp1, pp2, pp3, pp4)
    if (intersection != None):
        return multbyscalar(res(p2,p1), intersection)
    pp1 = [p1[0], p1[2]]
    pp2 = [p2[0], p2[2]]
    pp3 = [p3[0], p3[2]]
    pp4 = [p4[0], p4[2]]
    intersection = aux_inter(pp1, pp2, pp3, pp4)
    if (intersection != None):
        return multbyscalar(res(p2,p1), intersection)
    pp1 = [p1[1], p1[2]]
    pp2 = [p2[1], p2[2]]
    pp3 = [p3[1], p3[2]]
    pp4 = [p4[1], p4[2]]
    intersection = aux_inter(pp1, pp2, pp3, pp4)
    if (intersection != None):
        return multbyscalar(res(p2,p1), intersection)
    return None
        
    

    
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


# de aca en adelante se asume que el vector +x esta en  +y  cross +z  ((1,0,0) para la camara 1)

    
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




if ( (abs(dot(plusZ, plusY))>EPS) or (abs(dot(plusZ, plusX))>EPS) or (abs(dot(plusY, plusX))>EPS) ):
    print "Los vectores base del sistema de coordenadas de la camara 2 no son ortogonales"
    
# projectando los puntos sobre la segunda camara.

projection_on_second_camera = []

for p in points:
    t = res(p, centerOfProjection)
    pp = [dot(t, plusX), dot(t, plusY), dot(t, plusZ)]
    if (pp[2] < 1.5):
        print "los puntos debe estar en frente de la camara 2"
        sys.exit()
    np = [pp[0] / pp[2], pp[1] / pp[2], 1]
    projection_on_second_camera.append(np)  
    
print "first:", projection_on_first_camera
print "second:", projection_on_second_camera



print "==="
def test_triangulate():
    array = []
    for i in range(0,8):
        proj1 = projection_on_first_camera[i]
        proj2 = projection_on_second_camera[i]
        p1 = [0,0,0]
        p2 = proj1
        p3 = centerOfProjection
        p4 = multbyscalar(plusX,proj2[0])
        p4 = sumv(p4, multbyscalar(plusY,proj2[1]))
        p4 = sumv(p4, multbyscalar(plusZ,proj2[2]))
        p4 = sumv(p4, centerOfProjection)
        #print p1,p2,p3,p4
        array.append(get_intersection(p1, p2, p3, p4))
    print "==="
    print array
    
            
A = []

for idx in xrange(0, len(points)):
    proj1 = projection_on_first_camera[idx]
    proj2 = projection_on_second_camera[idx]
    u = proj1[0]
    v = proj1[1]
    up = proj2[0]
    vp = proj2[1]
    a = [u * up, up * v, up, u * vp, v * vp, vp, u, v, 1]
    A.append(a)
  
print "===="  
    
print A
    
    
    
#test_triangulate()
#print get_intersection([0,0,0], [5,5,5], [-5,-5,5], [0, 0 , 0])




    




