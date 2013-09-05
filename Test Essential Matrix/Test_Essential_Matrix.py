


import sys
from math import sqrt
from numpy import * 
from math import cos
from math import sin
from math import pi


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
    
    
def triple_product(a, b, c):
    return dot(a, cross(b, c))
    
    
def all_coplanar(array):
    """ under construction, not ready"""
    return None
    
def get_rotation_matrix(x, y , z):
    matriz_x = matrix([ [1,0,0] , [0, cos(x), -sin(x)], [0, sin(x), cos(x)] ])
    matriz_y = matrix([ [cos(y),0,sin(y)] , [0, 1, 0], [ -sin(y), 0, cos(y)] ])
    matriz_z = matrix([ [cos(z),-sin(z),0] , [sin(z), cos(z), 0], [0, 0, 1] ])
    return matriz_x * matriz_y * matriz_z
    
    
def get_skew_matrix(tx, ty, tz):
    return matrix([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    
    
def is_skew_matrix(T):
    if (abs(T[0,0] > EPS) or abs(T[1,1] > EPS) or abs(T[2,2] > EPS)):
        return False
    for i in xrange(3):
        for j  in xrange(i + 1, 3):
            if ( abs(T[i,j] + T[j, i]) > EPS):
                return False
    return True
    
## UTIL

def Util_Print(B):
    if (len(B.shape) == 2):
        for x in xrange(0, B.shape[0]):
            for y in xrange(0, B.shape[1]):
                print "%.5f" % B[x,y],
            print
    else:
        for x in xrange(0, B.shape[0]):
            print "%.5f" % B[x],
        print
    
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
    
if (all_coplanar(points)):
    print "todos los puntos son coplanares"
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
    #print "==="
    #print array
            
A = []

for idx in xrange(0, len(points)):
    proj1 = projection_on_first_camera[idx]
    proj2 = projection_on_second_camera[idx]
    u = proj1[0]
    v = proj1[1]
    up = proj2[0]
    vp = proj2[1]
    a = [u * up, u * vp, u, up * v, v * vp, v, up, vp, 1]
    A.append(a)
  
"""
for row in A:
    for col in row:
        print col,
    print "\n"
"""   
  
Amatrix = matrix(A)

#print A

U,s,V = linalg.svd(A)

rank_A = 0
for singular_value in s:
    if (singular_value > EPS):
        rank_A += 1

print "====== A (Estimada)======="
print "El rank de la matriz A es %d" % rank_A
print "A = "
Util_Print(Amatrix)
print "U="
Util_Print(U)
print "S="
Util_Print(s)
print "V="
Util_Print(V)
print "======= End A =========="

#nS = matrix(zeros(shape=(8,9)))

#for x in xrange(0, s.shape[0]):
#    nS[x,x] = s[x]
#    
#prueba = (U * nS) * V

#print "A="
#Util_Print(prueba)

sol = matrix(zeros([9,1]))

for idx in xrange(0, 9):
    sol[idx, 0] = V[8, idx]
    
#print A * sol # aca se verifica que sol efectivamente sea una solucion AX = 0

tmp = [[sol[0,0], sol[1,0], sol[2,0]], [sol[3,0], sol[4,0], sol[5,0]], [sol[6,0], sol[7,0], sol[8,0]]]


Essentialmatrix = matrix(tmp)




def TestEssentialMatrix(E):
    print "Test: Epipolar constraint Test"
    for idx in xrange(len(points)):
        x = matrix(projection_on_first_camera[idx])
        xp = matrix(projection_on_second_camera[idx])
        xp = xp.transpose()
        print x * (E * xp)


#TestEssentialMatrix(Essentialmatrix) # Aca se prueba que la matriz esencial efectivamente satisfaga la restriccion epipolar


U,s,V = linalg.svd(Essentialmatrix)
print "====== Essential (Estimada)======="
print "E = "
Util_Print(Essentialmatrix)
print "U="
Util_Print(U)
print "S="
Util_Print(s)
print "V="
Util_Print(V)
print "======= End Essential =========="


def Test_Epipolar_with_Rotation_and_Traslation():
    print "Test: from Rotation and traslation: OK" 
    Rotation = get_rotation_matrix(pi/2 , -pi/2, 0)
    T = [10, 0, 8]
    for idx in xrange(0, len(points)):
        x = projection_on_first_camera[idx]
        xp = matrix(projection_on_second_camera[idx])
        xp = xp.transpose()
        Rx = Rotation * xp
        Rx = [Rx[0,0], Rx[1,0], Rx[2,0]]
        tmp = cross_product(T, Rx)
        print dot(x, tmp)
        
def Test_coplanar_one_coordinate_system():
    print "Test: Coplanar one coordinate system: OK"
    T = [10, 0, 8]
    for i in xrange(0, len(points)):
        proj1 = projection_on_first_camera[i]
        proj2 = projection_on_second_camera[i]
        p4 = multbyscalar(plusX,proj2[0])
        p4 = sumv(p4, multbyscalar(plusY,proj2[1]))
        p4 = sumv(p4, multbyscalar(plusZ,proj2[2]))
        v = p4
        print dot(T, cross(v, proj1))
        

R = get_rotation_matrix(pi/2 , -pi/2, 0)
t = get_skew_matrix(10, 0, 8)
Essential_no_estimada = t * R


#Test_Epipolar_with_Rotation_and_Traslation() # Test OK
#Test_coplanar_one_coordinate_system() # Test OK
#TestEssentialMatrix(Essentialmatrix) # Test OK
#TestEssentialMatrix(Essential_no_estimada) # Test OK

print "E = (No estimada)"
print Essential_no_estimada
print "E = (estimada)"
print Essentialmatrix


"""
U,s,V = linalg.svd(Essential_no_estimada)

print "U="
Util_Print(U)
print "S="
Util_Print(s)
print "V="
Util_Print(V)
"""

def Debug(cosa, verbose):
    if (verbose):
        print cosa

def Possible_Solutions(E, verbose = False):
    """ se asume que la matriz E, es una matriz esencial valida, rank 2 """
    Debug("enumerando las posibilidades", verbose)
    U,s,VT = linalg.svd(E)
    S = matrix(diag([s[0],s[1],s[2]]))
    W = matrix([[0,1,0],[-1,0,0],[0,0,1]])
    Rot1 = U * W * VT
    Rot2 = U * W.transpose() * VT
    T1skew = U * W * S * U.transpose()
    if (not(is_skew_matrix(T1skew))):
        print "La matriz de Traslacion deberia ser skew y no lo es"
        return
    T1 = matrix([T1skew[2,1], T1skew[0,2], T1skew[1,0]]).H
    T2 = -1*T1
    Debug("Rotacion 1", verbose)
    Debug(Rot1, verbose)
    Debug("Rotacion 2", verbose)
    Debug(Rot2, verbose)
    Debug("Traslacion 1", verbose)
    T1 = (1 / abs(T1[0,0])) * T1  # normalizacion de la traslacion de esta forma (1, x, y)
    Debug(T1, verbose)
    Debug("Traslacion 2", verbose)
    T2 = (1 / abs(T2[0,0])) * T2  # normalizacion de la traslacion de esta forma (1, x, y)
    Debug(T2, verbose)
    return [[Rot1, T1], [Rot1, T2], [Rot2, T1], [Rot2, T2]]
    
    
def 
    
def Disambiguate(solutions):
    """ dada la condicion ideal de este experimento se va a triangular solamente la primera correspondencia
        en un real-setting debe ser un punto o muchos puntos aleatorios"""
    proj1 = projection_on_first_camera[0]
    proj2 = projection_on_second_camera[0]
    for s in xrange(0, len(solutions)):
        R = solutions[s][0]
        t = solutions[s][1]
        
    
    
    
solutions = Possible_Solutions(Essentialmatrix)

print solutions
    

    




