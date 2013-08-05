
INC := /usr/include/opencv
LIB := -lopencv_highgui -lopencv_core
FLAGS := -g

test: img_test.cpp img_lib.h 
	g++ img_test.cpp -o test -I$(INC) $(LIB) $(FLAGS)

