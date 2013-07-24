
INC := /usr/include/opencv
LIB := -lcv -lhighgui
FLAGS := -g

test: img_test.cpp img_lib.h 
	g++ img_test.cpp -o test -I$(INC) $(LIB) $(FLAGS)

