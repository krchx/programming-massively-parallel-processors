CXX = hipcc
CXXFLAGS = -I../../lib_/PngHandlib
LDFLAGS = -lpng

SRC = main.cpp ../../lib_/PngHandlib/PNGHand.cpp
OBJ = $(SRC:.cpp=.o)
EXE = blur_image

all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(OBJ) -o $(EXE) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXE)