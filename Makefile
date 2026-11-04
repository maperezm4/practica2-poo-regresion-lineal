CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
OBJS     = MatrixOps.o LinearRegression.o CSVReader.o

all: regresion memoria

regresion: $(OBJS) main.o
	$(CXX) $(CXXFLAGS) -o regresion $(OBJS) main.o

memoria: $(OBJS) memory_analysis.o
	$(CXX) $(CXXFLAGS) -o memoria $(OBJS) memory_analysis.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: all
	./regresion
	./memoria

clean:
	rm -f *.o regresion memoria

.PHONY: all run clean
