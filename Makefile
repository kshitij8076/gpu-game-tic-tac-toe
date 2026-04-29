CXX = nvcc
CXXFLAGS = --std c++17 -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -lcuda

.PHONY: clean build run

build:
	$(CXX) $(CXXFLAGS) src/tictactoe.cu -o tictactoe.exe

clean:
	rm -f tictactoe.exe

run:
	./tictactoe.exe

all: clean build run
