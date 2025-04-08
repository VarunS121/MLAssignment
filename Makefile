# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Source files
SRCS = main.cpp image.cpp logreg.cpp cnn.cpp utils.cpp
OBJS = $(SRCS:.cpp=.o)

# Output binary
TARGET = cnn_vs_logreg

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f *.o $(TARGET)
