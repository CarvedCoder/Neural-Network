cmake_minimum_required(VERSION 3.10)
project(Neural_Network)

set(CMAKE_CXX_STANDARD 17)



# Create executable
add_executable(NeuralNetwork main.c)

include_directories(/usr/local/include)

if(MSVC)
    target_compile_options(NeuralNetwork PRIVATE /W4 /WX)
else()
    target_compile_options(NeuralNetwork PRIVATE -Wall -Wextra -Werror)
endif()
