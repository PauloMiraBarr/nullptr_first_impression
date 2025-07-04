cmake_minimum_required(VERSION 3.29)
project(project_nullptr_0)

set(CMAKE_CXX_STANDARD 20)

add_executable(main main.cpp
                    TextLoader.cpp)
# agregar todos los cpp de ser preciso :P


# Casos de prueba unitarios (falta agregar cach2)
add_executable(TextLoaderApp TextLoaderTest.cpp TextLoader.cpp)
