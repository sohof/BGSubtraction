cmake_minimum_required(VERSION 3.2) # I have cmake version 3.22.1 installed.

PROJECT(Exjobb LANGUAGES CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.5.3_2/lib/cmake/opencv4)

# When using all the compile options we get a weird error, because sth in cuda opencv  
# maybe uses some c+11 feature which clang warns about and is config. to fail on.
# error: '_Atomic' is a C11 extension [-Werror,-Wc11-extensions]
# We can simply specifically deactivate this warning with the -Wno-c11-extensions 
# compiler flag

# ADDING COMPILER AND LINKER OPTIONS TO GET MORE WARNINGS AND EARLY FAILURES.
add_compile_options(-Wall -Wextra -pedantic -Werror  -Wno-c11-extensions -fsanitize=address,undefined)
add_link_options(-Wall -Wextra -pedantic -Werror -fsanitize=address,undefined)


# MAKE SURE OPENCV IS INSTALLED CORRECTLY. Each library prepares file/s which cmake uses to configure everything.
find_package(OpenCV REQUIRED)
message(${CMAKE_CXX_COMPILER_ID})
message(${CMAKE_CXX_COMPILER_VERSION})
message(${CMAKE_CXX_COMPILER})

# INCLUDE OPENCV DIRECTORIES. With homebrew includes are currently in /opt/homebrew/Cellar/opencv/4.5.3_2/include/opencv4
include_directories(${OpenCV_INCLUDE_DIRS}) # Has systemwide scope all target in all subdirectories
#target_include_directories(Exjobb "../include") # has target scope—it adds includ path to the include path for target specifix target.
#target_include_directories(Exjobb include)
add_subdirectory(mylib)


# MACRO TO COMPILE CPP FILES
MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} mylib)
  target_include_directories(${name} PUBLIC include)
ENDMACRO()

# COMPILE CPP FILES USING THIS LINE
######## EDIT THE FILE NAMES ########
add_example(ProcessVideo)
add_example(Main)
add_example(NN_Model)

