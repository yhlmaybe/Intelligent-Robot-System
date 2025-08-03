aux_source_directory(. BRAINDEEPLEARN_LIST)
add_library(BRAINDEEPLEARN_LIST ${BRAINDEEPLEARN_LIST})

target_link_libraries(BRAINDEEPLEARN_LIST PUBLIC python3.8)
target_include_directories(BRAINDEEPLEARN_LIST PUBLIC ${Python_INCLUDE_DIRS})
