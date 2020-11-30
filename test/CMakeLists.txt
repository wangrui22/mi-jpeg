aux_source_directory(./ SRC)

add_executable(test ${SRC})

target_link_libraries(test jpeg)
