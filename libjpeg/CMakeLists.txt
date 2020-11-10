aux_source_directory(./ SRC)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(libjpeg_sample ${SRC})

target_link_libraries(libjpeg_sample jpeg)
