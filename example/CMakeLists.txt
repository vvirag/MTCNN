set(EXAMPLE_FILES 
    main.cpp)

foreach(EXAMPLE_FILE ${EXAMPLE_FILES})

  get_filename_component(SOURCE_NAME ${EXAMPLE_FILE} NAME_WE)
  
  if (UNIX AND NOT APPLE)
    set(EXE_NAME "${SOURCE_NAME}.bin")
  elseif (WIN32)
    set(EXE_NAME "${SOURCE_NAME}")
  endif ()

  message(STATUS "Adding Example ${EXE_NAME}")
  add_executable(${EXE_NAME} ${EXAMPLE_FILE})
  target_link_libraries( ${EXE_NAME} mtcnn_cpp ${Caffe_LIBS} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})
  
endforeach()
