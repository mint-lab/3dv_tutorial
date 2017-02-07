# CMake file for sba's demo program
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)
LINK_LIBRARIES(${PROJECT_NAME})

# ADD_EXECUTABLE(eucsbademo eucsbademo.c imgproj.c readparams.c eucsbademo.h readparams.h)
ADD_EXECUTABLE(test_cvsba test_cvsba.cpp readparams.c)
ADD_EXECUTABLE(generate_synthetic_data generate_synthetic_data.cpp)

# libraries the demo depends on
IF(HAVE_F2C)
  SET(UTILS_REQUIRED_LIBRARIES ${PROJECT_NAME} ${LAPACK_LIB} ${BLAS_LIB} ${F2C_LIB})
ELSE(HAVE_F2C)
  SET(UTILS_REQUIRED_LIBRARIES ${PROJECT_NAME} ${LAPACK_LIB} ${BLAS_LIB} ${F77_LIB} ${I77_LIB})
ENDIF(HAVE_F2C)

# TARGET_LINK_LIBRARIES(eucsbademo  ${UTILS_REQUIRED_LIBRARIES})
TARGET_LINK_LIBRARIES(test_cvsba  ${UTILS_REQUIRED_LIBRARIES} ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES(generate_synthetic_data  ${UTILS_REQUIRED_LIBRARIES} ${OpenCV_LIBS} )


# make sure that the library is built before the demo
# ADD_DEPENDENCIES(eucsbademo sba)
ADD_DEPENDENCIES(test_cvsba sba)
