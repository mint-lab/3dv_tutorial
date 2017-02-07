INCLUDE_DIRECTORIES(.)

FILE(GLOB hdrs_base "cvsba.h" )
FILE(GLOB srcs_base "*.c*")

SET(hdrs ${hdrs_base})
SET(srcs ${srcs_base})

# sba library source files
ADD_LIBRARY(${PROJECT_NAME} SHARED ${srcs} ${hdrs})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ) 
TARGET_LINK_LIBRARIES(${PROJECT_NAME}  ${REQUIRED_LIBRARIES})


INSTALL(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
	LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
        ARCHIVE DESTINATION lib COMPONENT main)			# Install the dll.a file in lib directory

INSTALL(FILES ${hdrs_base}
        DESTINATION include/${PROJECT_NAME}
        COMPONENT main)