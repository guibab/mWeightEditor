cmake_minimum_required(VERSION 3.2.2)
project(mWeightEditor)
SET(MODULE_NAME ${PROJECT_NAME} CACHE STRING "Name of the module")
install(DIRECTORY ${PROJECT_NAME} DESTINATION ${MODULE_NAME}/scripts)
