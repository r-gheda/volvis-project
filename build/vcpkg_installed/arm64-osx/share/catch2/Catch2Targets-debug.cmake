#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Catch2::Catch2WithMain" for configuration "Debug"
set_property(TARGET Catch2::Catch2WithMain APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Catch2::Catch2WithMain PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/debug/lib/manual-link/libCatch2WithMain.a"
  )

list(APPEND _cmake_import_check_targets Catch2::Catch2WithMain )
list(APPEND _cmake_import_check_files_for_Catch2::Catch2WithMain "${_IMPORT_PREFIX}/debug/lib/manual-link/libCatch2WithMain.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
