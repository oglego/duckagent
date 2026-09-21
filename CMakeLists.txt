cmake_minimum_required(VERSION 3.14)

# Set extension name here
set(TARGET_NAME duckagent)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD "17" CACHE STRING "C++ standard to enforce")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src/include)

#===--------------------------------------------------------------------===#
# agent.cpp - local LLM agent runtime (llama.cpp-backed), used to implement
# ai_enrich. Pulled in via FetchContent rather than as a git submodule so the
# extension repo stays self-contained; pin GIT_TAG to a specific commit once
# we've settled on a version to build against.
#===--------------------------------------------------------------------===#
include(FetchContent)

# Keep agent.cpp's own build minimal: we only need the `model` (and, later,
# `agent` for the tool-calling loop) targets, not its examples/tests/MCP/OAuth
# extras.
set(AGENT_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AGENT_CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AGENT_CPP_BUILD_MCP OFF CACHE BOOL "" FORCE)
set(AGENT_CPP_BUILD_OAUTH OFF CACHE BOOL "" FORCE)
set(AGENT_CPP_BUNDLED_LLAMA ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  agent_cpp
  GIT_REPOSITORY https://github.com/mozilla-ai/agent.cpp.git
  GIT_TAG        main # TODO: pin to a specific commit for reproducible builds
  GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(agent_cpp)

set(EXTENSION_SOURCES src/duckagent_extension.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link agent.cpp's `agent` target (pulls in `model` + llama.cpp transitively)
# into both the static library and the loadable extension.
target_link_libraries(${EXTENSION_NAME} agent-cpp::agent)
target_link_libraries(${LOADABLE_EXTENSION_NAME} agent-cpp::agent)
target_include_directories(${EXTENSION_NAME} PRIVATE duckdb/third_party/yyjson/include)
target_include_directories(${LOADABLE_EXTENSION_NAME} PRIVATE duckdb/third_party/yyjson/include)

install(
  TARGETS ${EXTENSION_NAME}
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
