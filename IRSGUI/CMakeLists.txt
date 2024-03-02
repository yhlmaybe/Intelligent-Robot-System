
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

aux_source_directory(. IRSGUI_LIST)
add_library(IRSGUI_LIST ${IRSGUI_LIST})

target_link_libraries(IRSGUI_LIST Qt5::Core Qt5::Gui Qt5::Widgets)
  