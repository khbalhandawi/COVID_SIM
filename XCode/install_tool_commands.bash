otool -L /Users/nyuad/Desktop/Khalil_repos/COVID_SIM/build/Release/COVID_SIM_UI.app/Contents/MacOS/COVID_SIM_UI
install_name_tool -change /Users/nyuad/Desktop/Khalil_repos/COVID_sim_app/libs/libCOVID_SIM.dylib @executable_path/../Frameworks/libCOVID_SIM.dylib /Users/nyuad/Desktop/Khalil_repos/build_Qt/COVID_SIM_UI.app/Contents/MacOS/COVID_SIM_UI

cd /Users/nyuad/Qt/5.15.2/clang_64/bin
./macdeployqt /Users/nyuad/Desktop/Khalil_repos/COVID_SIM/build/Release/COVID_SIM_UI.app