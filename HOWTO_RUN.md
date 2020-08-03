## How to Run Example Codes in Linux
### Preparation
* Install CMake, OpenCV, and Ceres Solver
* Download [example codes](https://github.com/sunglok/3dv_tutorial/archive/master.zip)

### Running Examples
1. Unzip example codes at `your_folder`
2. Build the example codes as like the following commands at `your_folder`
  ```bash
  mkdir build
  cd build
  cmake ..
  make install
  ```
3. Enjoy the examples at `your_folder/bin`

## How to Run Example Codes with Microsoft Visual Studio
### Preparation
* Install [Microsoft Visual Studio](https://www.visualstudio.com/) (shortly MSVS)
  * Visual Studio **Community** is _free_ for students, open-source, and individual developers.
  * We recommend the most recent version of MSVS (**at least >= 2015**) for [binary compatibility](https://docs.microsoft.com/ko-kr/cpp/porting/binary-compat-2015-2017).
    * If you want to use an older version of MSVS (<= 2013), please install [Microsoft Visual C++ Redistributable](https://support.microsoft.com/help/2977003/) for 2015-2019 (x64). Additionally please remember that you can only build examples in _release_ mode, not _debug_ mode.
* Download [example codes](https://github.com/sunglok/3dv_tutorial/archive/master.zip), [OpenCV binaries](https://github.com/sunglok/3dv_tutorial/releases/download/misc/OpenCV_v4.1.1_MSVS2017_x64.zip), and [Ceres Solver binaries](https://github.com/sunglok/3dv_tutorial/releases/download/misc/CeresSolver_v1.4.0_MSVS2019_x64.zip)

### Running Examples
1. Unzip example codes and binaries at `your_folder`
    * OpenCV and Ceres Solver will be located at `your_folder\EXTERNAL` folder. Please ignore duplicated `LICENSE` files.
2. Run your Visual Studio and open the solution file, `your_folder\msvs\3dv_tutorial.sln`
    * Or simply double-click `your_folder\msvs\3dv_tutorial.sln` file (if it is associated with MSVS)
3. Build the example codes in the solution (Menu > Build > Build Solution)
    * Their executable files will be located at `your_folder\bin` folder.
4. Enjoy the examples
