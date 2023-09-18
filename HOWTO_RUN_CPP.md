## How to Run Example Codes with Microsoft Visual Studio
### Prerequisites
* Install [Microsoft Visual Studio](https://www.visualstudio.com/) (shortly MSVS)
  * :memo: Note) Microsoft Visual Studio **Community** is _free_ for students, open-source contributors, and individual developers.
* Install [Git](https://git-scm.com/) and  [CMake](https://cmake.org/)
  * You need to [add their directory to PATH environment in Windows](https://stackoverflow.com/questions/44272416/how-to-add-a-folder-to-path-environment-variable-in-windows-10-with-screensho).
* Install [vcpkg](https://vcpkg.io/) ([more details](https://vcpkg.io/en/getting-started))
  * You need to move your working directory where you want to install _vcpkg_ (I assume your working directory as `C:/`).
  ```bash
  git clone https://github.com/Microsoft/vcpkg.git
  .\vcpkg\bootstrap-vcpkg.bat
  ```
* Install [OpenCV](https://opencv.org/) and [Ceres Solver](http://ceres-solver.org/) using _vcpkg_
  ```bash
  cd vcpkg
  vcpkg install opencv[world,contrib]:x64-windows --recurse
  vcpkg install ceres[suitesparse,cxsparse,eigensparse,tools]:x64-windows
  vcpkg integrate install
  ```

### Compiling and Running Examples
1. Clone the repository: `git clone https://github.com/mint-lab/3dv_tutorial.git`
    * You need to move your working directory where you want to copy the repository.
    * Or you can download [example codes and slides as a ZIP file](https://github.com/sunglok/3dv_tutorial/archive/master.zip) and unzip it where you want.
1. Generate MSVS solution and project files
    * You need to specify the _vcpkg_ directory where you installed it. (I assume you install it at `C:/`)
    ```bash
    cd 3dv_tutorial
    mkdir build
    cd build
    cmake .. "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
    ```
1. Run your MSVS and open the generated solution file, `build/3dv_tutorial.sln`.
1. Build the example codes in the solution (Menu > `Build` > `Build Solution`)
1. Run the examples using `F5` after specify your target as the startup project (Project Context Menu > `Set as Startup Project`)



## How to Run Example Codes in Linux
### Prerequisites
* Install [GCC](https://gcc.gnu.org/), [Git](https://git-scm.com/), [CMake](https://cmake.org/), [OpenCV](https://opencv.org/), and [Ceres Solver](http://ceres-solver.org/)

### Running Examples
1. Clone the repository: `git clone https://github.com/mint-lab/3dv_tutorial.git`
    * You need to move your working directory where you want to copy the repository.
    * Or you can download [example codes and slides as a ZIP file](https://github.com/sunglok/3dv_tutorial/archive/master.zip) and unzip it where you want.
1. Generate `Makefile` and project files
    ```bash
    cd 3dv_tutorial
    mkdir build
    cd build
    cmake ..
    make install
    ```
1. Run the examples