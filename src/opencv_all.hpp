#ifndef __OPENCV_ALL__
#define __OPENCV_ALL__

// Project Configuration in MSVS
// - C/C++ > General > Additional Include Directories: EXTERNAL\OpenCV\include
// - Linker > Additional Library Directories: EXTERNAL\OpenCV\lib

#ifdef _WIN32
#   pragma warning(disable: 4819) // Diable warnings related to OpenCV's code page (949) problem
#endif

#define OPENCV_ENABLE_NONFREE
#define DISABLE_OPENCV_24_COMPATIBILITY

// Include basic modules
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/shape.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/superres.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videostab.hpp>
#include <opencv2/viz.hpp>

// Include extra (contrib) modules
#ifdef USE_OPENCV_CONTRIB
#   include <opencv2/opencv.hpp>
#   include <opencv2/aruco.hpp>
#   include <opencv2/bgsegm.hpp>
#   include <opencv2/bioinspired.hpp>
#   include <opencv2/ccalib.hpp>
#   include <opencv2/dpm.hpp>
#   include <opencv2/face.hpp>
#   include <opencv2/fuzzy.hpp>
#   include <opencv2/line_descriptor.hpp>
#   include <opencv2/optflow.hpp>
#   include <opencv2/phase_unwrapping.hpp>
#   include <opencv2/plot.hpp>
#   include <opencv2/rgbd.hpp>
#   include <opencv2/saliency.hpp>
#   include <opencv2/stereo.hpp>
#   include <opencv2/structured_light.hpp>
#   include <opencv2/surface_matching.hpp>
#   include <opencv2/text.hpp>
#   include <opencv2/tracking.hpp>
#   include <opencv2/xfeatures2d.hpp>
#   include <opencv2/ximgproc.hpp>
#   include <opencv2/xobjdetect.hpp>
#   include <opencv2/xphoto.hpp>
#endif

#ifdef _WIN32
#   ifdef _DEBUG
#       pragma comment(lib, "opencv_world320d.lib")
#   else
#       pragma comment(lib, "opencv_world320.lib")
#   endif // End of '_DEBUG'
#   pragma comment(lib, "opengl32.lib")
#   pragma comment(lib, "vfw32.lib")
#endif // End of '_WIN32'

#endif // End of '__OPENCV_ALL__'
