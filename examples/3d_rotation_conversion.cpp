#include <iostream>
#include "Eigen/Dense"

int main() {
    // The given 3D rotation
    Eigen::Vector3d euler_degrees(45, 30, 60);  // Unit: [deg] in the XYZ-order (column vector)
    Eigen::Vector3d euler_radians = euler_degrees * M_PI / 180.0; // Unit: radian

    // Rotation matrix converted from euler angles
    Eigen::Matrix3d rotation_matrix = Eigen::AngleAxisd(euler_radians.x(), Eigen::Vector3d::UnitX())
                                      * Eigen::AngleAxisd(euler_radians.y(), Eigen::Vector3d::UnitY())
                                      * Eigen::AngleAxisd(euler_radians.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix();

    // Rotation vector (rodrigues' rotation)
    Eigen::AngleAxisd rotation_vector(rotation_matrix);

    // Quaternion
    Eigen::Quaterniond quaternion(rotation_vector);

    std::cout << "\n## Euler Angle (XYZ)\n" << euler_degrees.transpose() << std::endl;
    std::cout << "\n## Rotation Matrix\n" << rotation_matrix << std::endl;
    std::cout << "\n## Rotation Vector\n" << rotation_vector.angle() * rotation_vector.axis().transpose() << std::endl;
    std::cout << "\n## Quaternion\n" << quaternion.coeffs().transpose() << std::endl;

    return 0;
}