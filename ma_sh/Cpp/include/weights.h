#pragma once

#include <cmath>
#include <vector>

const std::vector<double> W0{
    0.5 * std::sqrt(1.0 / M_PI),
};

const std::vector<double> W1{
    0.5 * std::sqrt(1.5 / M_PI),
    0.5 * std::sqrt(1.5 / M_PI),
};

const std::vector<double> W2{
    0.25 * std::sqrt(5.0 / M_PI),
    0.5 * std::sqrt(7.5 / M_PI),
    0.25 * std::sqrt(7.5 / M_PI),
};

const std::vector<double> W3{
    0.25 * std::sqrt(7.0 / M_PI),
    0.125 * std::sqrt(42.0 / M_PI),
    0.25 * std::sqrt(105.0 / M_PI),
    0.125 * std::sqrt(70.0 / M_PI),
};

const std::vector<double> W4{
    0.1875 * std::sqrt(1.0 / M_PI),  0.375 * std::sqrt(10.0 / M_PI),
    0.375 * std::sqrt(5.0 / M_PI),   0.375 * std::sqrt(70.0 / M_PI),
    0.1875 * std::sqrt(35.0 / M_PI),
};

const std::vector<double> W5{
    1.0 / 16.0 * std::sqrt(11.0 / M_PI),
    1.0 / 16.0 * std::sqrt(165.0 / 2.0 / M_PI),
    1.0 / 8.0 * std::sqrt(1155.0 / 2.0 / M_PI),
    1.0 / 32.0 * std::sqrt(385.0 / M_PI),
    3.0 / 16.0 * std::sqrt(385.0 / 2.0 / M_PI),
    3.0 / 32.0 * std::sqrt(77.0 / M_PI),
};

const std::vector<double> W6{
    1.0 / 32.0 * std::sqrt(13.0 / M_PI),
    1.0 / 16.0 * std::sqrt(273.0 / 2.0 / M_PI),
    1.0 / 64.0 * std::sqrt(1365.0 / M_PI),
    1.0 / 32.0 * std::sqrt(1365.0 / M_PI),
    3.0 / 32.0 * std::sqrt(91.0 / 2.0 / M_PI),
    3.0 / 32.0 * std::sqrt(1001.0 / M_PI),
    1.0 / 64.0 * std::sqrt(3003.0 / M_PI),
};
