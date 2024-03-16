#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(pointnet2_ops, m) {
  m.doc() = "pybind11 pointnet2 ops plugin";

  m.def("gather_points", &gather_points, "pointnet2_ops.gather_points");
  m.def("gather_points_grad", &gather_points_grad,
        "pointnet2_ops.gather_points_grad");
  m.def("furthest_point_sampling", &furthest_point_sampling,
        "pointnet2_ops.furthest_point_sampling");

  m.def("three_nn", &three_nn, "pointnet2_ops.three_nn");
  m.def("three_interpolate", &three_interpolate,
        "pointnet2_ops.three_interpolate");
  m.def("three_interpolate_grad", &three_interpolate_grad,
        "pointnet2_ops.three_interpolate_grad");

  m.def("ball_query", &ball_query, "pointnet2_ops.ball_query");

  m.def("group_points", &group_points, "pointnet2_ops.group_points");
  m.def("group_points_grad", &group_points_grad,
        "pointnet2_ops.group_points_grad");
};
