#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/math.h>
#include <dolfinx/mesh/Mesh.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace febug_wrappers
{
void meshquality(py::module& m)
{

  m.def("dihedral_angle", [](dolfinx::mesh::Mesh& mesh)
  {
    mesh.topology().create_connectivity(3, 1);
    mesh.topology().create_connectivity(1, 0);
    const auto& c2e = mesh.topology().connectivity(3, 1);
    const auto& e2v = mesh.topology().connectivity(1, 0);

    const xtl::span<const double>& x_g = mesh.geometry().x();
    std::vector<std::int32_t> entity_list(
        mesh.topology().index_map(0)->size_local()
        + mesh.topology().index_map(0)->num_ghosts());
    std::iota(std::begin(entity_list), std::end(entity_list), 0);
    const xt::xtensor<std::int32_t, 2> e_to_g = entities_to_geometry(
        mesh, 0, entity_list, false);

    const std::int32_t num_cells = mesh.topology().index_map(3)->size_local();
    std::vector<double> dhangles(6*num_cells, 0.0);

    if (mesh.geometry().dim() != 3)
      throw std::runtime_error("Only supported for 3d meshes of tetrahedra");
    const std::size_t gdim = 3;

    xt::xtensor_fixed<double, xt::xshape<3>> p0;
    xt::xtensor_fixed<double, xt::xshape<3>> v1;
    xt::xtensor_fixed<double, xt::xshape<3>> v2;
    xt::xtensor_fixed<double, xt::xshape<3>> v3;

    auto copy_x_g = [&x_g, &gdim, &e_to_g](
        xt::xtensor_fixed<double, xt::xshape<3>>& v,
        const std::size_t idx)
    {
      const std::size_t pos = gdim * e_to_g(idx, 0);
      for (std::size_t i=0; i < gdim; ++i)
        v[i] = x_g[pos + i];
    };

    auto norm2 = [](const xt::xtensor_fixed<double, xt::xshape<3>>& v)
    { return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); };

    auto dot = [](const xt::xtensor_fixed<double, xt::xshape<3>>& u,
                  const xt::xtensor_fixed<double, xt::xshape<3>>& v)
    { return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]; };

    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      const auto& edges = c2e->links(c);

      for (std::int32_t i = 0; i < 6; ++i)
      {
        const auto i0 = e2v->links(edges[i])[0];
        const auto i1 = e2v->links(edges[i])[1];
        const auto i2 = e2v->links(edges[5-i])[0];
        const auto i3 = e2v->links(edges[5-i])[1];

        copy_x_g(p0, i0);
        copy_x_g(v1, i1);
        copy_x_g(v2, i2);
        copy_x_g(v3, i3);

        v1 -= p0;
        v2 -= p0;
        v3 -= p0;

        v1 /= norm2(v1);
        v2 /= norm2(v2);
        v3 /= norm2(v3);

        double cphi = (dot(v2, v3) - dot(v1, v2)*dot(v1, v3));
        cphi /= (norm2(dolfinx::math::cross(v1, v2)) * norm2(dolfinx::math::cross(v1, v3)));

        double dhangle = std::acos(cphi);
        dhangles[c*6 + i] = dhangle;
      }
    }
    return py::array_t<double>(dhangles.size(), dhangles.data());
  });
}
}
