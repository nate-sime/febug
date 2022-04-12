#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace febug_wrappers
{
void meshquality(py::module& m);
} // namespace febug_wrappers

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "Febug Python interface";
//  m.attr("__version__") = "0.0.0";

  // Create mpc submodule [mpc]
  py::module meshquality = m.def_submodule(
    "meshquality", "mesh quality module");
  febug_wrappers::meshquality(meshquality);
}
