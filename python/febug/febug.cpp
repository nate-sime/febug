#include <iostream>
#include <pybind11/pybind11.h>

#include <febug/febug.h>

namespace py = pybind11;

namespace febug_wrappers
{
void febug(py::module& m){
  m.def("test", &febug::test);
}
} // namespace febug_wrappers

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "febug Python interface";
  // m.attr("__version__") = FEBUG_VERSION;

  // Create febug submodule
  py::module febug = m.def_submodule("feb", "General module");
  febug_wrappers::febug(febug);
}
