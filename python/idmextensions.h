#pragma once

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include "tskit.h"

PyObject *idm_get_genomes(PyObject *args);
PyObject *idm_calculate_ibx(PyObject *args);
