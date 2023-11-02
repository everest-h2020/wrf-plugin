#pragma once

// NOTE: Currently, I have no idea how the data type of the compiled artifact is
//       decided by the build system, and I'm too lazy to find out. Instead, we
//       hard-code it (to a guess, and change it at most once).

/// The default Fortran integer type.
using INTEGER = int;
/// The default Fortran real type.
using REAL = float;
