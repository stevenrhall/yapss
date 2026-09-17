Project Status
==============

YAPSS is in initial development (0.x). The numerics are the settled part of the project;
the interface is not.

Stable
------

The transcription itself --- Legendre-Gauss, Legendre-Gauss-Radau, and
Legendre-Gauss-Lobatto collocation, multi-phase problems, segmented meshes, the four
derivative methods, and the Ipopt backend --- has been in use across the 0.2 and 0.3
series, and is not the subject of the redesign below. Problems that solve today will
still be solvable, and will give the same answers.

Changing
--------

Version 0.3.0 is expected to be the last release of the current API. A redesign is in
progress, covering how a problem is declared, how callbacks are written, and how a solution
is read. If it lands as planned, it will arrive in the next minor version and will not be
source-compatible: code written against 0.3 would be rewritten against the new API rather
than adjusted.

What this means for you
-----------------------

Pin to the minor version you developed against::

    yapss>=0.3,<0.4

Under 0.x versioning a minor-version bump may change the API in any case, so this is the
right pin whether or not the redesign lands on the schedule above. If you are starting
work now that you expect to outlive the redesign, the pin is the thing to get right: the
migration will be a deliberate piece of work rather than an upgrade.
