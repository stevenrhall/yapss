# Copyright (c) 2018 Centro de Estudos Aeronáuticos da UFMG
# Copyright (c) 2021-2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
#
# Derived from mseipopt (https://github.com/cea-ufmg/mseipopt), modified by
# the YAPSS authors. Original and modified portions are both under the MIT
# license; see the LICENSE file at the repository root.

"""The most simple ever IPOPT interface."""

from .bare import load_library, use_library

__all__ = ["load_library", "use_library"]
