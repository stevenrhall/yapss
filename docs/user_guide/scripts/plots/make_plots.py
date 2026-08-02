import runpy
import sys

import matplotlib

# This script only renders figures to PNG files -- it never needs a live window.
# Force the non-interactive Agg backend so runs are consistent across machines
# (e.g. a Mac with a GUI backend vs. a headless CI/RTD build), and so that
# plt.ion() doesn't trigger backend-specific redraw timing that can produce
# spurious "Ignoring fixed x/y limits to fulfill fixed data aspect" warnings.
matplotlib.use("Agg")

import matplotlib.pyplot as plt_

# Ensure an argument was passed
if len(sys.argv) < 2:
    raise ValueError("Please provide the example name as an argument")

# Get the name of the example from the command line arguments
name = sys.argv[1]

# Set block=False before running the module
plt_.ion()  # Turns on interactive mode

# Each example's main() ends with plt.show(), which is a no-op under Agg but
# still prints "FigureCanvasAgg is non-interactive, and thus cannot be shown".
# Silence it here rather than editing every example script.
plt_.show = lambda *args, **kwargs: None

# Run the specified module as a script
runpy.run_module(f"yapss.examples.{name}", run_name="__main__")

# Save any figures that were created
for i, figure in enumerate(plt_.get_fignums()):
    plt_.figure(figure)
    plt_.savefig(f"{name}_plot_{i+1}.png")

plt_.ioff()  # Turn off interactive mode
