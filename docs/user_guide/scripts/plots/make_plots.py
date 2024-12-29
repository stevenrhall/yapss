import runpy
import sys

import matplotlib.pyplot as plt_

# Ensure an argument was passed
if len(sys.argv) < 2:
    raise ValueError("Please provide the example name as an argument")

# Get the name of the example from the command line arguments
name = sys.argv[1]

# Set block=False before running the module
plt_.ion()  # Turns on interactive mode

# Run the specified module as a script
runpy.run_module(f"yapss.examples.{name}", run_name="__main__")

# Save any figures that were created
for i, figure in enumerate(plt_.get_fignums()):
    plt_.figure(figure)
    plt_.savefig(f"{name}_plot_{i+1}.png")

plt_.ioff()  # Turn off interactive mode
