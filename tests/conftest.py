import warnings

import matplotlib as mpl
import pytest


@pytest.fixture(autouse=True)
def block_mpl_backend():
    """Set the matplotlib backend to 'Agg' to prevent plots from displaying during tests."""
    mpl.use("Agg")
    mpl.rcParams["text.usetex"] = False

    # Suppress the specific UserWarning about FigureCanvasAgg being non-interactive
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*FigureCanvasAgg is non-interactive, and thus cannot be shown.*",
    )
