"""
solve() works off the main thread with the default catch_keyboard_interrupt.

signal.signal is permitted only on the main thread. Through 0.2.2 the SIGINT handler was
installed unconditionally whenever catch_keyboard_interrupt was true (the default), so a
solve from a worker thread -- a GUI, a notebook background cell, a thread pool -- raised
"signal only works in main thread" after all NLP setup, with no mention of the flag.
"""

import threading

from yapss._legacy.examples import brachistochrone_minimal


def test_solve_from_worker_thread_with_default_interrupt_flag():
    result: dict = {}

    def work() -> None:
        problem = brachistochrone_minimal.setup()
        assert problem.catch_keyboard_interrupt  # the default that used to fail
        problem.ipopt_options.print_level = 0
        try:
            result["solution"] = problem.solve()
        except BaseException as exc:  # noqa: BLE001 -- report it on the main thread
            result["error"] = exc

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()

    assert "error" not in result, result.get("error")
    assert result["solution"].nlp_info.ipopt_status == 0
