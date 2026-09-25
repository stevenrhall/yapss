"""

Reducing a problem to what the transcription reads.

The snapshot taken when a solve begins still knows about declarations, named fields, and the
user's per-phase callbacks. The transcription knows about none of that: it is given counts,
arrays, and callbacks, as `yapss._backend.spec.ProblemSpec`. This module makes one from the
other.

The callbacks it supplies are adapters: they receive what the transcription passes, build the
argument and output objects of this API, call the user's own per-phase callback, check what
came back, and hand the rows on. They are what remains of the bridge, and they will go when the
transcription calls the new callbacks directly.

"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from yapss._backend.callbacks import UserFunctions
from yapss._backend.input_args import callback_location, note_callback_error
from yapss._backend.solver import solve
from yapss._backend.spec import PhaseSpec, ProblemSpec, frozen_array

from .args import (
    ContinuousArg,
    ContinuousOut,
    DiscreteOut,
    Endpoint,
    EndpointArg,
    EndpointValues,
    phase_arg_class,
)
from .kinds import ReadOnlyRows, Rows
from .solution import Solution
from .vector import Maker

if TYPE_CHECKING:
    from collections.abc import Callable

    from .spec import PhaseSpec as PhaseSpec_
    from .spec import ProblemSpec as ProblemSpec_
    from .vector import Vector

__all__ = ["solve_problem", "to_transcription_spec"]


def _npoints(time: Any) -> int | None:
    """Return the number of time points, or None when a symbolic trace makes that meaningless."""
    try:
        return len(time)
    except TypeError:
        return None


def _check_return(result: Any, expected: Any, callback: Callable[..., Any], what: str) -> None:
    """Refuse anything but the output object or None.

    A callback that fills `out` and simply ends is correct, because the handler reads `out`
    either way; what is refused is a *different* value, which is nearly always the 0.3.0 habit
    of returning the rows.
    """
    if result is None or result is expected:
        return
    name = getattr(callback, "__qualname__", repr(callback))
    msg = (
        f"the {what}, '{name}', returned {result!r}. Fill 'out' and either return it or "
        f"return nothing."
    )
    raise TypeError(msg)


def _call(callback: Callable[..., Any], what: str, arg: Any, *args: Any) -> Any:
    """Call the user's callback, noting an exception it raises with the callback's name.

    The note is the one `call_callback` gives a callback it calls directly. It cannot give it
    here: it calls this module's adapter, which calls the user's function, and only the adapter
    knows which phase's callback that was. `arg` is the transcription's argument, which says
    whether this is the symbolic trace.
    """
    try:
        return callback(*args)
    except Exception as exc:
        note_callback_error(
            exc,
            f"Raised in the {what}: {callback_location(callback)}.",
            symbolic=arg._dtype is np.object_,
        )
        raise


def _check_complete(output: Any, callback: Callable[..., Any], what: str) -> None:
    """Refuse an output with any field left unassigned."""
    if output._is_complete():
        return
    name = getattr(callback, "__qualname__", repr(callback))
    msg = f"the {what}, '{name}', returned without assigning {', '.join(output._missing())}"
    raise ValueError(msg)


class _PhaseMakers:
    """What one phase's continuous call needs that does not change between calls."""

    _arg: ContinuousArg | None

    __slots__ = (
        "_arg",
        "arg_class",
        "callback",
        "control",
        "dynamics",
        "handle",
        "has_integral",
        "has_path",
        "has_state",
        "integrand",
        "parameter",
        "path",
        "state",
        "what",
    )

    def arg(self, data: Any, parameter: Any) -> ContinuousArg:
        """Return the continuous argument for one evaluation.

        The inputs are read-only vectors that read their rows where they are, so the argument
        is built once and pointed at each new evaluation's arrays rather than rebuilt. Only the
        inputs are reused: an output must start empty, because that is how "every declared field
        was assigned" is decided.
        """
        cached = self._arg
        if cached is None:
            cached = self.arg_class(
                self.handle,
                data.time,
                self.state.over(data.state),
                self.control.over(data.control),
                self.parameter.over(parameter),
            )
            self._arg = cached
            return cached
        setattr_ = object.__setattr__
        setattr_(cached, "_points", data.time)
        setattr_(cached.state, "_source", data.state)
        setattr_(cached.control, "_source", data.control)
        setattr_(cached.parameter, "_source", parameter)
        return cached

    def __init__(self, spec: ProblemSpec_, phase: PhaseSpec_) -> None:
        label = f"phase '{phase.name}'"
        self.handle = phase.handle
        self.arg_class = phase_arg_class(phase.independent)
        self.callback = phase.continuous
        self.what = f"continuous callback for {label}"
        self.state = Maker(phase.state, ReadOnlyRows, f"{label} state")
        self.control = Maker(phase.control, ReadOnlyRows, f"{label} control")
        self.parameter = Maker(spec.parameter, ReadOnlyRows, "parameter")
        self.dynamics = Maker(phase.state, Rows, f"{label} dynamics")
        self.path = Maker(phase.path, Rows, f"{label} path")
        self.integrand = Maker(phase.integral, Rows, f"{label} integrand")
        self._arg = None
        self.has_state = phase.state._nrows > 0
        self.has_path = phase.path._nrows > 0
        self.has_integral = phase.integral._nrows > 0


def _make_continuous(spec: ProblemSpec_) -> Callable[[Any], None]:
    """Return the 0.3.0 continuous callback that drives every phase's own callback."""
    makers = tuple(_PhaseMakers(spec, phase) for phase in spec.phases)

    def continuous(arg: Any) -> None:
        for index in arg.phase_list:
            maker = makers[index]
            data = arg.phase[index]
            new_arg = maker.arg(data, arg.parameter)
            out = ContinuousOut(maker.dynamics.make(), maker.path.make(), maker.integrand.make())
            result = _call(maker.callback, maker.what, arg, new_arg, out)
            _check_return(result, out, maker.callback, maker.what)
            _check_complete(out, maker.callback, maker.what)
            # The rows go into the array the assembly reads. They do not go through the
            # released API's output object, whose row-by-row bookkeeping the vector has
            # already done.
            if maker.has_state:
                _hand_over(data, "dynamics", out.dynamics)
            if maker.has_path:
                _hand_over(data, "path", out.path)
            if maker.has_integral:
                _hand_over(data, "integrand", out.integrand)

    return continuous


def _hand_over(data: Any, name: str, vector: Any) -> None:
    """Copy one continuous output's rows into the array the transcription reads.

    Parameters
    ----------
    data : ContinuousPhase
        The phase as the transcription passed it.
    name : str
        The output being handed over.
    vector : Vector
        The output the callback filled.
    """
    buffer = data.output_storage(name)
    for row, value in enumerate(vector._row_values()):
        buffer[row] = value


class _EndpointMakers:
    """What an endpoint call needs that does not change between calls.

    Each phase's `Endpoint` is built only when the callback asks for that phase, so an
    objective that reads one phase does not pay for the others.
    """

    __slots__ = (
        "_built",
        "final_state",
        "independent",
        "indices",
        "initial_state",
        "integral",
        "parameter",
    )

    def __init__(self, spec: ProblemSpec_) -> None:
        self._built: dict[int, tuple[Any, EndpointArg]] = {}
        self.indices = {phase.handle: phase.index for phase in spec.phases}
        self.initial_state = {}
        self.final_state = {}
        self.integral = {}
        self.independent = {}
        for phase in spec.phases:
            label = f"phase '{phase.name}'"
            handle = phase.handle
            self.initial_state[handle] = Maker(phase.state, ReadOnlyRows, f"{label} initial state")
            self.final_state[handle] = Maker(phase.state, ReadOnlyRows, f"{label} final state")
            self.integral[handle] = Maker(phase.integral, ReadOnlyRows, f"{label} integral")
            self.independent[handle] = phase.independent
        self.parameter = Maker(spec.parameter, ReadOnlyRows, "parameter")

    def build(self, handle: Any, arg: Any) -> Endpoint:
        """Return the endpoint values of one phase, from what the solver passed."""
        data = arg.phase[self.indices[handle]]
        name = self.independent[handle]
        return Endpoint(
            data,
            EndpointValues(
                self.initial_state[handle].over(data.initial_state),
                lambda: data.initial_time,
                name,
            ),
            EndpointValues(
                self.final_state[handle].over(data.final_state),
                lambda: data.final_time,
                name,
            ),
            self.integral[handle].over(data.integral),
        )

    def arg(self, arg: Any) -> EndpointArg:
        """Return the endpoint argument for `arg`, building it once per argument object.

        The transcription builds one argument object per factory and writes new values into the
        arrays behind it, and the vectors here read their rows where they are. So the wrapper
        can be built once and will see each new point without being rebuilt -- which is most of
        what an endpoint call used to cost.
        """
        cached = self._built.get(id(arg))
        if cached is None or cached[0] is not arg:
            endpoints = _Endpoints(self, arg)
            built = EndpointArg(endpoints, self.parameter.over(arg.parameter))
            self._built[id(arg)] = (arg, built)
            return built
        return cached[1]


class _Endpoints:
    """The endpoints of every phase, each built when it is first asked for and then kept.

    An `Endpoint` holds read-only views of the transcription's own arrays, so once built it
    keeps reading the current point. Its `duration` is the one value computed rather than read,
    so it is recomputed on each access.
    """

    __slots__ = ("_arg", "_built", "_makers")

    def __init__(self, makers: _EndpointMakers, arg: Any) -> None:
        self._makers = makers
        self._arg = arg
        self._built: dict[Any, Endpoint] = {}

    def __getitem__(self, handle: Any) -> Endpoint:
        """Return the endpoint values of `handle`, building them on first use."""
        endpoint = self._built.get(handle)
        if endpoint is None:
            if handle not in self._makers.indices:
                raise KeyError(handle)
            endpoint = self._makers.build(handle, self._arg)
            self._built[handle] = endpoint
        return endpoint


def _make_objective(spec: ProblemSpec_, makers: _EndpointMakers) -> Callable[[Any], None]:
    """Return the 0.3.0 objective callback that drives the user's objective."""
    callback = spec.objective_function

    def objective(arg: Any) -> None:
        value = _call(callback, "objective callback", arg, makers.arg(arg))
        if value is None:
            name = getattr(callback, "__qualname__", repr(callback))
            msg = f"the objective callback '{name}' returned nothing; it must return the objective"
            raise ValueError(msg)
        arg.objective = value

    return objective


def _make_discrete(
    spec: ProblemSpec_, makers: _EndpointMakers, callback: Callable[..., Any]
) -> Callable[[Any], None]:
    """Return the 0.3.0 discrete callback that drives the user's discrete callback."""
    discrete_maker = Maker(spec.discrete, Rows, "discrete")

    def discrete(arg: Any) -> None:
        out = DiscreteOut(discrete_maker.make())
        result = _call(callback, "discrete callback", arg, makers.arg(arg), out)
        _check_return(result, out, callback, "discrete callback")
        _check_complete(out, callback, "discrete callback")
        buffer = arg.output_storage()
        for row, value in enumerate(out.discrete._row_values()):
            buffer[row] = value

    return discrete


def _flat(declaration: type[Vector], values: dict[str, Any], pick: Any) -> list[Any]:
    """Return one value per declared row, taking `pick` of each stored element."""
    return [
        pick(values[name][0 if member is None else member]) for name, member in declaration._rows
    ]


def _bound_arrays(declaration: type[Vector], values: dict[str, Any]) -> tuple[Any, Any]:
    """Return the lower and upper bound of every declared row."""
    return (
        frozen_array(_flat(declaration, values, lambda bound: bound[0])),
        frozen_array(_flat(declaration, values, lambda bound: bound[1])),
    )


def _scale_array(declaration: type[Vector], values: dict[str, Any]) -> Any:
    """Return the scale of every declared row."""
    return frozen_array(_flat(declaration, values, float))


def _guess_grid(
    declarations: tuple[tuple[type[Vector], dict[str, Any]], ...], span: tuple[float, float]
) -> Any:
    """Return the time grid a phase's guess is given on.

    Each field carries its own sample times, which is what removes 0.3.0's requirement that
    every guessed row share one grid. The transcription still interpolates from a single grid
    per phase, so the sample times of every field are merged here, clipped to the phase.
    """
    t0, tf = span
    times = {t0, tf}
    for declaration, values in declarations:
        for name in declaration._fields:
            # Every row of a field, not just the first: a block field may be given one sampled
            # guess per row, each with its own sample times.
            for guess in values[name]:
                if guess[0] == "sampled":
                    times.update(t for t in guess[1].time.tolist() if t0 < t < tf)
    return np.array(sorted(times), dtype=float)


def _guess_rows(
    declaration: type[Vector], values: dict[str, Any], grid: Any, span: tuple[float, float]
) -> Any:
    """Return one row of guessed values per declared row, on `grid`.

    A constant holds over the phase; a pair is linear from one end to the other; samples are
    interpolated linearly, holding their end values where they do not reach the ends of the
    phase, which is what `numpy.interp` does.
    """
    t0, tf = span
    rows = []
    for name, member in declaration._rows:
        index = 0 if member is None else member
        guess = values[name][index]
        if guess[0] == "constant":
            rows.append(np.full(grid.shape, guess[1]))
        elif guess[0] == "linear":
            rows.append(np.interp(grid, [t0, tf], [guess[1], guess[2]]))
        else:
            sampled = guess[1]
            size = declaration._meta[name].rows
            row = sampled.rows(size, "guess", name)[index]
            rows.append(np.interp(grid, sampled.time, row))
    return np.array(rows, dtype=float) if rows else np.zeros((0, grid.size))


def _phase_spec(phase: PhaseSpec_) -> PhaseSpec:
    """Reduce one phase of the snapshot to the numbers the transcription reads."""
    nx, nu = phase.state._nrows, phase.control._nrows
    nq, nh = phase.integral._nrows, phase.path._nrows
    span = phase.time_guess
    grid = _guess_grid(
        ((phase.state, phase.state_guess), (phase.control, phase.control_guess)), span
    )
    state_lower, state_upper = _bound_arrays(phase.state, phase.state_bounds)
    initial_lower, initial_upper = _bound_arrays(phase.state, phase.state_initial)
    final_lower, final_upper = _bound_arrays(phase.state, phase.state_final)
    control_lower, control_upper = _bound_arrays(phase.control, phase.control_bounds)
    path_lower, path_upper = _bound_arrays(phase.path, phase.path_bounds)
    integral_lower, integral_upper = _bound_arrays(phase.integral, phase.integral_bounds)
    return PhaseSpec(
        index=phase.index,
        nx=nx,
        nu=nu,
        nq=nq,
        nh=nh,
        state_lower=state_lower,
        state_upper=state_upper,
        initial_state_lower=initial_lower,
        initial_state_upper=initial_upper,
        final_state_lower=final_lower,
        final_state_upper=final_upper,
        control_lower=control_lower,
        control_upper=control_upper,
        path_lower=path_lower,
        path_upper=path_upper,
        integral_lower=integral_lower,
        integral_upper=integral_upper,
        zero_mode_lower=frozen_array(np.full(nx, -np.inf)),
        zero_mode_upper=frozen_array(np.full(nx, np.inf)),
        initial_time_lower=phase.time_initial[0],
        initial_time_upper=phase.time_initial[1],
        final_time_lower=phase.time_final[0],
        final_time_upper=phase.time_final[1],
        duration_lower=0.0,
        duration_upper=math.inf,
        state_scale=_scale_array(phase.state, phase.state_scale),
        control_scale=_scale_array(phase.control, phase.control_scale),
        integral_scale=_scale_array(phase.integral, phase.integral_scale),
        dynamics_scale=_scale_array(phase.state, phase.state_defect_scale),
        path_scale=_scale_array(phase.path, phase.path_scale),
        time_scale=phase.time_scale,
        guess_time=frozen_array(grid),
        guess_state=frozen_array(_guess_rows(phase.state, phase.state_guess, grid, span)),
        guess_control=frozen_array(_guess_rows(phase.control, phase.control_guess, grid, span)),
        guess_integral=frozen_array(_flat(phase.integral, phase.integral_guess, float), nq),
        fraction=tuple(phase.mesh.fractions),
        collocation_points=tuple(phase.mesh.collocation_points),
    )


def to_transcription_spec(spec: ProblemSpec_) -> ProblemSpec:
    """Reduce a snapshot of the redesigned API's problem to what the transcription reads.

    Parameters
    ----------
    spec : yapss._api.spec.ProblemSpec
        The snapshot taken when the solve began.

    Returns
    -------
    yapss._backend.spec.ProblemSpec
        The same problem as counts, arrays, and callbacks.
    """
    functions = UserFunctions()
    endpoint_makers = _EndpointMakers(spec)
    functions.objective = _make_objective(spec, endpoint_makers)
    functions.continuous = _make_continuous(spec)
    if spec.discrete_function is not None:
        functions.discrete = _make_discrete(spec, endpoint_makers, spec.discrete_function)
    discrete_lower, discrete_upper = _bound_arrays(spec.discrete, spec.discrete_bounds)
    parameter_lower, parameter_upper = _bound_arrays(spec.parameter, spec.parameter_bounds)
    return ProblemSpec(
        name=spec.name,
        phases=tuple(_phase_spec(phase) for phase in spec.phases),
        nd=spec.discrete._nrows,
        ns=spec.parameter._nrows,
        discrete_lower=discrete_lower,
        discrete_upper=discrete_upper,
        parameter_lower=parameter_lower,
        parameter_upper=parameter_upper,
        discrete_scale=_scale_array(spec.discrete, spec.discrete_scale),
        parameter_scale=_scale_array(spec.parameter, spec.parameter_scale),
        objective_scale=spec.objective_scale,
        guess_parameter=frozen_array(
            _flat(spec.parameter, spec.parameter_guess, float), spec.parameter._nrows
        ),
        functions=functions,
        auxdata=None,
        sense=spec.sense,
        spectral_method=spec.spectral_method,
        derivative_method=spec.derivative_method,
        derivative_order=spec.derivative_order,
        ipopt_options=dict(spec.ipopt_options),
        catch_keyboard_interrupt=spec.catch_keyboard_interrupt,
    )


def solve_problem(spec: ProblemSpec_) -> tuple[Solution, Any]:
    """Solve the problem `spec` describes and return the solution in the new API's shape.

    Parameters
    ----------
    spec : yapss._api.spec.ProblemSpec
        The snapshot to solve.

    Returns
    -------
    Solution
        The solution, which holds data only.
    record
        The back end's record of the solve, which `Problem.solve` reads to warn about a solve
        that did not converge. The solution does not keep it: the record holds the problem.
    """
    transcription = to_transcription_spec(spec)
    record = solve(transcription)
    return Solution._from(spec, record, transcription), record
