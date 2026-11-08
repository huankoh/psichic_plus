import pytest

from applications.scorer import Output
from applications.scoring import Acceptance, encode_native_parameter
from .conftest import prepare, transfer


@pytest.mark.parametrize("parameter", ["alpha", "beta"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None, "nan", "inf", "-inf"])
def test_native_diagnostics_preserve_value_type_and_finite_affinity(make_run, tmp_path, parameter, value):
    run = make_run(n=1, psichic_output=Output.DISTRIBUTION)
    task, = prepare(run)
    evidence = transfer(tmp_path / "transfer", task, distribution_failure=True, **{parameter: value})
    assert run.accept(evidence) is Acceptance.ACCEPTED
    run.finalize()
    raw = next(run.results())["psichic"]["raw"]
    assert raw["raw_affinity"] == 6
    assert raw["distribution"]["raw_" + parameter] == repr(value)


@pytest.mark.parametrize("name", ["nan", "inf", "-inf"])
def test_native_parameter_encoding_distinguishes_strings_from_floats(name):
    assert encode_native_parameter(float(name)) == {"nonfinite_float": name}
    assert encode_native_parameter(name) == name
    assert encode_native_parameter(None) is None
    assert encode_native_parameter(4.0) == 4.0
