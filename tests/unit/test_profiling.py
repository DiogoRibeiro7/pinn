from pinnlab.utils.profiling import (
    profile_performance,
    MemoryTracker,
    PerformanceReport,
)


def test_profile_decorator_records_duration():
    report = PerformanceReport()

    @profile_performance(report=report)
    def slow_add(x, y):
        return x + y

    assert slow_add(1, 2) == 3
    assert len(report.records) == 1
    rec = report.records[0]
    assert rec.name.endswith("slow_add")
    assert rec.duration_sec >= 0


def test_memory_tracker_records_peak():
    with MemoryTracker() as mt:
        data = [0] * 10000
        data[0] = 1  # use the list
    assert mt.peak_memory > 0


def test_profile_decorator_uses_an_explicit_name():
    """A caller can label the record instead of inheriting __qualname__.

    Benchmark dashboards key their history off this string, so a nested
    function's generated name (``main.<locals>.run_train``) is both unreadable
    and awkward to change later without splitting the series.
    """
    report = PerformanceReport()

    @profile_performance(report=report, name="burgers_training")
    def run() -> int:
        return 7

    assert run() == 7
    assert report.records[0].name == "burgers_training"


def test_profile_decorator_defaults_to_the_qualified_name():
    report = PerformanceReport()

    def outer():
        @profile_performance(report=report)
        def inner() -> None:
            return None

        inner()

    outer()
    # Unnamed nested functions still carry the generated qualname.
    assert report.records[0].name.endswith("inner")
    assert "<locals>" in report.records[0].name
