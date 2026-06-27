from pinn.utils.profiling import profile_performance, MemoryTracker, PerformanceReport


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
