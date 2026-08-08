from data._retry import retry_with_backoff


def test_returns_result_on_first_success_without_retrying():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result = retry_with_backoff(fn, attempts=3, base_delay=0)
    assert result == "ok"
    assert len(calls) == 1


def test_retries_after_a_transient_failure_then_succeeds():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("transient")
        return "ok"

    result = retry_with_backoff(fn, attempts=3, base_delay=0)
    assert result == "ok"
    assert len(calls) == 2


def test_returns_none_after_exhausting_all_attempts():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("permanent failure")

    result = retry_with_backoff(fn, attempts=3, base_delay=0)
    assert result is None
    assert len(calls) == 3


def test_does_not_retry_a_successful_call_that_returns_none():
    calls = []

    def fn():
        calls.append(1)
        return None

    result = retry_with_backoff(fn, attempts=3, base_delay=0)
    assert result is None
    assert len(calls) == 1  # a clean None return is not an error -- no retry needed
