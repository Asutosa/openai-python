from __future__ import annotations

from typing import AsyncIterator, cast

import anyio
import httpx2
from anyio.lowlevel import checkpoint

from ._httpx2 import request_exceptions, stream_consumed_exceptions

_DRAIN_TIMEOUT = 0.1


async def drain_response_async(response: httpx2.Response) -> None:
    if response.is_closed:
        return
    try:
        with anyio.move_on_after(_DRAIN_TIMEOUT):
            # Discard raw transport bytes without decoding trailing SSE or content.
            async for _ in cast(AsyncIterator[bytes], response.stream):
                # Also enforce the deadline on transports yielding buffered data.
                await checkpoint()
    except request_exceptions() + stream_consumed_exceptions():
        pass  # Cleanup must not turn completed output into a request error.
    finally:
        # Close outside our timeout scope and shield cleanup from caller
        # cancellation. Cancellation still propagates after the response closes.
        with anyio.CancelScope(shield=True):
            await response.aclose()


from typing import Any, Callable, Optional, Awaitable


def _cleanup_error(
    primary: Optional[BaseException],
    failure: Optional[BaseException],
) -> Optional[BaseException]:
    if failure is None:
        return None
    if primary is None:
        return failure
    if not isinstance(failure, Exception) and isinstance(primary, Exception):
        return failure
    return None


def _keep(
    failure: Optional[BaseException], err: BaseException
) -> BaseException:
    if failure is None or (
        isinstance(failure, Exception) and not isinstance(err, Exception)
    ):
        return err
    return failure


def close_owned(
    *resources: object, primary: Optional[BaseException] = None
) -> None:
    failure: Optional[BaseException] = None
    for resource in resources:
        try:
            close = getattr(resource, "close", None)
            if close is not None:
                close()
        except BaseException as err:
            failure = _keep(failure, err)
    error = _cleanup_error(primary, failure)
    if error is not None:
        raise error


async def aclose_owned(
    *resources: object, primary: Optional[BaseException] = None
) -> None:
    failure: Optional[BaseException] = None
    with anyio.CancelScope(shield=True):
        for resource in resources:
            try:
                aclose = getattr(resource, "aclose", None)
                if aclose is not None:
                    await aclose()
            except BaseException as err:
                failure = _keep(failure, err)
    error = _cleanup_error(primary, failure)
    if error is not None:
        raise error


async def run_cleanup(
    cleanup: Callable[[], Awaitable[Any]],
    *,
    primary: Optional[BaseException] = None,
) -> None:
    failure: Optional[BaseException] = None
    with anyio.CancelScope(shield=True):
        try:
            await cleanup()
        except BaseException as err:
            failure = err
    error = _cleanup_error(primary, failure)
    if error is not None:
        raise error
