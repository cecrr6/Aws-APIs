"""
Shared utilities for Ayyad APIs library.

This module provides common utility functions used across different API modules.
"""

import json
import logging
import asyncio
import os
import time
from abc import ABC
from dataclasses import dataclass, field, asdict, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any, List, Tuple, Type

import aiohttp
import aiofiles


# Configure logging
logger = logging.getLogger(__name__)


# ==================== RapidAPI Helper Functions ====================

def create_rapidapi_headers(
    api_key: str,
    rapidapi_host: str,
    content_type: str = "application/json"
) -> Dict[str, str]:
    """
    Create standard RapidAPI headers.

    This is a shared utility to create consistent headers across all RapidAPI modules.

    Args:
        api_key: RapidAPI key for authentication
        rapidapi_host: RapidAPI host (e.g., "api-name.p.rapidapi.com")
        content_type: Content type for the request (default: "application/json")

    Returns:
        Dictionary with RapidAPI headers

    Example:
        headers = create_rapidapi_headers(
            api_key="your_key",
            rapidapi_host="example.p.rapidapi.com"
        )
    """
    return {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": rapidapi_host,
        "Content-Type": content_type
    }


async def validate_rapidapi_response(
    response: aiohttp.ClientResponse,
    auth_error_class: Type[Exception],
    request_error_class: Type[Exception],
    client_error_class: Optional[Type[Exception]] = None
) -> Dict[str, Any]:
    """
    Validate RapidAPI response and handle common errors.

    This is a shared utility to handle response validation consistently
    across all RapidAPI modules.

    Args:
        response: aiohttp ClientResponse object
        auth_error_class: Exception class to raise for authentication errors (401/403)
        request_error_class: Exception class to raise for server errors (5xx)
        client_error_class: Exception class to raise for client errors (4xx, excluding 401/403)

    Returns:
        Parsed JSON response as dictionary

    Raises:
        auth_error_class: If authentication fails (401/403)
        client_error_class: If client error (4xx) - should NOT be retried
        request_error_class: If server error (5xx) - can be retried

    Example:
        async with session.get(url) as response:
            data = await validate_rapidapi_response(
                response,
                MyAuthError,
                MyRequestError,
                MyClientError
            )
    """
    # Check for authentication errors
    if response.status in (401, 403):
        raise auth_error_class(f"Authentication failed: {response.status}")

    # Check for client errors (4xx) - should NOT be retried
    if 400 <= response.status < 500:
        error_text = await response.text()
        error_class = client_error_class or request_error_class
        raise error_class(
            f"Client error {response.status}: {error_text}",
            status_code=response.status,
            response_text=error_text
        )

    # Check for server errors (5xx) - can be retried
    if response.status >= 500:
        error_text = await response.text()
        raise request_error_class(
            f"Server error {response.status}: {error_text}",
            status_code=response.status,
            response_text=error_text
        )

    # Parse JSON response
    try:
        data: Dict[str, Any] = await response.json()
        return data
    except (aiohttp.ContentTypeError, ValueError):
        error_text = await response.text()
        raise request_error_class(f"Invalid JSON response: {error_text}")


# ==================== File Download Utilities ====================


async def download_file(
    url: str,
    output_path: Optional[Union[str, Path]] = None,
    return_bytes: bool = False,
    default_filename: str = "download",
    default_ext: str = ".bin",
    max_retries: int = 3,
    retry_delay: float = 2.0,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 8192,
    session: Optional[aiohttp.ClientSession] = None
) -> Union[bytes, str, None]:
    """
    Download a file from URL - unified function for the entire library.

    This is a shared utility function used by all API modules to download
    media files (images, videos, audio) from URLs with support for:
    - Progress tracking
    - Retry logic
    - Custom chunk sizes
    - Reusable sessions

    Args:
        url: URL to download from
        output_path: Path to save the file. If None, uses default_filename + extension from URL
        return_bytes: If True, returns bytes instead of saving to file
        default_filename: Default filename if can't determine from URL
        default_ext: Default extension if can't determine from URL
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay in seconds between retries (default: 2.0)
        show_progress: Show download progress in console (default: False)
        progress_callback: Optional callback function(downloaded_bytes, total_bytes)
        chunk_size: Size of chunks for streaming download (default: 8192)
        session: Optional aiohttp session to reuse (if None, creates new one)

    Returns:
        - bytes if return_bytes=True
        - str (file path) if saved to disk
        - None if download fails

    Example:
        # Simple download
        path = await download_file("https://example.com/video.mp4", "my_video.mp4")

        # Download with progress
        path = await download_file(
            "https://example.com/large_video.mp4",
            "video.mp4",
            show_progress=True,
            max_retries=5
        )

        # Get as bytes
        data = await download_file("https://example.com/image.jpg", return_bytes=True)

        # Custom progress callback
        def on_progress(downloaded, total):
            print(f"Downloaded: {downloaded}/{total} bytes")

        path = await download_file(
            "https://example.com/file.zip",
            progress_callback=on_progress
        )

    Note:
        This function is used internally by all media result classes (VideoInfo,
        Format, ImageDownloadResult, VideoDownloadResult, etc.) in their
        download() methods.
    """
    if not url:
        logger.error("No URL provided")
        return None

    # Determine output path for file downloads
    final_output_path: Optional[Path] = None
    if not return_bytes:
        if output_path is None:
            # Try to extract extension from URL
            ext = default_ext
            if "." in url:
                url_ext = url.split(".")[-1].split("?")[0]
                if url_ext and len(url_ext) <= 5:  # Reasonable extension length
                    ext = f".{url_ext}"
            output_path = f"{default_filename}{ext}"

        final_output_path = Path(output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Retry logic
    close_session: bool = False

    for attempt in range(max_retries):
        try:
            # Create session if not provided
            if session is None:
                session = aiohttp.ClientSession()
                close_session = True

            if attempt > 0:
                logger.info(f"[Download] Retry attempt {attempt + 1}/{max_retries} for: {url}")

            async with session.get(url) as response:
                if response.status != 200:
                    error_msg = f"HTTP {response.status}: Download failed"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                total_size: int = int(response.headers.get('content-length', 0))

                if return_bytes:
                    # Simple read for bytes
                    if show_progress and total_size > 0:
                        logger.info(f"[Download] Downloading {total_size:,} bytes from: {url}")

                    content: bytes = await response.read()

                    if show_progress:
                        logger.info(f"[Download] Completed: {len(content):,} bytes")

                    # Close session if we created it
                    if close_session:
                        await session.close()
                        session = None

                    return content

                # Streaming download to file
                if show_progress:
                    if total_size > 0:
                        logger.info(f"[Download] Starting download: {final_output_path} ({total_size:,} bytes)")
                    else:
                        logger.info(f"[Download] Starting download: {final_output_path}")

                downloaded: int = 0
                last_logged_percent: int = 0

                async with aiofiles.open(final_output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)

                        # Progress tracking
                        if total_size > 0:
                            percentage: float = (downloaded / total_size) * 100

                            # Console progress
                            if show_progress:
                                print(f"\r[Download] Progress: {percentage:.1f}% ({downloaded:,}/{total_size:,} bytes)", end="", flush=True)

                                # Log every 10%
                                current_milestone: int = int(percentage // 10) * 10
                                if current_milestone > last_logged_percent and current_milestone > 0:
                                    print()  # New line
                                    logger.info(f"[Download] {current_milestone}% completed")
                                    last_logged_percent = current_milestone

                            # Custom callback
                            if progress_callback:
                                progress_callback(downloaded, total_size)
                        else:
                            # Unknown size
                            if show_progress:
                                print(f"\r[Download] Downloaded: {downloaded:,} bytes", end="", flush=True)

                            if progress_callback:
                                progress_callback(downloaded, 0)

                if show_progress:
                    print()  # New line after progress
                    logger.info(f"[Download] Completed: {final_output_path} ({downloaded:,} bytes)")

                # Close session if we created it
                if close_session:
                    await session.close()
                    session = None

                return str(final_output_path)

        except Exception as e:
            logger.error(f"[Download] Attempt {attempt + 1} failed: {str(e)}")

            # Close session on error if we created it
            if close_session and session:
                await session.close()
                session = None

            # Retry if not last attempt
            if attempt < max_retries - 1:
                logger.warning(f"[Download] Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"[Download] All {max_retries} attempts failed")

    # All retries failed
    logger.error(f"[Download] Failed to download from: {url}")
    return None


# ==================== Base Classes and Utilities ====================


class BaseResponse:
    """
    Base class for all API response models.

    Provides automatic to_dict() and to_json() methods for dataclass instances.
    All response models should inherit from this class.

    Example:
        @dataclass
        class MyResult(BaseResponse):
            value: str
            count: int

        result = MyResult(value="test", count=5)
        print(result.to_json(indent=2))
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        if is_dataclass(self):
            return asdict(self)
        raise NotImplementedError("Subclass must be a dataclass")

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ==================== Exception Hierarchy ====================


class APIError(Exception):
    """
    Base exception for all API errors with rich context.

    Provides detailed error information including:
    - HTTP status code
    - Response text
    - Endpoint that failed
    - Request parameters
    - Retry count
    - Original exception
    - Timestamp

    Example:
        raise APIError(
            "Request failed",
            status_code=500,
            endpoint="/api/endpoint",
            retry_count=3
        )
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        endpoint: Optional[str] = None,
        request_params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        original_error: Optional[Exception] = None
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.status_code: Optional[int] = status_code
        self.response_text: Optional[str] = response_text
        self.endpoint: Optional[str] = endpoint
        self.request_params: Optional[Dict[str, Any]] = request_params
        self.retry_count: int = retry_count
        self.original_error: Optional[Exception] = original_error
        self.timestamp: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/debugging."""
        result: Dict[str, Any] = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "timestamp": self.timestamp
        }

        if self.status_code is not None:
            result["status_code"] = self.status_code
        if self.endpoint:
            result["endpoint"] = self.endpoint
        if self.retry_count > 0:
            result["retry_count"] = self.retry_count
        if self.response_text:
            result["response_text"] = self.response_text[:200]  # Truncate long responses

        return result

    def __str__(self) -> str:
        """String representation with context."""
        parts: List[str] = [self.message]
        if self.endpoint:
            parts.append(f"endpoint={self.endpoint}")
        if self.status_code:
            parts.append(f"status={self.status_code}")
        if self.retry_count > 0:
            parts.append(f"retries={self.retry_count}")
        return " | ".join(parts)


class AuthenticationError(APIError):
    """Raised when API authentication fails (401/403)."""
    pass


class ClientError(APIError):
    """Raised for client errors (4xx) - should NOT be retried."""
    pass


class RequestError(APIError):
    """Raised when API request fails (5xx or network errors) - can be retried."""
    pass


class InvalidInputError(APIError):
    """Raised when input validation fails."""
    pass


class DownloadError(APIError):
    """Raised when download operation fails."""
    pass


# ==================== Configuration Management ====================


@dataclass
class APIConfig:
    """
    Centralized configuration for API clients.

    Can be loaded from environment variables or created programmatically.

    Example:
        # From environment variables
        config = APIConfig.from_env("MYAPI")
        # Looks for: MYAPI_KEY, MYAPI_HOST, MYAPI_TIMEOUT, etc.

        # Programmatically
        config = APIConfig(
            api_key="your_key",
            timeout=60,
            max_retries=5,
            show_progress=True
        )

        # Use with API client
        async with MyAPI(config=config) as client:
            result = await client.method()
    """

    # API credentials
    api_key: Optional[str] = None
    rapidapi_host: Optional[str] = None

    # Request settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0

    # Download settings
    default_chunk_size: int = 8192
    show_progress: bool = False

    # Rate limiting (requests per second)
    rate_limit: Optional[float] = None

    # Extra headers
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, prefix: str = "AYYAD_API") -> "APIConfig":
        """
        Create config from environment variables.

        Args:
            prefix: Environment variable prefix (default: "AYYAD_API")

        Environment variables:
            {PREFIX}_KEY: API key
            {PREFIX}_HOST: RapidAPI host
            {PREFIX}_TIMEOUT: Request timeout in seconds
            {PREFIX}_MAX_RETRIES: Maximum retry attempts
            {PREFIX}_RETRY_DELAY: Delay between retries in seconds
            {PREFIX}_SHOW_PROGRESS: Show progress (true/false)

        Example:
            export AYYAD_API_KEY=your_key
            export AYYAD_API_TIMEOUT=60
            export AYYAD_API_MAX_RETRIES=5

            config = APIConfig.from_env()
        """
        return cls(
            api_key=os.getenv(f"{prefix}_KEY"),
            rapidapi_host=os.getenv(f"{prefix}_HOST"),
            timeout=int(os.getenv(f"{prefix}_TIMEOUT", "30")),
            max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv(f"{prefix}_RETRY_DELAY", "2.0")),
            show_progress=os.getenv(f"{prefix}_SHOW_PROGRESS", "").lower() == "true"
        )


# ==================== Retry Decorator ====================


def with_retry(
    max_attempts: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    no_retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None
) -> Callable:
    """
    Decorator for automatic retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 2.0)
        backoff: Multiplier for delay after each retry (default: 2.0)
        exceptions: Tuple of exceptions to catch and retry (default: RequestError, aiohttp.ClientError)
        no_retry_exceptions: Exceptions that should NOT be retried (default: ClientError, AuthenticationError)

    Example:
        @with_retry(max_attempts=5, delay=1.0, backoff=2.0)
        async def fetch_data(self):
            return await self._make_request("/endpoint")

        # First attempt fails -> wait 1.0s -> retry
        # Second attempt fails -> wait 2.0s -> retry
        # Third attempt fails -> wait 4.0s -> retry
        # etc.

    Note:
        ClientError (4xx) and AuthenticationError are never retried as they
        indicate client-side issues that won't be resolved by retrying.
    """
    # Default exceptions if not provided
    if exceptions is None:
        exceptions = (RequestError, aiohttp.ClientError, ClientError, AuthenticationError)

    # Exceptions that should NOT be retried
    if no_retry_exceptions is None:
        no_retry_exceptions = (ClientError, AuthenticationError)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay: float = delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except no_retry_exceptions:
                    # Don't retry client errors or auth errors - re-raise immediately
                    raise
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {str(e)}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")

            # Wrap the original exception in APIError with retry context
            if isinstance(last_exception, APIError):
                last_exception.retry_count = max_attempts - 1
                raise last_exception
            else:
                raise RequestError(
                    f"Request failed after {max_attempts} attempts",
                    original_error=last_exception,
                    retry_count=max_attempts - 1
                )

        return wrapper
    return decorator


# ==================== Progress Tracking ====================


@dataclass
class ProgressInfo:
    """
    Information about operation progress.

    Attributes:
        current: Current progress value
        total: Total expected value
        percentage: Percentage complete (0-100)
        elapsed_time: Seconds elapsed since start
        estimated_total_time: Estimated total time in seconds (if calculable)
        eta: Estimated time remaining in seconds (if calculable)
        speed: Items/bytes per second (if calculable)
    """
    current: int
    total: int
    percentage: float
    elapsed_time: float
    estimated_total_time: Optional[float] = None
    eta: Optional[float] = None
    speed: Optional[float] = None

    def __str__(self) -> str:
        """Human-readable progress string."""
        if self.eta and self.eta > 0:
            return f"{self.percentage:.1f}% ({self.current}/{self.total}) - ETA: {self.eta:.1f}s"
        return f"{self.percentage:.1f}% ({self.current}/{self.total})"


class ProgressTracker:
    """
    Tracks progress for long-running operations.

    Automatically calculates:
    - Percentage complete
    - Elapsed time
    - Speed (items/bytes per second)
    - ETA (estimated time remaining)

    Example:
        def on_progress(info: ProgressInfo):
            print(f"Progress: {info.percentage:.1f}% - ETA: {info.eta:.0f}s")

        tracker = ProgressTracker(total=100, callback=on_progress)

        for i in range(100):
            # Do work...
            tracker.update(i + 1)

        tracker.complete()
    """

    def __init__(
        self,
        total: int,
        callback: Optional[Callable[[ProgressInfo], None]] = None,
        update_interval: float = 0.5
    ) -> None:
        """
        Initialize progress tracker.

        Args:
            total: Total number of items/bytes expected
            callback: Function to call with ProgressInfo on updates
            update_interval: Minimum seconds between callbacks (default: 0.5)
        """
        self.total: int = total
        self.callback: Optional[Callable[[ProgressInfo], None]] = callback
        self.update_interval: float = update_interval

        self.current: int = 0
        self.start_time: float = time.time()
        self.last_update_time: float = 0.0

    def update(self, current: int, force: bool = False) -> None:
        """
        Update progress.

        Args:
            current: Current progress value
            force: Force callback even if update_interval hasn't elapsed
        """
        self.current = current

        now: float = time.time()
        if not force and (now - self.last_update_time) < self.update_interval:
            return

        self.last_update_time = now

        if self.callback:
            info: ProgressInfo = self.get_progress_info()
            self.callback(info)

    def get_progress_info(self) -> ProgressInfo:
        """Get current progress information."""
        elapsed: float = time.time() - self.start_time
        percentage: float = (self.current / self.total * 100) if self.total > 0 else 0

        # Calculate speed and ETA
        speed: float = self.current / elapsed if elapsed > 0 else 0
        remaining: int = self.total - self.current
        eta: Optional[float] = remaining / speed if speed > 0 else None
        estimated_total: Optional[float] = elapsed + eta if eta else None

        return ProgressInfo(
            current=self.current,
            total=self.total,
            percentage=percentage,
            elapsed_time=elapsed,
            estimated_total_time=estimated_total,
            eta=eta,
            speed=speed
        )

    def complete(self) -> None:
        """Mark operation as complete and trigger final callback."""
        self.update(self.total, force=True)


# ==================== Base API Client ====================


class BaseRapidAPI(ABC):
    BASE_URL: str
    DEFAULT_HOST: str

    def __init__(
        self,
        api_key: str,
        rapidapi_host: Optional[str] = None,
        timeout: int = 30,
        config: Optional[APIConfig] = None,
        session: Optional[aiohttp.ClientSession] = None  
    ) -> None:
        if config:
            self.api_key = config.api_key or api_key
            self.rapidapi_host = config.rapidapi_host or rapidapi_host or self.DEFAULT_HOST
            self.timeout = aiohttp.ClientTimeout(total=config.timeout)
            self.config = config
        else:
            self.api_key = api_key
            self.rapidapi_host = rapidapi_host or self.DEFAULT_HOST
            self.timeout = aiohttp.ClientTimeout(total=timeout)
            self.config = APIConfig(api_key=api_key, rapidapi_host=self.rapidapi_host, timeout=timeout)
        
        self._session = session 
        self._external_session = session is not None 
        logger.info(f"{self.__class__.__name__} initialized")

    async def __aenter__(self) -> "BaseRapidAPI":
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
            print(" [LOG]  تم إنشاء جلسة جديدة")
        else:
            print(f" [LOG] لجلسة المخزنة تعمل! ID: {id(self._session)}")
        return self
        
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any
    ) -> bool:
        if self._session and not self._external_session:
            await self._session.close()
            logger.debug("HTTP session closed")
        return False

    def _get_headers(self, content_type: str = "application/json") -> Dict[str, str]:
        headers: Dict[str, str] = create_rapidapi_headers(
            api_key=self.api_key,
            rapidapi_host=self.rapidapi_host,
            content_type=content_type
        )

        if self.config and self.config.extra_headers:
            headers.update(self.config.extra_headers)

        return headers
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Make HTTP request with validation.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            endpoint: API endpoint path (e.g., "/analyze")
            **kwargs: Additional arguments for aiohttp request (params, json, data, etc.)

        Returns:
            JSON response as dictionary

        Raises:
            APIError: If session not initialized
            AuthenticationError: If authentication fails (401/403)
            RequestError: If request fails

        Example:
            # GET request
            data = await self._make_request("GET", "/endpoint", params={"q": "value"})

            # POST request
            data = await self._make_request("POST", "/endpoint", json={"key": "value"})
        """
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        url: str = f"{self.BASE_URL}{endpoint}"

        # Add headers if not provided
        if 'headers' not in kwargs:
            kwargs['headers'] = self._get_headers()

        logger.debug(f"Making {method} request to {endpoint}")

        try:
            async with self._session.request(method, url, **kwargs) as response:
                return await validate_rapidapi_response(
                    response,
                    AuthenticationError,
                    RequestError,
                    ClientError
                )
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            raise RequestError(
                f"Network error: {str(e)}",
                endpoint=endpoint,
                original_error=e
            )

    async def _make_text_request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any
    ) -> str:
        """
        Make HTTP request and return response as plain text (for non-JSON APIs).

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "/search")
            **kwargs: Additional arguments for aiohttp request

        Returns:
            Response text as string

        Raises:
            APIError: If session not initialized
            AuthenticationError: If authentication fails (401/403)
            RequestError: If request fails
        """
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        url: str = f"{self.BASE_URL}{endpoint}"

        if 'headers' not in kwargs:
            kwargs['headers'] = self._get_headers()

        logger.debug(f"Making {method} text request to {endpoint}")

        try:
            async with self._session.request(method, url, **kwargs) as response:
                if response.status in (401, 403):
                    raise AuthenticationError(
                        "Authentication failed",
                        status_code=response.status,
                        endpoint=endpoint
                    )

                if response.status != 200:
                    error_text: str = await response.text()
                    raise RequestError(
                        "Request failed",
                        status_code=response.status,
                        response_text=error_text,
                        endpoint=endpoint
                    )

                return await response.text()

        except (AuthenticationError, RequestError):
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            raise RequestError(
                f"Network error: {str(e)}",
                endpoint=endpoint,
                original_error=e
            )

    async def _post_form_data(
        self,
        endpoint: str,
        form_data: Any,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        POST multipart form data to an endpoint.

        Content-Type is omitted so aiohttp sets it automatically for multipart,
        which is required for correct boundary generation.

        Args:
            endpoint: API endpoint path (e.g., "/analyze-audio")
            form_data: aiohttp.FormData instance
            params: Optional query parameters

        Returns:
            JSON response as dictionary

        Raises:
            APIError: If session not initialized
            AuthenticationError: If authentication fails (401/403)
            RequestError: If request fails
        """
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        url: str = f"{self.BASE_URL}{endpoint}"

        # Omit Content-Type — aiohttp sets it automatically for multipart/form-data
        headers: Dict[str, str] = {
            "x-rapidapi-host": self.rapidapi_host,
            "x-rapidapi-key": self.api_key,
        }
        if self.config and self.config.extra_headers:
            headers.update(self.config.extra_headers)

        logger.debug(f"Making POST form-data request to {endpoint}")

        try:
            async with self._session.post(url, headers=headers, data=form_data, params=params) as response:
                return await validate_rapidapi_response(
                    response,
                    AuthenticationError,
                    RequestError,
                    ClientError
                )
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {str(e)}")
            raise RequestError(
                f"Network error: {str(e)}",
                endpoint=endpoint,
                original_error=e
            )
