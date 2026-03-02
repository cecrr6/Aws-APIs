from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any
import asyncio

# Import base classes and utilities
from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    APIError,
    AuthenticationError,
    RequestError,
    InvalidInputError,
    DownloadError,
    APIConfig,
    with_retry,
)

logger = logging.getLogger(__name__)


# ==================== Exception Aliases ====================

class APIResponseError(RequestError):
    """Error when API doesn't return 200 status or returns an error message"""
    def __init__(self, message: str):
        super().__init__(f"API Error: {message}")
        self.message = message


# ==================== Data Models ====================

@dataclass
class Channel(BaseResponse):
    """Channel information"""
    name: str = None
    id: str = None
    thumbnails: List[dict] = None
    link: str = None


@dataclass
class Video(BaseResponse):
    """Base video object shared across multiple responses"""
    success: bool = False
    video_title: str = None
    video_id: str = None
    video_url: str = None
    thumbnail: str = None
    view_count: int = None
    duration: int = None
    description: str = None
    category: str = None
    tags: List[str] = None
    uploader: Channel = None

    @property
    def duration_formatted(self) -> str:
        """Format video duration to HH:MM:SS."""
        if not self.duration:
            return "00:00:00"
        hours, remainder = divmod(self.duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def views_formatted(self) -> str:
        """Format view count with K/M suffix."""
        if not self.view_count:
            return "0"
        if self.view_count >= 1_000_000:
            return f"{self.view_count / 1_000_000:.1f}M"
        elif self.view_count >= 1_000:
            return f"{self.view_count / 1_000:.1f}K"
        return str(self.view_count)


@dataclass
class VideoInfoResponse(BaseResponse):
    """Detailed response for /video-info endpoint"""
    success: bool = False
    title: str = None
    description: str = None
    duration_seconds: int = None
    duration_string: str = None
    upload_date: str = None
    view_count: int = None
    concurrent_view_count: Optional[int] = None
    thumbnail: str = None
    language: Optional[str] = None
    has_subtitles: bool = False
    subtitle_languages: List[str] = None
    uploader_info: dict = None
    id: str = None
    webpage_url: str = None
    webpage_url_domain: str = None
    formats: List[dict] = None


@dataclass
class TelegramResponse(BaseResponse):
    """Response when uploading YouTube video to Telegram"""
    success: bool = False
    file_url: str = None
    message_id: int = None
    chat_username: str = None
    job_id: Optional[str] = None
    status: Optional[str] = None
    progress_url: Optional[str] = None
    warning: Optional[str] = None


@dataclass
class DownloadResult(BaseResponse):
    """Represents result of a local file download"""
    file_path: str = None
    file_size: int = None


@dataclass
class LiveStream(BaseResponse):
    """Represents a live stream (HLS/MP4) or streaming URL"""
    success: bool = True
    url: str = None
    warning: Optional[str] = None


@dataclass
class TryAfterResponse(BaseResponse):
    """Response when download is queued for background processing (HTTP 425)"""
    success: bool = False
    message: str = None
    try_after: Optional[int] = None
    job_id: Optional[str] = None
    status: Optional[str] = None
    progress_url: Optional[str] = None


@dataclass
class DownloadProgressResponse(BaseResponse):
    """Response for /download_progress endpoint"""
    success: bool = True
    job_id: str = None
    status: str = None
    percentage: Optional[float] = None
    video_id: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    timestamp: Optional[float] = None
    # downloading-status extra fields
    downloaded_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    speed: Optional[float] = None
    eta: Optional[float] = None
    message: Optional[str] = None


@dataclass
class ServerDownloadField(BaseResponse):
    """Kept for backward compatibility — now download_url is a top-level field"""
    download_url: str = None


@dataclass
class ServerResponse(Video):
    """Response for /youtube_to_server — includes a temporary download URL (valid 10 min)"""
    download_url: str = None
    key: Optional[str] = None
    job_id: Optional[str] = None
    status: Optional[str] = None
    progress_url: Optional[str] = None
    warning: Optional[str] = None
    _api_instance: Optional[YouTubeAPI] = None

    # Backward-compat: expose old-style .download attribute
    @property
    def download(self) -> Optional[ServerDownloadField]:
        if self.download_url:
            return ServerDownloadField(download_url=self.download_url)
        return None

    async def download_file(self, file_path: str, max_retries: Optional[int] = None,
                            retry_delay: Optional[float] = None) -> DownloadResult:
        if not self._api_instance:
            raise DownloadError("API instance not available")
        if not self.download_url:
            raise DownloadError("No download URL available")
        logger.info(f"Downloading file to {file_path} from server...")

        retries = max_retries if max_retries is not None else self._api_instance._max_retries
        delay = retry_delay if retry_delay is not None else self._api_instance._retry_delay

        return await self._api_instance.download_file(self.download_url, file_path, retries, delay)


@dataclass
class VideoSearchResult(BaseResponse):
    """Single video result from /search"""
    title: str = None
    videoId: str = None           # canonical field (replaces old `id`)
    link: str = None
    thumbnails: Optional[List[dict]] = None
    channel: Optional[dict] = None
    duration: Optional[dict] = None
    viewCount: Optional[dict] = None
    publishedTime: Optional[str] = None
    descriptionSnippet: Optional[Any] = None
    # Legacy fields kept for backward compatibility
    id: Optional[str] = None
    type: Optional[str] = None
    richThumbnail: Optional[dict] = None
    shelfTitle: Optional[str] = None


class BackgroundJobError(RequestError):
    """Raised when the API returns HTTP 425 — download is queued in background.

    Access the structured response via ``exception.response``.

    Example::

        try:
            result = await api.youtube_to_telegram("dQw4w9WgXcQ")
        except BackgroundJobError as e:
            print(f"Queued. Retry after {e.response.try_after}s")
            print(f"Track progress: {e.response.progress_url}")
            print(f"Job ID: {e.response.job_id}")
    """

    def __init__(self, response: TryAfterResponse):
        self.response = response
        super().__init__(
            response.message or "Download is queued for background processing",
            status_code=425,
        )


# ==================== API Client ====================

class YouTubeAPI(BaseRapidAPI):
    """
    API client wrapper for YouTube to Telegram/Server/Host endpoints.
    """

    BASE_URL = "https://youtube-to-telegram-uploader-api.p.rapidapi.com"
    DEFAULT_HOST = "youtube-to-telegram-uploader-api.p.rapidapi.com"

    def __init__(
        self, 
        api_key: str, 
        timeout: int = 300, 
        max_retries: int = 5, 
        retry_delay: float = 1.0,
        max_wait_time: int = 0, 
        cookies: Optional[str] = None, 
        config: Optional[APIConfig] = None,
        wait_for_background: bool = True,
        session: Optional[aiohttp.ClientSession] = None
    ):
        super().__init__(
            api_key=api_key, 
            timeout=timeout, 
            config=config, 
            session=session 
        )

        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._max_wait_time = max_wait_time
        self._cookies = cookies
        self._wait_for_background = wait_for_background


    def _parse_response_data(self, data: dict, response_class):
        """Helper to convert API response into dataclass objects"""

        parsed_data = data.copy()

        if response_class == ServerResponse:
            parsed_data['_api_instance'] = self

        elif response_class == VideoSearchResult:
            # Normalise: `videoId` is canonical, keep `id` for backward compat
            if 'videoId' in parsed_data and not parsed_data.get('id'):
                parsed_data['id'] = parsed_data['videoId']
            elif 'id' in parsed_data and not parsed_data.get('videoId'):
                parsed_data['videoId'] = parsed_data['id']

        # Handle common Video fields present in most responses
        if 'uploader' in parsed_data and isinstance(parsed_data['uploader'], dict):
            uploader_data = parsed_data['uploader'].copy()
            if 'channel_name' in uploader_data:
                uploader_data['name'] = uploader_data.pop('channel_name')
            if 'channel_url' in uploader_data:
                uploader_data['link'] = uploader_data.pop('channel_url')
            parsed_data['uploader'] = Channel(**uploader_data)

        # Normalize tags to list
        if 'tags' in parsed_data and not isinstance(parsed_data['tags'], list):
            if parsed_data['tags'] is None:
                parsed_data['tags'] = []
            else:
                parsed_data['tags'] = [parsed_data['tags']] if isinstance(parsed_data['tags'], str) else []

        # Normalize subtitle_languages and formats for VideoInfoResponse
        if response_class == VideoInfoResponse:
            if 'subtitle_languages' in parsed_data and not isinstance(parsed_data['subtitle_languages'], list):
                parsed_data['subtitle_languages'] = (
                    [] if parsed_data['subtitle_languages'] is None
                    else [parsed_data['subtitle_languages']] if isinstance(parsed_data['subtitle_languages'], str)
                    else []
                )

            if 'formats' in parsed_data and not isinstance(parsed_data['formats'], list):
                parsed_data['formats'] = (
                    [] if parsed_data['formats'] is None
                    else [parsed_data['formats']] if isinstance(parsed_data['formats'], dict)
                    else []
                )

        try:
            return response_class(**parsed_data)
        except TypeError as e:
            logger.warning(f"Failed to create {response_class.__name__} with data: {parsed_data}")
            logger.warning(f"Error: {e}")
            safe_data = {}
            for field_name, field_info in response_class.__dataclass_fields__.items():
                safe_data[field_name] = parsed_data.get(field_name,
                                                        field_info.default if field_info.default is not None else None)
            return response_class(**safe_data)

    async def _wait_for_completion(self, job_id: Optional[str], try_after: int, endpoint: str) -> dict:
        """Poll /download_progress until the background job finishes, then return its result dict."""
        if not job_id:
            raise RequestError(
                "Background job queued but no job_id was provided to track it",
                endpoint=endpoint,
            )

        logger.info(f"[{endpoint}] Job {job_id} queued for background processing. Waiting {try_after}s...")
        await asyncio.sleep(try_after)

        poll_interval = 5   # seconds between each status check
        max_total_wait = 600  # stop polling after 10 minutes
        waited = try_after

        while waited < max_total_wait:
            progress = await self._request("download_progress", {"job_id": job_id})
            status = progress.get("status")

            if status == "completed":
                result = progress.get("result")
                if not result:
                    raise RequestError(
                        f"Job {job_id} completed but returned no result data",
                        endpoint=endpoint,
                    )
                logger.info(f"[{endpoint}] Job {job_id} completed successfully.")
                return result

            if status == "failed":
                raise RequestError(
                    progress.get("error") or "Background download failed",
                    endpoint=endpoint,
                )

            pct = progress.get("percentage")
            pct_str = f"{pct:.1f}%" if pct is not None else "..."
            logger.info(f"[{endpoint}] Job {job_id}: {status} ({pct_str}). Next check in {poll_interval}s...")
            await asyncio.sleep(poll_interval)
            waited += poll_interval

        raise RequestError(
            f"Background job {job_id} did not complete within {max_total_wait}s",
            endpoint=endpoint,
        )

    async def _request(self, endpoint: str, params: dict, extra_headers: Optional[dict] = None) -> dict:
        """Make an API request with error handling and try_after support"""
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        url = f"{self.BASE_URL}/{endpoint}"
        headers = self._get_headers()
        if self._cookies:
            headers["X-Cookies"] = self._cookies
        if extra_headers:
            headers.update(extra_headers)
        logger.info(f"Requesting: {url} with params: {params}")

        try:
            async with self._session.get(url, headers=headers, params=params) as response:
                if response.status in (401, 403):
                    raise AuthenticationError(
                        "Authentication failed",
                        status_code=response.status,
                        endpoint=endpoint
                    )

                if response.status == 425:
                    try:
                        data = await response.json()
                    except Exception:
                        text_response = await response.text()
                        data = {"success": False, "message": self._extract_error_message(text_response)}

                    if not self._wait_for_background:
                        raise BackgroundJobError(TryAfterResponse(
                            success=data.get("success", False),
                            message=data.get("message", "Download is queued for background processing"),
                            try_after=data.get("try_after"),
                            job_id=data.get("job_id"),
                            status=data.get("status"),
                            progress_url=data.get("progress_url"),
                        ))

                    return await self._wait_for_completion(
                        job_id=data.get("job_id"),
                        try_after=data.get("try_after", 30),
                        endpoint=endpoint,
                    )

                if response.status != 200:
                    error_text = await response.text()
                    clean_message = self._extract_error_message(error_text)
                    raise RequestError(
                        clean_message,
                        status_code=response.status,
                        endpoint=endpoint
                    )

                try:
                    data = await response.json()
                except Exception:
                    text_response = await response.text()
                    raise RequestError(
                        text_response or "Empty response from API",
                        status_code=response.status,
                        endpoint=endpoint
                    )

                if not isinstance(data, dict):
                    return data

                try_after = data.get("try_after")

                if try_after is not None:
                    if try_after > self._max_wait_time:
                        error_msg = data.get("message", f"Download delay required: {try_after} seconds")
                        raise RequestError(
                            error_msg,
                            status_code=response.status,
                            endpoint=endpoint
                        )

                    logger.info(f"Waiting {try_after} seconds before retry...")
                    await asyncio.sleep(try_after)

                    logger.info("Retrying request after waiting...")
                    async with self._session.get(url, headers=headers, params=params) as retry_response:  # headers already has extra_headers merged
                        if retry_response.status != 200:
                            error_text = await retry_response.text()
                            clean_message = self._extract_error_message(error_text)
                            raise RequestError(
                                clean_message,
                                status_code=retry_response.status,
                                endpoint=endpoint
                            )

                        data = await retry_response.json()

                if data.get("success", False) is False:
                    error_msg = data.get("message") or data.get("messages", "Unknown error")
                    raise RequestError(
                        error_msg,
                        endpoint=endpoint
                    )

                return data

        except (AuthenticationError, RequestError):
            raise
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            raise RequestError(
                f"Network error: {str(e)}",
                endpoint=endpoint,
                original_error=e
            )

    async def download_file(self, url: str, file_path: str, max_retries: Optional[int] = None,
                            retry_delay: Optional[float] = None) -> DownloadResult:
        """Download file from URL to local path with retry support"""
        from ..utils import download_file as _download_file

        retries = max_retries if max_retries is not None else self._max_retries
        delay = retry_delay if retry_delay is not None else self._retry_delay

        result_path = await _download_file(
            url=url, output_path=file_path,
            max_retries=retries, retry_delay=delay,
            show_progress=True, session=self._session
        )
        if result_path is None:
            raise DownloadError(f"Failed to download from {url}")
        return DownloadResult(file_path=result_path, file_size=Path(result_path).stat().st_size)

    def _extract_error_message(self, error_text: str) -> str:
        """Extract error message from API error response"""
        try:
            error_data = json.loads(error_text)
            if isinstance(error_data, dict):
                if "message" in error_data:
                    return error_data["message"]
                if "messages" in error_data:
                    return error_data["messages"]
        except Exception:
            pass
        return error_text

    # ==================== Public Methods ====================

    @with_retry(max_attempts=3, delay=1.0)
    async def video_info(self, url: str) -> VideoInfoResponse:
        """
        Get detailed metadata for any supported video URL.

        Supports: YouTube, TikTok, Instagram, Facebook, SoundCloud, and more.
        Results are cached for 1 hour.

        Args:
            url: Full video URL from any supported site
        """
        data = await self._request("video_info", {"video_url": url})
        return self._parse_response_data(data, VideoInfoResponse)

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_to_server(
        self,
        video_id: str,
        file_format: str = "mp4",
        quality: str = "best",
        webhook_url: Optional[str] = None,
        format: Optional[str] = None,  # Deprecated: use file_format
    ) -> ServerResponse:
        """
        Download YouTube video to server and get a temporary download URL (valid 10 min).

        Args:
            video_id: YouTube video ID (10-12 chars)
            file_format: Output format — "mp4" (video), "m4a" (audio), "mp3" (audio). Default: "mp4"
            quality: Output quality — "best", "worst", "1080p", "720p", "480p", "360p", "256k", "128k". Default: "best"
            webhook_url: Optional URL to receive POST callback when download completes
            format: Deprecated. Use file_format instead ("audio" → "m4a", "video" → "mp4")
        """
        params = {"video_id": video_id, "file_format": file_format, "quality": quality}
        if format is not None:
            params["format"] = format
        if webhook_url is not None:
            params["webhook_url"] = webhook_url
        data = await self._request("youtube_to_server", params)
        return self._parse_response_data(data, ServerResponse)

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_to_telegram(
        self,
        video_id: str,
        file_format: str = "m4a",
        quality: str = "best",
        webhook_url: Optional[str] = None,
        format: Optional[str] = None,  # Deprecated: use file_format
    ) -> TelegramResponse:
        """
        Upload YouTube video/audio to Telegram and get the message link.

        Args:
            video_id: YouTube video ID (10-12 chars)
            file_format: Output format — "m4a" (audio, cached), "mp3" (audio, not cached), "mp4" (video, cached). Default: "m4a"
            quality: Output quality — "best", "worst", "1080p", "720p", "480p", "360p", "256k", "128k". Default: "best"
            webhook_url: Optional URL to receive POST callback when upload completes
            format: Deprecated. Use file_format instead ("audio" → "m4a", "video" → "mp4")
        """
        params = {"video_id": video_id, "file_format": file_format, "quality": quality}
        if format is not None:
            params["format"] = format
        if webhook_url is not None:
            params["webhook_url"] = webhook_url
        data = await self._request("youtube_to_telegram", params)
        return self._parse_response_data(data, TelegramResponse)

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_live_hls(self, video_id: str, audio_only: bool = False) -> LiveStream:
        """
        Get HLS live stream playback URL for a YouTube Live (valid 10 min).

        Args:
            video_id: YouTube live video ID (10-12 chars)
            audio_only: Stream audio track only (no video). Default: False
        """
        data = await self._request(
            "youtube_live_hls",
            {"video_id": video_id, "audio_only": audio_only},
        )
        return LiveStream(success=data.get("success", True), url=data.get("url"), warning=data.get("warning"))

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_live_mp4(self, video_id: str, audio_only: bool = False) -> LiveStream:
        """
        Get MP4 live stream playback URL for a YouTube Live (valid 10 min).

        Args:
            video_id: YouTube live video ID (10-12 chars)
            audio_only: Stream audio track only (no video). Default: False
        """
        data = await self._request(
            "youtube_live_mp4",
            {"video_id": video_id, "audio_only": audio_only},
        )
        return LiveStream(success=data.get("success", True), url=data.get("url"), warning=data.get("warning"))

    @with_retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, limit: int = 10) -> List[VideoSearchResult]:
        """
        Search YouTube videos.

        Args:
            query: Search query string
            limit: Maximum number of results (1-50). Default: 10
        """
        data = await self._request("search", {"query": query, "limit": limit})
        results = []
        if isinstance(data, list):
            for item in data:
                results.append(self._parse_response_data(item, VideoSearchResult))
        return results

    @with_retry(max_attempts=3, delay=1.0)
    async def youtube_video_stream(
        self,
        video_id: str,
        file_format: str = "mp4",
        quality: str = "best",
        format: Optional[str] = None,  # Deprecated: use file_format
    ) -> LiveStream:
        """
        Get a temporary streaming URL for a YouTube video (valid 10 min).

        Args:
            video_id: YouTube video ID (10-12 chars)
            file_format: Output format — "mp4" (video), "m4a" (audio), "mp3" (audio). Default: "mp4"
            quality: Output quality — "best", "worst", "1080p", "720p", etc. Default: "best"
            format: Deprecated. Use file_format instead
        """
        params = {"video_id": video_id, "file_format": file_format, "quality": quality}
        if format is not None:
            params["format"] = format
        data = await self._request("youtube_video_stream", params)
        return LiveStream(success=data.get("success", True), url=data.get("url"), warning=data.get("warning"))

    @with_retry(max_attempts=3, delay=1.0)
    async def download_progress(self, job_id: str) -> DownloadProgressResponse:
        """
        Check the progress of a background download/upload job.

        Args:
            job_id: Job ID returned from youtube_to_server or youtube_to_telegram

        Returns:
            DownloadProgressResponse with status, percentage, and result/error when done
        """
        data = await self._request("download_progress", {"job_id": job_id})
        return self._parse_response_data(data, DownloadProgressResponse)
