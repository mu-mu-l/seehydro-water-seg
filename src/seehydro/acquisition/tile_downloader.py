"""在线地图瓦片下载器（仅供科研用途）."""

import io
import math
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from loguru import logger
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from tqdm import tqdm

TILE_PROVIDERS: dict[str, dict] = {
    "tianditu_img": {
        "url": (
            "https://t{s}.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
            "&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILECOL={x}&TILEROW={y}"
            "&TILEMATRIX={z}&tk={key}"
        ),
        "subdomains": ["0", "1", "2", "3", "4", "5", "6", "7"],
        "requires_key": True,
    },
    "google_satellite": {
        "url": "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "subdomains": ["0", "1", "2", "3"],
        "requires_key": False,
    },
}


def lon_lat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """经纬度转瓦片坐标."""
    n = 2**zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat)
    y = int(math.floor((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n))
    return x, y


def tile_to_lon_lat(x: int, y: int, zoom: int) -> tuple[float, float]:
    """瓦片坐标转经纬度（左上角）."""
    n = 2**zoom
    lon = (x / n) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


class TileDownloader:
    def __init__(
        self,
        provider: str = "google_satellite",
        api_key: str | None = None,
        request_interval: float = 0.35,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        if provider not in TILE_PROVIDERS:
            raise ValueError(f"未知瓦片服务商: {provider}，可选: {list(TILE_PROVIDERS.keys())}")

        self.provider: str = provider
        self.api_key: str | None = api_key
        self._config: dict = TILE_PROVIDERS[provider]
        self.request_interval = max(0.0, request_interval)
        self.max_backoff_seconds = max(1.0, max_backoff_seconds)
        self._last_request_ts = 0.0

        if bool(self._config["requires_key"]) and self.api_key is None:
            raise ValueError(f"服务商 '{provider}' 需要 API Key，请通过 api_key 参数传入。")

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SeeHydro/0.1 (Research)"})

    def download_tiles(
        self,
        bounds: tuple[float, float, float, float],
        zoom: int = 18,
        output_dir: str | Path = "data/cache/tiles",
    ) -> Path:
        """下载指定范围的瓦片并拼接为GeoTIFF.

        Args:
            bounds: (west, south, east, north) 经纬度范围。
            zoom: 瓦片缩放级别。
            output_dir: GeoTIFF 输出目录。

        Returns:
            生成的 GeoTIFF 文件路径。
        """
        west, south, east, north = bounds

        x_min, y_min = lon_lat_to_tile(west, north, zoom)
        x_max, y_max = lon_lat_to_tile(east, south, zoom)

        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tile_cols = x_max - x_min + 1
        tile_rows = y_max - y_min + 1
        total_width = tile_cols * 256
        total_height = tile_rows * 256

        mosaic = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        coords: list[tuple[int, int]] = [
            (x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)
        ]
        success_count = 0
        failed_count = 0

        for x, y in tqdm(coords, desc="Downloading tiles", unit="tile"):
            tile = self._download_single_tile(zoom, x, y)
            x_offset = (x - x_min) * 256
            y_offset = (y - y_min) * 256
            if tile is not None:
                mosaic[y_offset : y_offset + 256, x_offset : x_offset + 256, :] = tile
                success_count += 1
            else:
                failed_count += 1

        left, top = tile_to_lon_lat(x_min, y_min, zoom)
        right, bottom = tile_to_lon_lat(x_max + 1, y_max + 1, zoom)

        transform = from_bounds(left, bottom, right, top, total_width, total_height)
        tif_name = f"{zoom}_{x_min}_{y_min}_{x_max}_{y_max}.tif"
        tif_path = output_path / tif_name

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=total_height,
            width=total_width,
            count=3,
            dtype=np.uint8,
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(mosaic[:, :, 0], 1)
            dst.write(mosaic[:, :, 1], 2)
            dst.write(mosaic[:, :, 2], 3)

        logger.info(
            "瓦片下载完成: provider={}, zoom={}, 总计={}, 成功={}, 失败={}, 输出={}",
            self.provider,
            zoom,
            len(coords),
            success_count,
            failed_count,
            tif_path,
        )
        return tif_path

    def _download_single_tile(self, z: int, x: int, y: int, max_retries: int = 3) -> np.ndarray | None:
        """下载单个瓦片，返回256x256x3的numpy数组，失败返回None."""
        url = self._build_url(z, x, y)

        for attempt in range(1, max_retries + 1):
            try:
                self._respect_request_interval()
                response = self._session.get(url, timeout=10)
                self._last_request_ts = time.monotonic()
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    tile = np.array(image, dtype=np.uint8)
                    if tile.shape == (256, 256, 3):
                        return tile
                    logger.warning("瓦片尺寸异常 z={}, x={}, y={}, shape={}", z, x, y, tile.shape)
                    time.sleep(self._compute_backoff(attempt))
                    continue
                if response.status_code == 429:
                    sleep_seconds = self._compute_retry_delay(response, attempt)
                    logger.warning(
                        "瓦片请求触发限流 z={}, x={}, y={}, attempt={}, retry_in={:.2f}s",
                        z, x, y, attempt, sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                    continue
                else:
                    logger.warning(
                        "瓦片请求返回非200状态 z={}, x={}, y={}, status={}, attempt={}",
                        z, x, y, response.status_code, attempt,
                    )
                    time.sleep(self._compute_backoff(attempt))
                    continue
            except Exception as exc:
                logger.warning("瓦片请求异常 z={}, x={}, y={}, attempt={}, error={}", z, x, y, attempt, exc)

            time.sleep(self._compute_backoff(attempt))

        logger.warning("瓦片下载失败（已重试{}次） z={}, x={}, y={}", max_retries, z, x, y)
        return None

    def _build_url(self, z: int, x: int, y: int) -> str:
        """构建瓦片URL."""
        subdomain = random.choice(self._config["subdomains"])
        template: str = str(self._config["url"])
        return template.format(z=z, x=x, y=y, s=subdomain, key=self.api_key or "")

    def _respect_request_interval(self) -> None:
        """在两次请求之间维持最小间隔，降低触发服务端限流的概率。"""
        if self.request_interval <= 0:
            return

        elapsed = time.monotonic() - self._last_request_ts
        remaining = self.request_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _compute_backoff(self, attempt: int) -> float:
        """指数退避并加入轻微抖动，避免重复碰撞限流窗口。"""
        base_seconds = min(self.max_backoff_seconds, float(2 ** (attempt - 1)))
        return base_seconds + random.uniform(0.1, 0.5)

    def _compute_retry_delay(self, response: requests.Response, attempt: int) -> float:
        """优先尊重 Retry-After，其次使用指数退避。"""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                retry_after_seconds = float(retry_after)
                return min(self.max_backoff_seconds, max(1.0, retry_after_seconds))
            except ValueError:
                logger.debug("忽略无法解析的 Retry-After: {}", retry_after)

        return self._compute_backoff(attempt)
