"""高分辨率影像扫描检索与在线地图瓦片下载拼接模块。"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from loguru import logger
from PIL import Image
from tqdm import tqdm

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
except ImportError as e:
    raise ImportError(f"请安装 rasterio: {e}") from e

try:
    from shapely.geometry import box, shape
    from shapely.strtree import STRtree
except ImportError as e:
    raise ImportError(f"请安装 shapely: {e}") from e

try:
    import cv2
except ImportError as e:
    raise ImportError(f"请安装 opencv-python: {e}") from e


class HighResManager:
    """管理本地高分辨率遥感影像，支持扫描、空间索引和范围查询。"""

    SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".img", ".jp2"}

    def __init__(self, data_dir: str | Path = "data/highres") -> None:
        """初始化影像管理器。

        Args:
            data_dir: 高分辨率影像根目录。
        """
        self.data_dir = Path(data_dir)
        self._images: list[dict[str, Any]] = []
        self._spatial_index: STRtree | None = None
        self._index_geometries: list[Any] = []
        self._geometry_id_to_index: dict[int, int] = {}

    def scan(self) -> list[dict[str, Any]]:
        """递归扫描目录并读取影像元数据。

        Returns:
            影像信息列表，每项包含 path、bounds、crs、resolution、size_mb。
        """
        self._images = []

        if not self.data_dir.exists():
            logger.warning("数据目录不存在: {}", self.data_dir)
            return self._images

        files = [
            p
            for p in self.data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        for path in files:
            try:
                with rasterio.open(path) as ds:
                    bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
                    crs_obj = ds.crs
                    if crs_obj is None:
                        crs: int | str | None = None
                    else:
                        crs = crs_obj.to_epsg() if crs_obj.to_epsg() is not None else str(crs_obj)
                    resolution = (abs(ds.transform.a), abs(ds.transform.e))
                    size_mb = path.stat().st_size / (1024**2)
                    self._images.append({
                        "path": path,
                        "bounds": bounds,
                        "crs": crs,
                        "resolution": resolution,
                        "size_mb": size_mb,
                    })
            except Exception as e:  # noqa: BLE001
                logger.warning("读取影像失败，已跳过: {} | 错误: {}", path, e)

        logger.info("影像扫描完成: {} 个文件", len(self._images))
        return self._images

    def build_spatial_index(self) -> None:
        """基于影像 bounds 构建 STRtree 空间索引。"""
        if not self._images:
            self.scan()

        geometries = []
        self._geometry_id_to_index = {}

        for idx, img in enumerate(self._images):
            bounds = img["bounds"]
            geom = box(*bounds)
            geometries.append(geom)
            self._geometry_id_to_index[id(geom)] = idx

        self._index_geometries = geometries
        if geometries:
            self._spatial_index = STRtree(geometries)
            logger.info("空间索引构建完成: {} 个要素", len(geometries))
        else:
            self._spatial_index = None
            logger.warning("空间索引构建跳过：未发现可用影像")

    def query_by_bounds(self, bounds: tuple[float, float, float, float]) -> list[Path]:
        """按包围盒查询相交影像。

        Args:
            bounds: 查询范围 (minx, miny, maxx, maxy)。

        Returns:
            相交影像路径列表。
        """
        if self._spatial_index is None:
            self.build_spatial_index()

        if self._spatial_index is None:
            return []

        query_geom = box(*bounds)
        raw_result = self._spatial_index.query(query_geom, predicate="intersects")
        indices = self._normalize_query_result(raw_result)
        return [self._images[i]["path"] for i in indices]

    def query_by_geometry(self, geometry: Any) -> list[Path]:
        """按几何对象查询相交影像。

        Args:
            geometry: shapely 几何对象或 GeoJSON dict。

        Returns:
            相交影像路径列表。
        """
        if isinstance(geometry, dict):
            geometry = shape(geometry)

        if self._spatial_index is None:
            self.build_spatial_index()

        if self._spatial_index is None:
            return []

        raw_result = self._spatial_index.query(geometry, predicate="intersects")
        indices = self._normalize_query_result(raw_result)
        return [self._images[i]["path"] for i in indices]

    def get_info(self) -> dict[str, Any]:
        """获取影像目录统计信息。

        Returns:
            统计字典，包含 file_count、total_size_mb、resolution_range、crs_list。
        """
        if not self._images:
            self.scan()

        file_count = len(self._images)
        total_size_mb = float(sum(item["size_mb"] for item in self._images))

        if file_count > 0:
            x_resolutions = [float(item["resolution"][0]) for item in self._images]
            min_res = float(min(x_resolutions))
            max_res = float(max(x_resolutions))
        else:
            min_res = 0.0
            max_res = 0.0

        crs_list = sorted({item["crs"] for item in self._images if item["crs"] is not None}, key=str)

        return {
            "file_count": file_count,
            "total_size_mb": total_size_mb,
            "resolution_range": {"min": min_res, "max": max_res},
            "crs_list": crs_list,
        }

    def _normalize_query_result(self, raw_result: Any) -> list[int]:
        """将 STRtree 查询结果规范化为影像索引列表。

        Args:
            raw_result: STRtree.query() 的原始返回值。

        Returns:
            去重排序后的整数索引列表。
        """
        result_indices: list[int] = []

        if raw_result is None:
            return result_indices

        # shapely 2.x 返回 numpy 整数索引数组
        if isinstance(raw_result, np.ndarray) and np.issubdtype(raw_result.dtype, np.integer):
            return sorted({int(i) for i in raw_result.tolist()})

        # 兼容旧版本返回几何对象列表
        if isinstance(raw_result, (list, tuple, np.ndarray)):
            for item in raw_result:
                if isinstance(item, (int, np.integer)):
                    result_indices.append(int(item))
                else:
                    idx = self._geometry_id_to_index.get(id(item))
                    if idx is not None:
                        result_indices.append(idx)

        return sorted(set(result_indices))


class TileDownloader:
    """在线地图瓦片下载器，支持按范围抓取并拼接为 GeoTIFF。

    支持天地图等在线地图服务的瓦片下载。
    注意：仅供科研用途。
    """

    PROVIDERS = {
        "tianditu": (
            "https://t{server}.tianditu.gov.cn/img_w/wmts?"
            "SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&"
            "TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk={api_key}"
        ),
        "tianditu_label": (
            "https://t{server}.tianditu.gov.cn/cia_w/wmts?"
            "SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=cia&STYLE=default&"
            "TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk={api_key}"
        ),
    }
    TILE_SIZE = 256

    def __init__(
        self,
        provider: str = "tianditu",
        api_key: str | None = None,
        request_interval: float = 0.35,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        """初始化瓦片下载器。

        Args:
            provider: 预置服务商名称（"tianditu"/"tianditu_label"）或自定义 URL 模板。
            api_key: 服务 API 密钥。
        """
        self.provider = provider
        self.api_key = api_key
        self.url_template = self.PROVIDERS.get(provider, provider)
        self.request_interval = max(0.0, request_interval)
        self.max_backoff_seconds = max(1.0, max_backoff_seconds)
        self._last_request_ts = 0.0
        self._session = requests.Session()

    def download_tiles(
        self,
        bounds: tuple[float, float, float, float],
        zoom: int = 18,
        output_dir: str | Path = "data/cache/tiles",
    ) -> Path:
        """下载指定范围内的瓦片并拼接为 GeoTIFF。

        Args:
            bounds: 地理范围 (lon_min, lat_min, lon_max, lat_max)，EPSG:4326。
            zoom: 缩放级别。
            output_dir: 瓦片缓存目录。

        Returns:
            拼接后的 GeoTIFF 路径。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        x_min, y_min, x_max, y_max = self._tile_bounds(zoom, bounds)
        total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
        logger.info(
            "开始下载瓦片: zoom={}, x=[{}, {}], y=[{}, {}], 总数={}",
            zoom, x_min, x_max, y_min, y_max, total_tiles,
        )

        with tqdm(total=total_tiles, desc="下载瓦片", unit="tile") as pbar:
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    tile_path = output_dir / f"{zoom}_{x}_{y}.png"
                    if not tile_path.exists():
                        arr = self._download_single_tile(zoom, x, y)
                        if arr is not None:
                            img = Image.fromarray(arr, mode="RGB")
                            img.save(tile_path, format="PNG")
                    pbar.update(1)

        output_path = output_dir.parent / f"merged_{zoom}_{bounds[0]:.4f}_{bounds[1]:.4f}.tif"
        return self.merge_tiles(output_dir, output_path, zoom=zoom)

    def _tile_bounds(
        self,
        zoom: int,
        bounds: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        """计算地理范围对应的瓦片索引范围。

        Args:
            zoom: 缩放级别。
            bounds: 地理范围 (lon_min, lat_min, lon_max, lat_max)，EPSG:4326。

        Returns:
            (x_min, y_min, x_max, y_max)，y_min 对应北边（纬度较大）。
        """
        lon_min, lat_min, lon_max, lat_max = bounds

        # Web Mercator 纬度有效范围裁剪
        lat_min = max(min(lat_min, 85.05112878), -85.05112878)
        lat_max = max(min(lat_max, 85.05112878), -85.05112878)

        n = 2**zoom

        def lon_to_tile_x(lon: float) -> int:
            return math.floor((lon + 180.0) / 360.0 * n)

        def lat_to_tile_y(lat: float) -> int:
            return math.floor(
                (
                    1.0
                    - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat)))
                    / math.pi
                )
                / 2.0
                * n
            )

        x1 = lon_to_tile_x(lon_min)
        x2 = lon_to_tile_x(lon_max)
        # 北纬（lat_max）对应较小的 y 值
        y_north = lat_to_tile_y(lat_max)
        y_south = lat_to_tile_y(lat_min)

        x_min = max(0, min(x1, x2))
        x_max = min(n - 1, max(x1, x2))
        y_min = max(0, min(y_north, y_south))
        y_max = min(n - 1, max(y_north, y_south))

        return x_min, y_min, x_max, y_max

    def _download_single_tile(self, z: int, x: int, y: int) -> np.ndarray | None:
        """下载单个瓦片并解码为 RGB 数组。

        Args:
            z: 缩放级别。
            x: 瓦片列号。
            y: 瓦片行号。

        Returns:
            shape=(H, W, 3) 的 uint8 RGB 数组，全部失败时返回 None。
        """
        for attempt in range(1, 4):
            server = random.randint(0, 7)
            url = self.url_template.format(
                server=server,
                z=z,
                x=x,
                y=y,
                api_key=self.api_key or "",
            )
            try:
                self._respect_request_interval()
                response = self._session.get(url, timeout=10)
                self._last_request_ts = time.monotonic()
                if response.status_code == 429:
                    sleep_seconds = self._compute_retry_delay(response, attempt)
                    logger.warning(
                        "瓦片下载触发限流: z={}, x={}, y={}, attempt={}, retry_in={:.2f}s",
                        z, x, y, attempt, sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                    continue
                if response.status_code != 200:
                    logger.warning(
                        "瓦片下载失败(status={}): z={}, x={}, y={}, attempt={}",
                        response.status_code, z, x, y, attempt,
                    )
                    time.sleep(self._compute_backoff(attempt))
                    continue

                buf = np.frombuffer(response.content, np.uint8)
                img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    logger.warning("瓦片解码失败: z={}, x={}, y={}, attempt={}", z, x, y, attempt)
                    time.sleep(self._compute_backoff(attempt))
                    continue

                img_rgb: np.ndarray = img_bgr[:, :, ::-1]
                return img_rgb.astype(np.uint8, copy=False)

            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "瓦片下载异常: z={}, x={}, y={}, attempt={}, error={}",
                    z, x, y, attempt, e,
                )
                time.sleep(self._compute_backoff(attempt))

        return None

    def _respect_request_interval(self) -> None:
        """在两次请求间增加最小间隔，降低瞬时请求密度。"""
        if self.request_interval <= 0:
            return

        elapsed = time.monotonic() - self._last_request_ts
        remaining = self.request_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _compute_backoff(self, attempt: int) -> float:
        """指数退避并加入抖动，避免重复命中同一限流窗口。"""
        base_seconds = min(self.max_backoff_seconds, float(2 ** (attempt - 1)))
        return base_seconds + random.uniform(0.1, 0.5)

    def _compute_retry_delay(self, response: requests.Response, attempt: int) -> float:
        """优先读取 Retry-After，缺失时退化为指数退避。"""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                retry_after_seconds = float(retry_after)
                return min(self.max_backoff_seconds, max(1.0, retry_after_seconds))
            except ValueError:
                logger.debug("忽略无法解析的 Retry-After: {}", retry_after)

        return self._compute_backoff(attempt)

    def merge_tiles(
        self,
        tile_dir: Path,
        output_path: Path,
        zoom: int | None = None,
    ) -> Path:
        """将缓存目录中的 PNG 瓦片拼接并写出 GeoTIFF（EPSG:3857）。

        Args:
            tile_dir: 存放 PNG 瓦片的目录，文件命名格式为 "{z}_{x}_{y}.png"。
            output_path: 输出 GeoTIFF 路径。
            zoom: 缩放级别，若为 None 则从第一个瓦片文件名中解析。

        Returns:
            输出 GeoTIFF 路径。
        """
        tile_dir = Path(tile_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tile_files = sorted(tile_dir.glob("*.png"))
        if not tile_files:
            raise FileNotFoundError(f"未在目录中找到 PNG 瓦片: {tile_dir}")

        tile_records: list[tuple[int, int, int, Path]] = []
        for fp in tile_files:
            parts = fp.stem.split("_")
            if len(parts) != 3:
                logger.warning("跳过不符合命名规则的瓦片: {}", fp.name)
                continue
            try:
                z_val, tx, ty = int(parts[0]), int(parts[1]), int(parts[2])
                tile_records.append((z_val, tx, ty, fp))
            except ValueError:
                logger.warning("跳过无法解析索引的瓦片: {}", fp.name)

        if not tile_records:
            raise ValueError(f"没有可用于拼接的瓦片: {tile_dir}")

        if zoom is None:
            zoom = tile_records[0][0]

        filtered = [rec for rec in tile_records if rec[0] == zoom]
        if not filtered:
            raise ValueError(f"目录中不存在 zoom={zoom} 的瓦片: {tile_dir}")

        xs = [rec[1] for rec in filtered]
        ys = [rec[2] for rec in filtered]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        total_width = (x_max - x_min + 1) * self.TILE_SIZE
        total_height = (y_max - y_min + 1) * self.TILE_SIZE
        canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        for _, tx, ty, fp in tqdm(filtered, desc="合并瓦片", unit="tile"):
            col = tx - x_min
            row = ty - y_min
            x0 = col * self.TILE_SIZE
            y0 = row * self.TILE_SIZE

            with Image.open(fp) as img:
                rgb = img.convert("RGB")
                arr = np.array(rgb, dtype=np.uint8)

            if arr.shape[0] != self.TILE_SIZE or arr.shape[1] != self.TILE_SIZE:
                # TODO: 若存在非标准尺寸瓦片，可按需决定重采样策略
                logger.warning("瓦片尺寸非 256x256，已跳过: {}", fp.name)
                continue

            canvas[y0: y0 + self.TILE_SIZE, x0: x0 + self.TILE_SIZE, :] = arr

        # Web Mercator 地理坐标计算（单位：米）
        r = 6378137.0
        n = 2**zoom
        world_merc = 2.0 * math.pi * r

        geo_left = x_min / n * world_merc - math.pi * r
        geo_right = (x_max + 1) / n * world_merc - math.pi * r
        geo_top = math.pi * r - y_min / n * world_merc
        geo_bottom = math.pi * r - (y_max + 1) / n * world_merc

        transform = from_bounds(geo_left, geo_bottom, geo_right, geo_top, total_width, total_height)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=total_height,
            width=total_width,
            count=3,
            dtype=np.uint8,
            crs=CRS.from_epsg(3857),
            transform=transform,
        ) as dst:
            dst.write(canvas[:, :, 0], 1)
            dst.write(canvas[:, :, 1], 2)
            dst.write(canvas[:, :, 2], 3)

        logger.info("瓦片拼接完成: {}", output_path)
        return output_path
