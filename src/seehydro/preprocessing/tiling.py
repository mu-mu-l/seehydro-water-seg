"""影像切片工具."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import rasterio
from loguru import logger
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform


@dataclass(slots=True)
class TileInfo:
    """切片元信息."""

    tile_path: Path
    source_path: Path
    row: int
    col: int
    x_offset: int
    y_offset: int
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    crs: str | None = None

    def to_record(self) -> dict[str, object]:
        """将切片信息转换为可序列化字典."""
        record = asdict(self)
        record["tile_path"] = str(self.tile_path)
        record["source_path"] = str(self.source_path)
        record["bounds"] = ",".join(str(v) for v in self.bounds)
        return record

    @classmethod
    def from_record(cls, record: dict[str, object]) -> "TileInfo":
        """从序列化字典恢复切片信息."""
        bounds_raw = str(record["bounds"]).split(",")
        bounds = tuple(float(v) for v in bounds_raw)
        return cls(
            tile_path=Path(str(record["tile_path"])),
            source_path=Path(str(record["source_path"])),
            row=int(record["row"]),
            col=int(record["col"]),
            x_offset=int(record["x_offset"]),
            y_offset=int(record["y_offset"]),
            width=int(record["width"]),
            height=int(record["height"]),
            bounds=bounds,
            crs=None if pd.isna(record.get("crs")) else str(record.get("crs")),
        )


class TileGenerator:
    """滑窗切片生成器."""

    def __init__(self, tile_size: int = 512, overlap: float = 0.25):
        if tile_size <= 0:
            raise ValueError(f"tile_size 必须大于 0，当前值: {tile_size}")
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap 必须在 [0, 1) 范围内，当前值: {overlap}")

        self.tile_size = tile_size
        self.overlap = overlap
        step = int(round(tile_size * (1 - overlap)))
        self.step = max(1, step)

    def generate_tiles(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        prefix: str = "tile",
        min_valid_ratio: float = 0.5,
    ) -> list[TileInfo]:
        """对单幅栅格影像进行滑窗切片."""
        if not (0 <= min_valid_ratio <= 1):
            raise ValueError(f"min_valid_ratio 必须在 [0, 1] 范围内，当前值: {min_valid_ratio}")

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tile_infos: list[TileInfo] = []

        with rasterio.open(image_path) as src:
            row_offsets = self._build_offsets(src.height)
            col_offsets = self._build_offsets(src.width)

            for row_idx, y_off in enumerate(row_offsets):
                for col_idx, x_off in enumerate(col_offsets):
                    width = min(self.tile_size, src.width - x_off)
                    height = min(self.tile_size, src.height - y_off)
                    if width <= 0 or height <= 0:
                        continue

                    window = Window(col_off=x_off, row_off=y_off, width=width, height=height)
                    data = src.read(window=window)

                    if self._compute_valid_ratio(data, src.nodata) < min_valid_ratio:
                        continue

                    tile_path = output_dir / f"{prefix}_r{row_idx:04d}_c{col_idx:04d}.tif"
                    profile = src.profile.copy()
                    profile.update(
                        width=width,
                        height=height,
                        transform=window_transform(window, src.transform),
                    )

                    with rasterio.open(tile_path, "w", **profile) as dst:
                        dst.write(data)

                    info = TileInfo(
                        tile_path=tile_path,
                        source_path=image_path,
                        row=row_idx,
                        col=col_idx,
                        x_offset=int(x_off),
                        y_offset=int(y_off),
                        width=int(width),
                        height=int(height),
                        bounds=tuple(float(v) for v in window_bounds(window, src.transform)),
                        crs=None if src.crs is None else str(src.crs),
                    )
                    tile_infos.append(info)

        logger.info("切片完成: {} -> {} 个切片", image_path, len(tile_infos))
        return tile_infos

    def reassemble(
        self,
        tiles: dict,
        tile_infos,
        output_path,
        original_profile,
        merge_strategy: str = "mean",
    ) -> Path:
        """重组切片结果.

        当前仅保留 API 占位，后续可按预测掩膜重建整图。
        """
        raise NotImplementedError("reassemble 尚未实现")

    def save_tile_index(self, tile_infos, output_path) -> Path:
        """保存切片索引 CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = [info.to_record() for info in tile_infos]
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        logger.info("切片索引已保存: {}", output_path)
        return output_path

    def load_tile_index(self, csv_path: str | Path) -> list[TileInfo]:
        """读取切片索引 CSV."""
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        return [TileInfo.from_record(record) for record in df.to_dict(orient="records")]

    def _build_offsets(self, length: int) -> list[int]:
        """构建滑窗起始偏移列表，保证覆盖尾部区域."""
        if length <= self.tile_size:
            return [0]

        offsets = list(range(0, length - self.tile_size + 1, self.step))
        last_offset = length - self.tile_size
        if offsets[-1] != last_offset:
            offsets.append(last_offset)
        return offsets

    @staticmethod
    def _compute_valid_ratio(data, nodata) -> float:
        """估算切片有效像素占比."""
        if data.size == 0:
            return 0.0
        if nodata is None:
            return 1.0

        valid_mask = data != nodata
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.any(axis=0)
        return float(valid_mask.mean())
