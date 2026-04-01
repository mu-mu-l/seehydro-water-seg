# SeeHydro - Codex 上下文文件

## ⚠️ 编码强制规则（Codex 必读）

1. **严格使用下面列出的 import 路径**，不要尝试其他路径
2. **严格按 API 签名调用**，不要用 try/except 探测多种签名
3. **不要发明不存在的 API**，只用本文件中列出的函数和类
4. **不要写兼容层/适配器**，直接按签名调用
5. **不要用 importlib.import_module 动态导入**，用固定 import

## 项目概述
南水北调中线渠道水面识别与辅助分析项目。当前最成熟方向是从卫星/底图影像中识别渠道水面，并输出估算水面宽度等辅助结果。

## 包结构
```
src/seehydro/           # 主包，import: from seehydro.xxx import yyy
├── cli.py              # CLI入口, app = typer.Typer()
├── acquisition/        # 数据获取
│   ├── route.py        # RouteDataLoader, load_route()
│   ├── gee.py          # GEEDownloader
│   ├── highres.py      # HighResManager, TileDownloader
│   └── tile_downloader.py  # TileDownloader (瓦片下载)
├── preprocessing/      # 预处理
│   ├── clip.py         # clip_raster_by_geometry(), clip_along_route(), batch_clip()
│   ├── enhance.py      # compute_ndwi(), compute_ndvi(), apply_clahe()
│   ├── normalize.py    # normalize_percentile(), normalize_minmax(), normalize_image()
│   └── tiling.py       # TileGenerator, TileInfo
├── models/             # 深度学习模型
│   ├── seg_model.py    # SegmentationModel, create_seg_model()
│   ├── det_model.py    # DetectionModel, DET_CLASSES
│   └── inference.py    # InferencePipeline
├── training/           # 训练
│   ├── dataset.py      # SegmentationDataset, DetectionDataset
│   ├── augmentation.py # get_seg_train_transform(), get_seg_val_transform()
│   ├── metrics.py      # SegmentationMetrics, compute_iou(), compute_miou()
│   ├── train_seg.py    # train_segmentation(), DiceCELoss
│   └── train_det.py    # train_detection()
├── extraction/         # 参数提取
│   ├── canal_params.py # extract_canal_params(), extract_centerline(), measure_width_profile()
│   ├── bridge_params.py# extract_bridge_params()
│   ├── structure_params.py # extract_siphon_params(), extract_aqueduct_params(), extract_gate_params()
│   └── geo_measure.py  # pixel_to_geo(), measure_distance_m(), compute_perpendicular()
├── export/             # 输出
│   ├── vector_io.py    # save_geodataframe(), export_all_results()
│   ├── report.py       # generate_summary_report(), save_report()
│   └── visualization.py# mask_to_rgb(), plot_width_profile(), create_folium_map()
└── utils/              # 工具
    ├── config.py       # load_config(), get_project_root()
    ├── logger.py       # setup_logger()
    ├── geo_utils.py    # pixel_to_geo(), measure_distance_m(), get_utm_crs()
    └── raster_utils.py # read_raster(), write_raster(), compute_ndwi(), compute_ndvi()
```

## 关键 Import 路径（必须严格遵守）
```python
from seehydro.acquisition.route import RouteDataLoader, load_route
from seehydro.acquisition.gee import GEEDownloader
from seehydro.acquisition.highres import HighResManager
from seehydro.preprocessing.clip import clip_raster_by_geometry, clip_along_route
from seehydro.preprocessing.enhance import compute_ndwi, compute_ndvi, apply_clahe
from seehydro.preprocessing.normalize import normalize_image
from seehydro.preprocessing.tiling import TileGenerator, TileInfo
from seehydro.models.seg_model import SegmentationModel, create_seg_model, SEG_CLASSES
from seehydro.models.det_model import DetectionModel, DET_CLASSES, DET_CLASSES_CN
from seehydro.models.inference import InferencePipeline
from seehydro.training.dataset import SegmentationDataset, DetectionDataset
from seehydro.training.metrics import SegmentationMetrics, compute_miou
from seehydro.training.train_seg import train_segmentation
from seehydro.training.train_det import train_detection
from seehydro.extraction.canal_params import extract_canal_params
from seehydro.extraction.bridge_params import extract_bridge_params
from seehydro.extraction.structure_params import extract_all_structures
from seehydro.extraction.geo_measure import pixel_to_geo, measure_distance_m
from seehydro.export.vector_io import save_geodataframe, export_all_results
from seehydro.export.report import generate_summary_report, save_report
from seehydro.export.visualization import create_folium_map, plot_width_profile
from seehydro.utils.config import load_config, get_project_root
from seehydro.utils.logger import setup_logger
```

## 关键 API 签名

### RouteDataLoader
```python
class RouteDataLoader:
    def from_osm(self, bbox: tuple | None = None) -> gpd.GeoDataFrame
    def from_geojson(self, path: str | Path) -> gpd.GeoDataFrame
    def buffer(self, gdf: gpd.GeoDataFrame, width_m: float) -> gpd.GeoDataFrame
    def split_segments(self, gdf: gpd.GeoDataFrame, length_m: float) -> list[gpd.GeoDataFrame]
    def save(self, gdf: gpd.GeoDataFrame, path: str | Path, driver: str = "GeoJSON") -> None
    def get_route_info(self, gdf: gpd.GeoDataFrame) -> dict
```

### TileGenerator
```python
class TileGenerator:
    def __init__(self, tile_size: int = 512, overlap: float = 0.25)
    def generate_tiles(self, image_path, output_dir, prefix="tile", min_valid_ratio=0.5) -> list[TileInfo]
    def reassemble(self, tiles: dict, tile_infos, output_path, original_profile, merge_strategy="mean") -> Path
    def save_tile_index(self, tile_infos, output_path) -> Path
```

### SegmentationModel
```python
class SegmentationModel:
    def __init__(self, model_name="DeepLabV3Plus", encoder="resnet101", encoder_weights="imagenet", in_channels=3, num_classes=5, device=None)
    def predict(self, image: torch.Tensor) -> torch.Tensor  # (C,H,W) -> (H,W)
    def predict_proba(self, image: torch.Tensor) -> torch.Tensor
    def load_weights(self, path) -> None
    def save_weights(self, path) -> None
```

### DetectionModel
```python
class DetectionModel:
    def __init__(self, model_path=None, model_name="yolov8m.pt", device=None, conf_threshold=0.25)
    def predict(self, image, conf=None) -> list[dict]  # each: {bbox, confidence, class_id, class_name}
    def predict_batch(self, images, conf=None) -> list[list[dict]]
    def train(self, data_yaml, epochs=200, imgsz=1024, batch=4, project="models/trained/detection", name="run") -> Path
```

### extract_canal_params
```python
def extract_canal_params(mask_path, water_class_id=1, berm_class_id=3, interval_m=50) -> dict:
    # Returns: {"centerline": LineString, "width_profile": GeoDataFrame, "mean_estimated_water_surface_width_m": float, ...}
```

### InferencePipeline
```python
class InferencePipeline:
    def __init__(self, seg_model_path=None, det_model_path=None, seg_config=None, det_config=None, device=None)
    def run_segmentation(self, tile_dir, output_dir, normalize_method="percentile") -> dict[str, Path]
    def run_detection(self, tile_dir, conf=None) -> dict[str, list[dict]]
    def run_full_pipeline(self, tile_dir, output_dir) -> dict
```

## 依赖
torch, segmentation-models-pytorch, ultralytics, rasterio, geopandas, shapely, pyproj, scikit-image, albumentations, folium, typer, loguru, omegaconf, pydantic

## 配置
configs/default.yaml - OmegaConf格式，用 load_config() 加载

## 分割类别
0:背景 1:渠道水面 2:渠道边坡 3:马道 4:管理道路

## 检测类别
0:公路桥 1:铁路桥 2:倒虹吸入口 3:倒虹吸出口 4:渡槽 5:节制闸 6:退水闸 7:分水口
