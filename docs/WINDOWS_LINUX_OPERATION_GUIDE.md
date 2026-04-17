# SeeHydro Windows / Linux 操作手册

本文档整理了从进入工作目录、下载遥感影像、裁剪切片、模型推理、参数提取到导出 `shp` 的常用命令。

## 1. 目录约定

以下命令默认在仓库根目录执行：

- Linux: `/tmp/seehydro-water-seg`
- Windows: `D:\seehydro-water-seg`

你可以根据自己的实际路径替换。

## 2. Linux 版

### 2.1 进入工作目录

```bash
cd /tmp/seehydro-water-seg
export PYTHONPATH=/tmp/seehydro-water-seg/src
```

### 2.2 天地图下载遥感影像

先设置天地图密钥：

```bash
export TDT_KEY=你的天地图密钥
```

再执行下载：

```bash
python3 -m seehydro.cli download tiles \
  --bbox 114.35,38.20,114.39,38.23 \
  --provider tianditu_img \
  --zoom 18 \
  --output-dir raw_geotiff
```

如果不想用环境变量，也可以直接把 key 写在命令里：

```bash
python3 -m seehydro.cli download tiles \
  --bbox 114.35,38.20,114.39,38.23 \
  --provider tianditu_img \
  --zoom 18 \
  --api-key 你的天地图密钥 \
  --output-dir raw_geotiff
```

### 2.3 其他下载方法

#### 方法 A: Google 卫星底图

```bash
python3 -m seehydro.cli download tiles \
  --bbox 114.35,38.20,114.39,38.23 \
  --provider google_satellite \
  --zoom 18 \
  --output-dir raw_geotiff_google
```

#### 方法 B: Sentinel-2 下载

按配置文件下载：

```bash
python3 -m seehydro.cli download sentinel2 \
  --config configs/default.yaml \
  --output data/sentinel2
```

按 bbox 下载：

```bash
python3 -m seehydro.cli download sentinel2 \
  --config configs/default.yaml \
  --bbox 114.35,38.20,114.39,38.23 \
  --output data/sentinel2/sentinel2_bbox.tif
```

#### 方法 C: 已有本地大幅遥感图

如果你已经有大图，不需要再在线下载，直接放到例如 `raw_geotiff/` 目录即可，然后从裁剪或切片开始。

### 2.4 按线路裁剪大图

如果你有中心线、渠道线或者路线矢量文件，可以先裁剪：

```bash
python3 -m seehydro.cli preprocess clip \
  --input raw_geotiff \
  --route data/route/snbd_centerline.geojson \
  --buffer 2000 \
  --output data/clipped
```

如果没有路线文件，这一步可以跳过，直接切片。

### 2.5 大图切片

对裁剪结果切片：

```bash
python3 -m seehydro.cli preprocess tile \
  --input data/clipped \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

如果你没有做裁剪，直接对原始大图切片：

```bash
python3 -m seehydro.cli preprocess tile \
  --input raw_geotiff \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

### 2.6 分割推理

使用 GPU:

```bash
python3 -m seehydro.cli infer \
  --input data/tiles \
  --model-seg models/seg_water/seg_best.pth \
  --config configs/segmentation_binary_water.yaml \
  --output outputs/infer \
  --device cuda:0 \
  --batch-size 4
```

使用 CPU:

```bash
python3 -m seehydro.cli infer \
  --input data/tiles \
  --model-seg models/seg_water/seg_best.pth \
  --config configs/segmentation_binary_water.yaml \
  --output outputs/infer \
  --device cpu \
  --batch-size 1
```

### 2.7 提取中心线和宽度等参数

```bash
python3 -m seehydro.cli extract \
  --input outputs/infer/merged \
  --output outputs/extraction \
  --sample-interval 50
```

这一步现在会尽量输出：

- `*_water_mask.geojson`
- `*_centerline.geojson`
- `*_width_profile.geojson`
- `*_berm_mask.geojson`（如果有马道类别）
- `*_berm_width_profile.geojson`（如果能提取到马道宽度）
- `reports/*_summary.csv`
- `reports/*_summary.xlsx`
- `summary.json`

补充说明：

- 如果某张图暂时提不出中心线，程序仍会尽量保留 `water_mask` 面结果。
- 宽度结果属于估算值，只能用于辅助分析，不能直接当正式测绘或设计参数。

### 2.8 导出 GeoJSON / SHP

导出 GeoJSON:

```bash
python3 -m seehydro.cli export \
  --input outputs/extraction \
  --format geojson \
  --report outputs/export_reports
```

导出 SHP:

```bash
python3 -m seehydro.cli export \
  --input outputs/extraction \
  --format shapefile \
  --report outputs/export_reports
```

也可以直接传 `outputs/extraction/vectors`。

补充说明：

- 导出 `shapefile` 时，程序会自动处理字段名长度限制。
- 报表现在统一使用：
  `类别 / 子类 / 数量 / 指标项 / 指标值 / 单位 / 备注`

### 2.9 一条命令跑通快速流程

带天地图下载:

```bash
export TDT_KEY=你的天地图密钥

python3 -m seehydro.cli pipeline quickstart \
  --bbox 114.35,38.20,114.39,38.23 \
  --provider tianditu_img \
  --zoom 18 \
  --model-seg models/seg_water/seg_best.pth \
  --config configs/segmentation_binary_water.yaml \
  --device cuda:0 \
  --workspace outputs/pipeline_run
```

使用已有本地影像:

```bash
python3 -m seehydro.cli pipeline quickstart \
  --raw-input raw_geotiff \
  --route data/route/snbd_centerline.geojson \
  --buffer 2000 \
  --model-seg models/seg_water/seg_best.pth \
  --config configs/segmentation_binary_water.yaml \
  --device cuda:0 \
  --workspace outputs/pipeline_run
```

## 3. Windows 版

### 3.1 进入工作目录

```powershell
cd /d D:\seehydro-water-seg
$env:PYTHONPATH="D:\seehydro-water-seg\src"
```

如果你使用 CMD:

```cmd
cd /d D:\seehydro-water-seg
set PYTHONPATH=D:\seehydro-water-seg\src
```

### 3.2 天地图下载遥感影像

PowerShell:

```powershell
$env:TDT_KEY="你的天地图密钥"
python -m seehydro.cli download tiles `
  --bbox 114.35,38.20,114.39,38.23 `
  --provider tianditu_img `
  --zoom 18 `
  --output-dir raw_geotiff
```

CMD:

```cmd
set TDT_KEY=你的天地图密钥
python -m seehydro.cli download tiles ^
  --bbox 114.35,38.20,114.39,38.23 ^
  --provider tianditu_img ^
  --zoom 18 ^
  --output-dir raw_geotiff
```

### 3.3 其他下载方法

#### 方法 A: Google 卫星底图

PowerShell:

```powershell
python -m seehydro.cli download tiles `
  --bbox 114.35,38.20,114.39,38.23 `
  --provider google_satellite `
  --zoom 18 `
  --output-dir raw_geotiff_google
```

#### 方法 B: Sentinel-2

```powershell
python -m seehydro.cli download sentinel2 `
  --config configs/default.yaml `
  --bbox 114.35,38.20,114.39,38.23 `
  --output data/sentinel2/sentinel2_bbox.tif
```

#### 方法 C: 使用已有本地大图

把现有 `tif/tiff` 放进 `raw_geotiff\` 目录，然后从裁剪或切片步骤开始。

### 3.4 按线路裁剪

```powershell
python -m seehydro.cli preprocess clip `
  --input raw_geotiff `
  --route data\route\snbd_centerline.geojson `
  --buffer 2000 `
  --output data\clipped
```

### 3.5 切片

```powershell
python -m seehydro.cli preprocess tile `
  --input data\clipped `
  --size 512 `
  --overlap 0.25 `
  --output data\tiles
```

### 3.6 分割推理

GPU:

```powershell
python -m seehydro.cli infer `
  --input data\tiles `
  --model-seg models\seg_water\seg_best.pth `
  --config configs\segmentation_binary_water.yaml `
  --output outputs\infer `
  --device cuda:0 `
  --batch-size 4
```

CPU:

```powershell
python -m seehydro.cli infer `
  --input data\tiles `
  --model-seg models\seg_water\seg_best.pth `
  --config configs\segmentation_binary_water.yaml `
  --output outputs\infer `
  --device cpu `
  --batch-size 1
```

### 3.7 提取参数

```powershell
python -m seehydro.cli extract `
  --input outputs\infer\merged `
  --output outputs\extraction `
  --sample-interval 50
```

提取输出现在会尽量包括：

- `*_water_mask.geojson`
- `*_centerline.geojson`
- `*_width_profile.geojson`
- `*_berm_mask.geojson`
- `*_berm_width_profile.geojson`
- `reports\*_summary.csv`
- `reports\*_summary.xlsx`
- `summary.json`

### 3.8 导出 SHP

```powershell
python -m seehydro.cli export `
  --input outputs\extraction `
  --format shapefile `
  --report outputs\export_reports
```

也可以直接传：

```powershell
--input outputs\extraction\vectors
```

### 3.9 快速流程

天地图下载 + 推理:

```powershell
$env:TDT_KEY="你的天地图密钥"

python -m seehydro.cli pipeline quickstart `
  --bbox 114.35,38.20,114.39,38.23 `
  --provider tianditu_img `
  --zoom 18 `
  --model-seg models\seg_water\seg_best.pth `
  --config configs\segmentation_binary_water.yaml `
  --device cuda:0 `
  --workspace outputs\pipeline_run
```

已有本地遥感图:

```powershell
python -m seehydro.cli pipeline quickstart `
  --raw-input raw_geotiff `
  --route data\route\snbd_centerline.geojson `
  --buffer 2000 `
  --model-seg models\seg_water\seg_best.pth `
  --config configs\segmentation_binary_water.yaml `
  --device cuda:0 `
  --workspace outputs\pipeline_run
```

## 4. 常见输出文件

推理输出:

- `outputs/infer/segmentation/`: 每个切片的分割掩膜
- `outputs/infer/merged/`: 回拼后的整图掩膜

参数提取输出:

- `outputs/extraction/vectors/*_water_mask.geojson`: 自动提取的水面掩膜面
- `outputs/extraction/vectors/*_centerline.geojson`: 自动提取的中心线
- `outputs/extraction/vectors/*_width_profile.geojson`: 水面宽度采样结果
- `outputs/extraction/vectors/*_berm_mask.geojson`: 马道掩膜面结果
- `outputs/extraction/vectors/*_berm_width_profile.geojson`: 马道宽度采样结果
- `outputs/extraction/reports/*.csv`: 汇总表
- `outputs/extraction/reports/*.xlsx`: Excel 汇总表
- `outputs/extraction/summary.json`: 总结结果

导出结果:

- `outputs/extraction/vectors/export_shapefile/`: 导出的 SHP
- `outputs/extraction/vectors/export_geojson/`: 导出的 GeoJSON
- `outputs/export_reports/`: 单独导出的汇总报表

## 5. 推荐执行顺序

### 方案 1: 天地图在线下载

1. 进入仓库目录
2. 设置 `PYTHONPATH`
3. 设置 `TDT_KEY`
4. 下载大图
5. 可选裁剪
6. 切片
7. 推理
8. 提参
9. 导出 SHP

### 方案 2: 已有本地大图

1. 进入仓库目录
2. 把已有 `tif/tiff` 放到 `raw_geotiff/`
3. 可选裁剪
4. 切片
5. 推理
6. 提参
7. 导出 SHP

## 6. 说明

- 如果机器支持 CUDA，推理建议使用 `--device cuda:0`
- 如果显存不够，把 `--batch-size` 从 `4` 降到 `2` 或 `1`
- 如果没有路线文件，不需要强制手动画中心线；可以直接对大图切片再推理
- 后处理和导出主要是 CPU 过程，不依赖 GPU
- `pipeline quickstart` 现在会尽量输出和单独执行 `extract` 一致的结果
- 报表列结构已统一，便于后续再次处理
