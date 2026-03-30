# SeeHydro Water Seg

这是一个给零基础用户准备的渠道水面分割训练项目。

这份 README 不讲大而全的概念，重点只做一件事：

- 教你从 0 开始准备图片
- 教你手工标注
- 教你把标注变成训练数据
- 教你开始训练一个最基础的水面分割模型
- 教你训练完以后怎么理解结果、下一步该做什么

如果你是第一次接触这个项目，请先只做：

- 二分类水面分割

不要一开始就做：

- 桥梁检测
- 闸门识别
- 多类别精细分割
- 全流程自动推理

原因很简单：

- 当前仓库里，最稳、最完整、最适合新手跑通的是 `水面分割训练`
- 有些 CLI 功能还没有实现，直接去碰会卡住

## 1. 你最终要做成什么

你最后最少应该得到一个模型文件，例如：

```text
models/seg_water/seg_best.pth
```

它表示：

- 你已经成功训练了一个基础分割模型
- 模型可以输入一张图，输出哪里是水面、哪里不是水面

在这个最基础任务里，类别只有 2 个：

- `0 = background`
- `1 = canal_water`

## 2. 先记住：你到底要先做什么

很多人一上来就问：

- 能不能先训练？
- 能不能先跑模型？
- 能不能直接推理？

对这个仓库来说，最稳的顺序是：

1. 准备图片
2. 手工标注
3. 把标注转换成训练数据
4. 检查训练数据
5. 开始训练
6. 看训练结果
7. 继续补数据、继续训练

也就是说：

- **不能在没有标注数据的情况下直接开始训练**

因为训练需要两样东西：

- 输入图像
- 对应答案

这里的“答案”就是掩膜图，也就是：

- 哪些像素是水面
- 哪些像素不是水面

如果你没有做标注，没有生成掩膜，就没有办法正常训练。

## 3. 当前仓库里哪些功能能用，哪些别碰

### 当前最适合新手跑通的功能

- 用 Labelme 做手工标注
- 用脚本把 Labelme `json` 转成 `images + masks`
- 用脚本检查数据是否合格
- 用现成配置训练一个二分类水面分割模型

### 当前不要作为第一步去碰的功能

以下 CLI 已经留了入口，但目前还没实现完整逻辑：

- `preprocess clip`
- `infer`
- `extract`
- `export`

也就是说，你现在最该做的是：

- 先把训练流程跑通

## 3.1 如果你下载的是遥感图，先分清楚两种目标

很多人会混在一起问：

- 我是要训练模型
- 我又想保留遥感图里的空间信息

这两个目标有关，但不是一回事。

### 目标 A：只想先把模型训出来

这时候最核心的是：

- 图像
- 标注
- 掩膜

只要图像和掩膜配对正确，就可以训练。

这种情况下，就算没有地理坐标，模型也能学。

### 目标 B：不仅想训练，还想保留遥感空间信息

如果你后面还要做这些事情：

- 回到地图上定位
- 计算真实宽度、长度、面积
- 做参数提取
- 结果叠加到 GIS

那你就不能只关心像素。

你还要尽量保留这些信息：

- `CRS`
- `transform`
- `bounds`
- 像素分辨率
- `nodata`

### 先记住一个结论

- **训练模型本身不一定依赖空间信息**
- **但遥感工程应用通常需要空间信息**

所以如果你下载的是遥感图，推荐优先走：

- **保留 GeoTIFF 信息的数据准备路线**

而不是只把图当成普通图片来处理。

## 3.2 当前仓库里，哪些步骤能保留空间信息

这一步很重要，先别猜，先记结论。

### 会保留空间信息的步骤

- `preprocess tile`

也就是：

- 把 GeoTIFF 大图切成很多 GeoTIFF 小图

为什么它能保留？

因为切片代码会把原图的窗口变换写进每个小图，并记录 `crs`：

- [tiling.py](/root/SeeHydro/src/seehydro/preprocessing/tiling.py)

### 可以在特定条件下继承空间信息的步骤

- `scripts/convert_labelme_to_masks.py`

这个脚本现在支持一种更实用的情况：

- 如果 Labelme 的 `json` 还能正确指向原始 `.tif/.tiff`
- 并且 Labelme 实际使用的图像尺寸和原始 GeoTIFF 尺寸一致

那么脚本在输出训练数据时，会尽量继承这些信息：

- `crs`
- `transform`
- `nodata`

也就是说，它现在不再是“一定丢空间信息”的脚本。

但要注意，它能否继承成功，取决于你的标注流程是否满足前提。

### 什么情况下会继承失败

所以你要记住：

- 如果你给 Labelme 的是普通图片，比如 `png/jpg`
- 或者 `json` 已经找不到原始 GeoTIFF
- 或者标注图和原始 GeoTIFF 尺寸不一致

那么训练仍然可以做，但这时空间参考就无法安全继承。

## 3.3 如果你想保留信息，最推荐你怎么走

如果你的原始数据是遥感 GeoTIFF，最推荐你按下面顺序来：

1. 先检查原图有没有空间信息
2. 再把原图切成带空间信息的小图
3. 再拿这些小图去做标注
4. 再用支持继承空间参考的转换脚本生成训练数据
5. 训练时尽量保留 image/mask 的空间参考
6. 后续如果要做工程落图，再进一步处理结果回写

先别追求一步到位把所有后处理都打通。

你现在最重要的是先建立正确的数据路线。

## 4. 你需要准备什么软件

最少需要：

1. `Python 3.10`
2. `Git`
3. `Labelme`

如果你还要自己准备遥感数据，常见还会用到：

4. 能下载遥感图的平台账号
5. 基础 GIS/遥感查看工具

例如：

- QGIS
- ArcGIS
- Google Earth Engine 账号
- 天地图 key

### Labelme 是干什么的

Labelme 是一个手工标注工具。

你在图上把水面区域圈出来，保存后会得到一个 `.json` 文件。

你可以把它理解成：

- 图片 = 题目
- 标注 = 标准答案
- 模型 = 学做题的人

## 5. 你第一次最推荐怎么开始

不要一开始就拿几百张图，也不要一开始就搞大区域全线数据。

最推荐这样做：

1. 先找 `3 张` 图试跑一遍完整流程
2. 确认命令都能跑通
3. 再扩到 `20 到 50 张`
4. 再正式训练第一版模型

第一次的目标不是做出最强模型，而是：

- 数据路径正确
- 标注方式正确
- 脚本能跑
- 训练能结束
- 模型能学到东西

## 5.1 如果你现在什么目录都还没有，先这样建

如果你是第一次动手，推荐先在项目根目录下把最基础的目录建好。

先进入项目目录：

```bash
cd /root/SeeHydro
```

然后创建目录：

```bash
mkdir -p raw_geotiff
mkdir -p data/tiles
mkdir -p data/seg_water/images
mkdir -p data/seg_water/masks
mkdir -p labelme_work
mkdir -p models/seg_water
mkdir -p logs/seg_water
```

### 这些目录分别是干什么的

- `raw_geotiff`
  放你原始下载的遥感大图，尽量不要乱改

- `data/tiles`
  放切片后的 GeoTIFF 小图

- `labelme_work`
  放你准备用 Labelme 标注的图片和对应 `json`

- `data/seg_water/images`
  放训练用图像

- `data/seg_water/masks`
  放训练用掩膜

- `models/seg_water`
  放训练好的模型

- `logs/seg_water`
  放训练日志

## 5.2 现在还没数据的话，先从哪里弄数据

如果你现在手里一张图都没有，可以先从下面这些来源准备数据。

### 方式 A：自己已有本地遥感 GeoTIFF

这是最简单的情况。

如果你手里已经有：

- `.tif`
- `.tiff`

那就直接放到：

```text
/root/SeeHydro/raw_geotiff/
```

然后从“检查空间信息”和“切片”开始往下走。

### 方式 B：Google Earth Engine 下载 Sentinel-2

仓库里已经有 GEE 下载器：

- [gee.py](/root/SeeHydro/src/seehydro/acquisition/gee.py)

它主要适合：

- 下载 Sentinel-2
- 按线路分段下载
- 输出 GeoTIFF

这类数据的特点是：

- 获取相对规范
- 自带空间参考
- 分辨率通常适合做基础水面识别

如果你后面想走程序化下载，这条路线是仓库里比较正统的一条。

### 但要先知道一个现实情况

当前 CLI 里的这个命令：

```bash
python -m seehydro.cli download sentinel2
```

还没有实现完整逻辑。

所以你现在不要按这个命令走。

当前最稳的做法是：

- 直接用仓库里的 `GEEDownloader` 写一小段 Python 下载

### GEE 下载 Sentinel-2 的傻瓜式步骤

下面这套步骤我直接按：

- **南水北调中线京石段附近的一小段**

来写。

这样你可以直接复制用，不用先自己去想测试区域。

#### 第 1 步：先准备 GEE 账号

你至少要有：

- 一个可用的 Google 账号
- 已经能使用 Google Earth Engine

如果你之前没配过，第一次运行下载代码时，通常会弹出认证流程。

#### 第 2 步：进入项目环境

```bash
cd /root/SeeHydro
source .venv/bin/activate
export PYTHONPATH=/root/SeeHydro/src
```

#### 第 3 步：先准备一个很小的下载范围

第一次不要下载太大范围。

这里直接给你一个京石段起步用的小范围：

- `bbox = [114.35, 38.20, 114.39, 38.23]`

格式是：

```python
[min_lon, min_lat, max_lon, max_lat]
```

这一步你先不要改，先直接用这组范围跑第一次。

#### 第 4 步：执行最小下载脚本

下面这段脚本的作用是：

- 初始化 GEE
- 指定一个矩形范围
- 下载一张 Sentinel-2 影像
- 保存到 `raw_geotiff/`

```bash
python - <<'PY'
import ee
from pathlib import Path

from seehydro.acquisition.gee import GEEDownloader

output_path = Path("raw_geotiff/jingshi_sentinel2_small.tif")

# 京石段附近的小范围测试区域
# 格式: [min_lon, min_lat, max_lon, max_lat]
bbox = [114.35, 38.20, 114.39, 38.23]

downloader = GEEDownloader()
geometry = ee.Geometry.Rectangle(bbox)

image = downloader.get_sentinel2(
    geometry=geometry,
    date_range=("2024-01-01", "2024-12-31"),
    cloud_pct_max=10,
    bands=["B2", "B3", "B4"],
)

downloader.download_image(
    image=image,
    geometry=geometry,
    output_path=output_path,
    scale=10,
)

print("下载完成:", output_path)
PY
```

#### 第 5 步：确认文件已经下来了

```bash
ls -lh raw_geotiff
```

你应该至少能看到类似：

```text
raw_geotiff/jingshi_sentinel2_small.tif
```

#### 第 6 步：马上检查这张图有没有空间信息

```bash
python - <<'PY'
import rasterio

path = "raw_geotiff/jingshi_sentinel2_small.tif"

with rasterio.open(path) as src:
    print("width =", src.width)
    print("height =", src.height)
    print("count =", src.count)
    print("crs =", src.crs)
    print("transform =", src.transform)
    print("bounds =", src.bounds)
    print("nodata =", src.nodata)
PY
```

如果 `crs`、`transform`、`bounds` 正常，那说明这张图已经可以进入后面的流程。

#### 第 7 步：对下载回来的 Sentinel-2 做切片

```bash
python -m seehydro.cli preprocess tile \
  --input raw_geotiff/jingshi_sentinel2_small.tif \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

#### 第 8 步：确认切片结果

```bash
ls -lh data/tiles | head
```

你应该能看到：

- 一批切片 `.tif`
- 一个 `tile_index.csv`

#### 第 9 步：先看看切片都叫什么名字

```bash
find data/tiles -maxdepth 1 -type f -name "*.tif" | sort | head -n 10
```

你会看到类似：

```text
data/tiles/jingshi_sentinel2_small_r0000_c0000.tif
data/tiles/jingshi_sentinel2_small_r0000_c0001.tif
data/tiles/jingshi_sentinel2_small_r0001_c0000.tif
```

#### 第 10 步：从切片里挑 3 到 10 张到标注目录

第一次只拿几张就够，不要一次全搬过去。

最简单的做法是直接复制前 3 张：

```bash
cp data/tiles/jingshi_sentinel2_small_r0000_c0000.tif labelme_work/
cp data/tiles/jingshi_sentinel2_small_r0000_c0001.tif labelme_work/
cp data/tiles/jingshi_sentinel2_small_r0001_c0000.tif labelme_work/
```

如果这几个文件名和你实际切出来的不一样，就按上一步 `find` 出来的名字替换。

#### 第 11 步：确认标注目录里已经有图

```bash
ls -lh labelme_work
```

你应该能看到刚复制进去的 `.tif`。

#### 第 12 步：再去做 Labelme 标注

到这里你再打开 Labelme，开始标：

- `water`

#### 第 13 步：Labelme 里怎么做

1. 打开 `Labelme`
2. 点击 `Open Dir`
3. 选择 `/root/SeeHydro/labelme_work`
4. 打开第一张 `tif`
5. 点击 `Create Polygons`
6. 沿着水面边缘画多边形
7. 标签统一输入 `water`
8. 保存
9. 每一张都保存出一个同名 `.json`

标完后你的目录应该像这样：

```text
labelme_work/
  jingshi_sentinel2_small_r0000_c0000.tif
  jingshi_sentinel2_small_r0000_c0000.json
  jingshi_sentinel2_small_r0000_c0001.tif
  jingshi_sentinel2_small_r0000_c0001.json
```

#### 第 14 步：把标注转换成训练数据

```bash
python scripts/convert_labelme_to_masks.py \
  --input-dir labelme_work \
  --output-root data/seg_water
```

#### 第 15 步：确认训练数据已经生成

```bash
find data/seg_water -maxdepth 2 -type f | sort | head -n 20
```

你应该能看到：

- `data/seg_water/images/*.tif`
- `data/seg_water/masks/*.tif`

#### 第 16 步：检查转换后的训练数据有没有保留空间信息

```bash
python - <<'PY'
import rasterio

for path in [
    "data/seg_water/images/jingshi_sentinel2_small_r0000_c0000.tif",
    "data/seg_water/masks/jingshi_sentinel2_small_r0000_c0000.tif",
]:
    with rasterio.open(path) as src:
        print(path)
        print("crs =", src.crs)
        print("transform =", src.transform)
        print("bounds =", src.bounds)
        print("nodata =", src.nodata)
        print()
PY
```

#### 第 17 步：检查训练数据格式是否合格

```bash
python scripts/validate_seg_dataset.py \
  --image-dir data/seg_water/images \
  --mask-dir data/seg_water/masks \
  --num-classes 2
```

#### 第 18 步：开始训练第一版模型

```bash
python -m seehydro.cli train segmentation \
  --config configs/segmentation_binary_water.yaml
```

#### 第 19 步：确认模型文件已经生成

```bash
ls -lh models/seg_water
```

你应该重点看这个文件：

```text
models/seg_water/seg_best.pth
```

#### 第 20 步：如果第一版能跑通，再扩大样本

第一次不要着急扩太大。

最稳的路线是：

1. 先用这 3 到 10 张跑通
2. 再从 `data/tiles/` 里挑更多切片
3. 再继续标注
4. 再重新训练

#### 第 21 步：如果你后面想换成别的渠段

你只要改这几个地方：

1. 下载脚本里的 `bbox`
2. 输出文件名
3. 后面复制切片时对应的文件名

其他流程都不变。

#### 第 22 步：这条京石段起步版流程的最短总结

后面的顺序就是：

1. `convert_labelme_to_masks.py`
2. `validate_seg_dataset.py`
3. `train segmentation`

### 如果京石段这个 bbox 下载失败，按这个顺序排查

第一次跑 GEE，失败并不稀奇。

你不要一报错就全部推翻，按下面顺序一点点排。

#### 情况 1：一运行就卡在认证或者提示 GEE 初始化失败

这通常不是 bbox 的问题，而是：

- GEE 账号还没认证好
- 本机还没有有效凭据
- 当前环境没法完成认证

你先做这几件事：

1. 确认你能正常使用自己的 Google Earth Engine 账号
2. 重新进入项目虚拟环境
3. 再重新执行一次最小下载脚本

如果第一次运行弹出认证流程，就按提示完成认证。

#### 情况 2：脚本能跑，但最后没有下载出 tif

你先检查：

```bash
ls -lh raw_geotiff
```

如果目录里没有：

- `jingshi_sentinel2_small.tif`

那先不要继续切片，先回头查下载这一步。

#### 情况 3：提示区域太大、请求失败、导出失败

第一次最常见的处理办法不是乱改代码，而是：

- **先缩小 bbox**

你现在的起步 bbox 是：

```python
bbox = [114.35, 38.20, 114.39, 38.23]
```

如果它失败，你先改成更小的测试范围，例如：

```python
bbox = [114.35, 38.20, 114.37, 38.215]
```

也就是：

- 经度范围缩小
- 纬度范围缩小

先让第一次下载成功，比一开始范围大更重要。

#### 情况 4：提示影像为空、拿不到合适影像、结果不理想

这通常优先改的是：

- `date_range`
- `cloud_pct_max`

你现在的默认参数是：

```python
date_range=("2024-01-01", "2024-12-31")
cloud_pct_max=10
```

如果这组条件太严，你可以先放宽云量限制，比如：

```python
cloud_pct_max=20
```

或者先换一个时间范围，例如：

```python
date_range=("2023-01-01", "2024-12-31")
```

最保守的调参顺序是：

1. 先放宽 `cloud_pct_max`
2. 再扩大 `date_range`
3. 最后才考虑换区域

#### 情况 5：下载出来了，但 tif 打不开或者空间信息不正常

不要急着切片，先检查：

```bash
python - <<'PY'
import rasterio

path = "raw_geotiff/jingshi_sentinel2_small.tif"

with rasterio.open(path) as src:
    print("width =", src.width)
    print("height =", src.height)
    print("count =", src.count)
    print("crs =", src.crs)
    print("transform =", src.transform)
    print("bounds =", src.bounds)
    print("nodata =", src.nodata)
PY
```

如果：

- `crs` 是空的
- `bounds` 明显异常
- 图像尺寸特别离谱

那说明这张结果图本身就不适合继续走后面的流程，先回到下载步骤重试。

#### 情况 6：下载成功了，但切片后没有生成多少小图

这时优先检查两件事：

1. 原图本身是不是太小
2. `--size 512` 对这张图是不是太大

如果原图范围比较小，512 可能切不出很多块。

你可以试试改成：

```bash
python -m seehydro.cli preprocess tile \
  --input raw_geotiff/jingshi_sentinel2_small.tif \
  --size 256 \
  --overlap 0.25 \
  --output data/tiles
```

也就是说：

- 图太小，就把切片尺寸调小

#### 情况 7：切片出来了，但你复制到 `labelme_work/` 的文件名对不上

先不要手敲猜名字，直接先看：

```bash
find data/tiles -maxdepth 1 -type f -name "*.tif" | sort | head -n 20
```

然后按实际存在的文件名复制。

#### 情况 8：标注后转换失败

这时优先检查：

1. `labelme_work/` 里有没有对应的 `.json`
2. 标签是不是统一写成了 `water`
3. `json` 里的 `imagePath` 还能不能找到原始标注图

不要一上来就怀疑训练代码，先看标注文件本身。

#### 情况 9：转换成功了，但空间信息没有保留下来

优先检查这三个问题：

1. 你给 Labelme 的是不是原始 tif 切片，而不是另存后的 png/jpg
2. `json` 还能不能找到原始 tif
3. 标注图尺寸和原始 tif 尺寸是否一致

如果这三条有一条不满足，空间参考就可能继承失败。

#### 最后给你一个最保守的排障原则

如果第一次跑不通，调整顺序建议永远是：

1. 先缩小 bbox
2. 再放宽 `cloud_pct_max`
3. 再扩大 `date_range`
4. 再减小切片 `size`
5. 最后才去改更复杂的代码逻辑

### 如果你想扩大到多个区域

先不要一步就下很多大图。

更稳的做法是：

1. 先下载一个小区域
2. 先切片
3. 先标几张
4. 先训练第一版
5. 跑通后再下载更多区域

这样最不容易把问题堆在一起。

### 方式 C：在线地图瓦片下载后拼接

仓库里也有高分图瓦片下载器：

- [highres.py](/root/SeeHydro/src/seehydro/acquisition/highres.py)

它支持这类思路：

- 按经纬度范围下载在线地图瓦片
- 再拼接成 GeoTIFF

当前代码里已经写了这些 provider：

- `tianditu`
- `tianditu_label`

这条路线适合：

- 你想快速拿一个区域的高分底图
- 你已经有天地图 key

### 方式 D：从现成平台手工下载，再放进项目

如果你现在不想先折腾代码下载，也完全可以手工下载，再放到项目里。

实际操作上，你只要保证：

- 下载结果最好是 `.tif/.tiff`
- 最好保留空间参考
- 放进 `raw_geotiff/`

就可以接入这套流程。

### 对新手最推荐哪种

如果你是第一次跑：

- 最推荐先找几张现成 GeoTIFF
- 先手工放进 `raw_geotiff/`
- 先把切片、标注、训练流程跑通

不要一上来就把“自动下载数据”也当作第一层难点。

## 5.3 下载回来以后先怎么整理数据

不管你从哪里拿到图，先做这几件事：

1. 原始大图统一放到 `raw_geotiff/`
2. 文件名尽量简单，不要太乱
3. 不要随手改成 png/jpg
4. 不要覆盖原始图
5. 后续处理都尽量基于副本或切片做

例如：

```text
/root/SeeHydro/raw_geotiff/
  scene_001.tif
  scene_002.tif
  scene_003.tif
```

## 6. 目录结构建议你这样建

你可以在项目根目录下先准备一个手工工作区：

```text
/root/SeeHydro/
  labelme_work/
    a001.png
    a002.png
    a003.png
```

标注完成以后，同目录下会出现：

```text
/root/SeeHydro/
  labelme_work/
    a001.png
    a001.json
    a002.png
    a002.json
    a003.png
    a003.json
```

后面转换完成后，训练数据目录会长这样：

```text
/root/SeeHydro/
  data/
    seg_water/
      images/
        a001.tif
        a002.tif
      masks/
        a001.tif
        a002.tif
```

## 7. 第一步：准备图片

先准备少量图片，不要太多。

建议第一批：

- `20 到 50 张`

如果你只是试流程：

- `3 张` 就够

### 图片要求

尽量满足：

- 尽量清晰
- 尽量同类场景
- 尽量同一类分辨率
- 先别拿特别复杂的图

### 可以用什么格式

标注阶段可以先用这些常见格式：

- `png`
- `jpg`
- `jpeg`

后面脚本会帮你转换成训练用的 `.tif`。

### 如果你准备的是遥感 GeoTIFF

建议你优先准备：

- `.tif`
- `.tiff`

并且尽量保留原始下载文件，不要一上来就另存成：

- 截图
- 普通 png
- 普通 jpg

因为一旦变成普通图片，很多空间参考信息就没了。

## 7.0 如果你现在还没有图片，最傻瓜式拿数据的方法

如果你现在连一张训练图都没有，最推荐这样做：

1. 先准备 `3 张` 到 `10 张` 遥感 GeoTIFF
2. 先放到 `raw_geotiff/`
3. 先确认这些图能正常打开
4. 先检查有没有 `crs/transform`
5. 再切片
6. 再标注
7. 再训练

### 一个最保守的起步策略

你先不要追求“大区域全覆盖”，先做一个很小的区域就够。

例如你可以先只准备：

- 同一个区域的少量图
- 同一种分辨率的图
- 场景差异不要太大

这样更容易先跑通。

## 7.1 先检查你的遥感图有没有空间信息

如果你下载的是 `.tif/.tiff`，第一步不是直接训练，也不是马上标注。

第一步应该是：

- 先检查它有没有地理参考信息

### 怎么检查

先进入环境：

```bash
cd /root/SeeHydro
source .venv/bin/activate
export PYTHONPATH=/root/SeeHydro/src
```

然后执行：

```bash
python - <<'PY'
import rasterio

path = "把这里改成你的tif路径"

with rasterio.open(path) as src:
    print("width =", src.width)
    print("height =", src.height)
    print("count =", src.count)
    print("crs =", src.crs)
    print("transform =", src.transform)
    print("bounds =", src.bounds)
    print("nodata =", src.nodata)
PY
```

### 看什么结果

你重点看这几项：

- `crs`
- `transform`
- `bounds`

如果出现下面这种情况：

- `crs` 不是 `None`
- `transform` 不是默认空变换
- `bounds` 看起来是正常坐标范围

那这张图通常就是带空间信息的 GeoTIFF。

如果这些值是空的，或者明显不正常，那说明：

- 这张图可能只是普通 tif
- 或者下载时空间参考已经丢了

## 7.2 如果你要保留信息，先切图，再标注

如果你的图很大，又想尽量保留遥感信息，推荐顺序不是：

- 先把大图随便导成 png 去标

而是：

- **先用 GeoTIFF 做切片**
- **再用切出来的小图做后续处理**

### 为什么先切片

因为切片时，当前项目代码会保留每个小图的空间参考关系。

也就是说，原图有空间信息的话，小图通常也还能保留。

### 切片命令

```bash
python -m seehydro.cli preprocess tile \
  --input /你的tif目录或单个tif \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

### 这条命令会得到什么

例如：

```text
data/tiles/
  xxx_r0000_c0000.tif
  xxx_r0000_c0001.tif
  xxx_r0001_c0000.tif
  tile_index.csv
```

这里的重点不是“切成了小图”本身，而是：

- 这些小图仍然是 tif
- 它们来自原始 GeoTIFF
- 索引里记录了来源关系

### 切完以后怎么确认信息还在

你可以再检查任意一个切片：

```bash
python - <<'PY'
import rasterio

path = "data/tiles/把这里改成某个切片文件名.tif"

with rasterio.open(path) as src:
    print("crs =", src.crs)
    print("transform =", src.transform)
    print("bounds =", src.bounds)
PY
```

如果这些信息还在，说明这一步没把空间参考丢掉。

## 7.3 切片到底是在做什么

很多人会把“切片”“裁剪”“拆图”混在一起。

你在这个项目里当前最该理解成：

- **把一张很大的 GeoTIFF 拆成很多小的 GeoTIFF**

这样做的原因是：

- 大图不方便直接标注
- 大图不方便直接训练
- 小图更容易管理
- 小图更容易控制样本质量

### 举个最简单的例子

如果你有一张很大的图：

```text
scene_001.tif
```

切片后可能会变成：

```text
data/tiles/
  scene_001_r0000_c0000.tif
  scene_001_r0000_c0001.tif
  scene_001_r0001_c0000.tif
  scene_001_r0001_c0001.tif
  tile_index.csv
```

### `size` 和 `overlap` 怎么理解

- `--size 512`
  每个小图大小是 `512 x 512`

- `--overlap 0.25`
  相邻小图之间有 25% 重叠

为什么要重叠？

因为如果完全不重叠，目标可能刚好被切在边缘，影响标注和训练。

### 新手第一次怎么设参数

最推荐先这样：

```bash
python -m seehydro.cli preprocess tile \
  --input raw_geotiff \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

这是一个最稳的起步参数组合。

## 7.4 切片之后你不要立刻全量去标

切完片以后，新手最容易犯的错是：

- 一下子把所有切片都拿去标

不推荐这样做。

最推荐这样：

1. 先从 `data/tiles/` 挑 `3 到 10 张`
2. 先复制到 `labelme_work/`
3. 先标完这几张
4. 先把流程跑通
5. 再回来继续挑更多切片

### 为什么要先挑少量样本

因为你要先确认：

- 标注方式对不对
- 转换脚本对不对
- 数据检查能不能过
- 训练能不能起

而不是一开始就做大量重复劳动。

### 最简单的做法

你可以直接手工复制少量切片到标注目录：

```bash
cp data/tiles/scene_001_r0000_c0000.tif labelme_work/
cp data/tiles/scene_001_r0000_c0001.tif labelme_work/
cp data/tiles/scene_001_r0001_c0000.tif labelme_work/
```

然后打开 Labelme 去标。

## 8. 第二步：用 Labelme 手工标注

这一部分是你必须亲手做的，脚本不能替你做。

### 如果你很在意保留空间信息，这一节要特别注意

Labelme 更适合做“图像级标注”，不适合当作完整的遥感空间信息管理工具。

所以你要明确：

- Labelme 主要负责帮你画出水面范围
- 它不是为了帮你维护 GeoTIFF 的完整地理参考

也就是说，当前项目这条路线下：

- **标注能做**
- **训练能做**
- **空间信息继承主要依赖原始 GeoTIFF 是否还能被转换脚本找到**

如果你后面要非常严格地把预测结果再落回地图，后续还需要补一层“把结果重新挂回原始空间参考”的处理。

### 你到底标什么

只标一个标签：

- `water`

注意，必须统一写成：

- `water`

不要写成：

- `Water`
- `canal`
- `river`
- `水面`

### 你在 Labelme 里怎么操作

1. 打开 `Labelme`
2. 点击 `Open Dir`
3. 选择 `/root/SeeHydro/labelme_work`
4. 打开一张图片
5. 选择 `Create Polygons`
6. 沿着水面边缘一点一点画
7. 闭合后输入标签名：`water`
8. 保存

### 标注时你只记住一条

- 水面就标
- 不是水面的不要标

### 标完后会得到什么

例如你标了 `a001.png`，通常会得到：

```text
labelme_work/
  a001.png
  a001.json
```

这个 `json` 还不能直接用于训练。

你还要把它转换成掩膜图。

## 9. 第三步：把 Labelme JSON 转成训练数据

仓库里已经有现成脚本：

- [scripts/convert_labelme_to_masks.py](/root/SeeHydro/scripts/convert_labelme_to_masks.py)

这个脚本会做两件事：

1. 读取 Labelme 的 `json`
2. 生成训练所需的：
   `data/seg_water/images/*.tif`
   `data/seg_water/masks/*.tif`

### 先进入项目环境

```bash
cd /root/SeeHydro
source .venv/bin/activate
export PYTHONPATH=/root/SeeHydro/src
```

### 然后执行转换

```bash
python scripts/convert_labelme_to_masks.py \
  --input-dir labelme_work \
  --output-root data/seg_water
```

### 这条命令是什么意思

- `--input-dir labelme_work`
  表示去 `labelme_work/` 里找 `.json` 标注文件

- `--output-root data/seg_water`
  表示把结果输出到：
  `data/seg_water/images`
  `data/seg_water/masks`

### 转换后会得到什么

例如：

```text
data/seg_water/images/a001.tif
data/seg_water/masks/a001.tif
data/seg_water/images/a002.tif
data/seg_water/masks/a002.tif
```

### 这里有几个硬要求

1. 图像和掩膜文件名必须一致
2. 掩膜必须是单通道
3. 掩膜像素值只能是 `0` 和 `1`

含义是：

- `0 = 背景`
- `1 = 水面`

### 这里一定要知道现在的规则

当前这个脚本适合做：

- 把标注转换成训练可用的图像和掩膜

而且在下面这个前提成立时：

- `json` 仍然指向原始 `.tif/.tiff`
- 原始图尺寸和标注图尺寸一致

它现在会尽量把原图的空间参考继承到输出的 `image tif` 和 `mask tif`。

所以如果你的目标是：

- 先把模型训起来

那这个脚本没问题。

如果你的目标是：

- 训练过程中尽量保留原始 GeoTIFF 空间参考

那你现在可以直接用这版脚本，但前提仍然是你的数据流程没有把 GeoTIFF 来源关系打断。

## 10. 第四步：检查数据是不是合格

仓库里也已经有现成检查脚本：

- [scripts/validate_seg_dataset.py](/root/SeeHydro/scripts/validate_seg_dataset.py)

这个脚本会检查：

- 图像和掩膜是否一一对应
- 尺寸是否一致
- 掩膜是否单通道
- 掩膜类别值是否越界

### 运行命令

```bash
python scripts/validate_seg_dataset.py \
  --image-dir data/seg_water/images \
  --mask-dir data/seg_water/masks \
  --num-classes 2
```

### 如果一切正常，你会看到类似输出

```text
影像数量: 20
掩膜数量: 20
有效配对: 20
掩膜类别值统计: {0: 20, 1: 20}
数据集检查通过
```

### 如果这一步没过

不要开始训练。

先修数据。

最常见的问题是：

- 图片和掩膜名字对不上
- 掩膜尺寸不一致
- 掩膜里出现了不该有的类别值
- 没有标到水面，整张 mask 全是 0

## 11. 第五步：开始训练

仓库里已经准备好一个最适合新手起步的配置：

- [configs/segmentation_binary_water.yaml](/root/SeeHydro/configs/segmentation_binary_water.yaml)

这个配置默认是：

- 二分类
- 模型：`Unet`
- 编码器：`resnet18`
- 输入尺寸：`256`
- 训练轮数：`30`
- batch size：`4`

### 训练命令

```bash
python -m seehydro.cli train segmentation \
  --config configs/segmentation_binary_water.yaml
```

### 它会用哪些数据目录

这个配置默认读取：

- `data/seg_water/images`
- `data/seg_water/masks`

所以你只要前面的转换和检查都通过，这里通常就能直接训。

### 训练结果会放在哪里

默认最佳模型会保存在：

```text
models/seg_water/seg_best.pth
```

### 训练时到底发生了什么

代码会做这些事情：

1. 读取你的图像和掩膜
2. 自动划分训练集和验证集
3. 对图像做归一化
4. 做基础数据增强
5. 开始训练分割模型
6. 每轮计算验证指标
7. 保存效果最好的模型

对应代码在：

- [train_seg.py](/root/SeeHydro/src/seehydro/training/train_seg.py)
- [dataset.py](/root/SeeHydro/src/seehydro/training/dataset.py)

## 12. 训练时你应该怎么看结果

训练日志里你会看到类似这些信息：

- `train_loss`
- `val_loss`
- `mIoU`
- `PA`

### 这些词是什么意思

- `train_loss`
  训练集损失，越小通常越好

- `val_loss`
  验证集损失，越小通常越好

- `mIoU`
  分割里很常用的指标，越大越好

- `PA`
  像素准确率，越大越好

### 对新手来说最重要看什么

先只看两件事：

1. 训练是不是正常结束了
2. `mIoU` 有没有比一开始变好

第一次训练不要太纠结绝对数值。

第一次最重要的是：

- 模型能正常学
- 指标不是完全不动
- 没有明显报错

## 13. 训练完成后你要怎么判断“有没有训好”

你先不要追求“行业最好”。

对第一版模型来说，满足下面这些就算成功：

1. 训练正常结束
2. 成功生成 `seg_best.pth`
3. 验证指标不是 0
4. 标注质量没明显问题
5. 你知道下一步该补哪些数据

### 如果训练结果很差，优先检查什么

按这个顺序排查：

1. 标签名是不是统一写成 `water`
2. 掩膜是不是只有 `0` 和 `1`
3. 图片和掩膜是不是同名配对
4. 你的标注是不是很粗糙
5. 样本是不是太少
6. 图片场景差异是不是太大

## 14. 如果你的原图很大，要不要先拆图

很多遥感原图很大，这时候你会问：

- 我要不要先把大图拆成小图？

答案是：

- **如果原图特别大，通常建议先拆成小图，再标注、再训练**

仓库里已经有“切片”功能，也就是把大图滑窗拆成很多小块：

- [cli.py](/root/SeeHydro/src/seehydro/cli.py#L147)
- [tiling.py](/root/SeeHydro/src/seehydro/preprocessing/tiling.py)

### 切片命令

```bash
python -m seehydro.cli preprocess tile \
  --input /你的tif目录或单个tif \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

### 这条命令适合什么情况

适合：

- 你的 `.tif` 太大
- 你不方便直接对整张图标注
- 你想把大图拆成很多小图块

### 这一步和训练是什么关系

它不是训练本身。

它属于：

- 训练前的数据准备步骤

### 如果你问“怎么尽量保留遥感信息”，这一节就是关键

最推荐的理解方式是：

- `tile` 是当前仓库里最接近“保留 GeoTIFF 空间关系”的步骤
- 所以如果你很在意遥感信息，尽量从 `tile` 开始组织你的数据

也就是：

1. 原始 GeoTIFF 保留不动
2. 用 `tile` 切出小图
3. 基于小图做标注和训练
4. 保留 `tile_index.csv`

这样你后面至少还能追溯：

- 每个切片来自哪张原图
- 它在原图中的位置
- 它对应的空间参考

### 注意

当前仓库里能直接跑通的是：

- `tile`

但下面这个还没有完成：

- `preprocess clip`

也就是说：

- 你可以做“大图切小图”
- 但现在不要指望直接用现成 CLI 做“按线路缓冲区裁剪”

## 15. 训练完以后怎么用这个模型

先把预期说清楚：

- 当前仓库里的 `infer` CLI 还是 `TODO`

所以你现在训练完以后，最现实的用途不是“一键生产部署”，而是：

1. 确认你已经具备一个可继续优化的模型
2. 继续补标注数据
3. 继续训练更好的版本
4. 后续再补推理流程

也就是说，当前项目阶段更像：

- 先把训练闭环跑通

而不是：

- 一键全自动落地系统

## 16. 新手最容易犯的错误

### 错误 1：标签名不统一

必须统一写：

- `water`

### 错误 2：图像和掩膜文件名对不上

必须像这样：

- `images/a001.tif`
- `masks/a001.tif`

### 错误 3：掩膜里出现多余类别值

在二分类里，掩膜应该只有：

- `0`
- `1`

### 错误 4：一开始就做太多类别

不要一开始同时做：

- 水面
- 边坡
- 马道
- 道路
- 桥梁
- 闸门

先只做水面。

### 错误 5：还没检查数据就开始训练

先跑检查脚本，再训练。

### 错误 6：一开始就拿超大规模数据

先用少量样本跑通闭环，再扩大规模。

## 17. 你每天最傻瓜式的实际操作

如果你完全按最保守路线做，每天就是这几步：

1. 往 `labelme_work/` 放新图片
2. 打开 Labelme 标 `water`
3. 保存出 `json`
4. 运行转换脚本
5. 运行检查脚本
6. 开始训练
7. 看模型结果
8. 继续补图、补标注

## 17.1 如果你要“保留遥感信息”，每天按这个版本做

如果你下载的是遥感图，而且你明确希望后续还能做地理分析，那就按下面这套顺序做。

### 第 1 步：先保留原始 GeoTIFF

不要先把原始图改成：

- png
- jpg
- 截图

先把原始下载数据单独放好，例如：

```text
/root/SeeHydro/raw_geotiff/
  scene_001.tif
  scene_002.tif
```

### 第 2 步：检查原始图有没有空间信息

用前面的 `rasterio` 检查命令看：

- `crs`
- `transform`
- `bounds`

### 第 3 步：用 `tile` 切成小图

```bash
python -m seehydro.cli preprocess tile \
  --input raw_geotiff \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

### 第 4 步：保留切片索引

不要删：

- `data/tiles/tile_index.csv`

这个文件后面很重要。

### 第 5 步：从切片里挑一小批先去标注

建议第一批就挑：

- `3 到 10 张`

不要一口气全标。

### 第 6 步：用 Labelme 标这些切片

只标：

- `water`

### 第 7 步：转换成训练数据

```bash
python scripts/convert_labelme_to_masks.py \
  --input-dir labelme_work \
  --output-root data/seg_water
```

### 第 8 步：检查数据

```bash
python scripts/validate_seg_dataset.py \
  --image-dir data/seg_water/images \
  --mask-dir data/seg_water/masks \
  --num-classes 2
```

### 第 9 步：开始训练

```bash
python -m seehydro.cli train segmentation \
  --config configs/segmentation_binary_water.yaml
```

### 第 10 步：保留原始图、切片和索引，不要乱删

因为你后面如果想把结果重新对应回原图，最有价值的就是这些中间资料：

- 原始 GeoTIFF
- 切片 tif
- `tile_index.csv`
- 标注文件
- 训练好的模型

## 18. 一套完整命令抄这里

### 进入项目环境

```bash
cd /root/SeeHydro
source .venv/bin/activate
export PYTHONPATH=/root/SeeHydro/src
```

### 如果你现在什么都还没建，先建目录

```bash
mkdir -p raw_geotiff
mkdir -p data/tiles
mkdir -p data/seg_water/images
mkdir -p data/seg_water/masks
mkdir -p labelme_work
mkdir -p models/seg_water
mkdir -p logs/seg_water
```

### 如果你手里已经有 GeoTIFF，先放到这里

```text
/root/SeeHydro/raw_geotiff/
```

### 先检查原图有没有空间信息

```bash
python - <<'PY'
import rasterio

path = "raw_geotiff/把这里改成你的文件名.tif"

with rasterio.open(path) as src:
    print("width =", src.width)
    print("height =", src.height)
    print("count =", src.count)
    print("crs =", src.crs)
    print("transform =", src.transform)
    print("bounds =", src.bounds)
    print("nodata =", src.nodata)
PY
```

### 先切片

```bash
python -m seehydro.cli preprocess tile \
  --input raw_geotiff \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

### 从切片里挑几张到标注目录

```bash
cp data/tiles/把这里改成某个切片文件名.tif labelme_work/
```

### 京石段起步版一条龙命令

如果你现在就想直接照着跑一次，先按这套执行：

```bash
cd /root/SeeHydro
source .venv/bin/activate
export PYTHONPATH=/root/SeeHydro/src
mkdir -p raw_geotiff data/tiles data/seg_water/images data/seg_water/masks labelme_work models/seg_water logs/seg_water
```

```bash
python - <<'PY'
import ee
from pathlib import Path
from seehydro.acquisition.gee import GEEDownloader

output_path = Path("raw_geotiff/jingshi_sentinel2_small.tif")
bbox = [114.35, 38.20, 114.39, 38.23]

downloader = GEEDownloader()
geometry = ee.Geometry.Rectangle(bbox)
image = downloader.get_sentinel2(
    geometry=geometry,
    date_range=("2024-01-01", "2024-12-31"),
    cloud_pct_max=10,
    bands=["B2", "B3", "B4"],
)
downloader.download_image(
    image=image,
    geometry=geometry,
    output_path=output_path,
    scale=10,
)
print("下载完成:", output_path)
PY
```

```bash
python - <<'PY'
import rasterio
path = "raw_geotiff/jingshi_sentinel2_small.tif"
with rasterio.open(path) as src:
    print("crs =", src.crs)
    print("transform =", src.transform)
    print("bounds =", src.bounds)
PY
```

```bash
python -m seehydro.cli preprocess tile \
  --input raw_geotiff/jingshi_sentinel2_small.tif \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

```bash
find data/tiles -maxdepth 1 -type f -name "*.tif" | sort | head -n 10
```

```bash
cp data/tiles/jingshi_sentinel2_small_r0000_c0000.tif labelme_work/
cp data/tiles/jingshi_sentinel2_small_r0000_c0001.tif labelme_work/
cp data/tiles/jingshi_sentinel2_small_r0001_c0000.tif labelme_work/
```

然后去 Labelme 标这 3 张，标签统一写：

```text
water
```

标完后继续：

```bash
python scripts/convert_labelme_to_masks.py \
  --input-dir labelme_work \
  --output-root data/seg_water
```

```bash
python scripts/validate_seg_dataset.py \
  --image-dir data/seg_water/images \
  --mask-dir data/seg_water/masks \
  --num-classes 2
```

```bash
python -m seehydro.cli train segmentation \
  --config configs/segmentation_binary_water.yaml
```

### 如果你是直接标注普通图片

```bash
python scripts/convert_labelme_to_masks.py \
  --input-dir labelme_work \
  --output-root data/seg_water
```

```bash
python scripts/validate_seg_dataset.py \
  --image-dir data/seg_water/images \
  --mask-dir data/seg_water/masks \
  --num-classes 2
```

```bash
python -m seehydro.cli train segmentation \
  --config configs/segmentation_binary_water.yaml
```

### 如果你手里是超大的 tif，先拆图

```bash
python -m seehydro.cli preprocess tile \
  --input /你的tif目录或单个tif \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

然后你可以从这些切出来的小图里挑一部分去标注，再继续做：

- Labelme 标注
- 转掩膜
- 检查数据
- 训练

## 19. 最推荐你的起步路线

如果你现在还没开始，最推荐你按下面这个顺序来：

1. 先准备 `3 张` 图片
2. 先手工标这 `3 张`
3. 先跑转换脚本
4. 先跑检查脚本
5. 先训练第一版模型
6. 确认能生成 `models/seg_water/seg_best.pth`
7. 再扩到 `20 到 50 张`
8. 再继续训练更稳定的版本

## 19.1 如果你明确要“保留信息”，最推荐路线改成这样

1. 先准备原始 GeoTIFF
2. 先检查 GeoTIFF 是否有 `crs/transform/bounds`
3. 先用 `preprocess tile` 切成小图
4. 保存好 `tile_index.csv`
5. 先挑少量切片做标注
6. 再转换为训练数据
7. 再检查训练数据
8. 再训练第一版模型
9. 原始 tif、切片 tif、索引文件全部保留

## 19.2 如何确认转换后的训练数据真的保留了信息

你不要只凭感觉判断，直接检查输出结果。

例如执行完：

```bash
python scripts/convert_labelme_to_masks.py \
  --input-dir labelme_work \
  --output-root data/seg_water
```

之后，检查某个输出样本：

```bash
python - <<'PY'
import rasterio

for path in [
    "data/seg_water/images/把这里改成你的样本名.tif",
    "data/seg_water/masks/把这里改成你的样本名.tif",
]:
    with rasterio.open(path) as src:
        print(path)
        print("crs =", src.crs)
        print("transform =", src.transform)
        print("bounds =", src.bounds)
        print("nodata =", src.nodata)
        print()
PY
```

### 怎么判断算成功

如果你看到：

- `crs` 不是 `None`
- `transform` 正常
- `bounds` 正常

那就说明这批训练数据已经把空间参考带过来了。

如果这些值是空的，就回去检查这几个问题：

1. `json` 里的 `imagePath` 还能不能找到原始 GeoTIFF
2. 你标注时是不是用了普通 `png/jpg`
3. 标注图尺寸是不是和原始 GeoTIFF 一致

## 20. 一句话总结

对这个仓库最稳的用法不是“直接上大项目”，而是：

- **先用少量数据，把 `标注 -> 转换 -> 检查 -> 训练` 这条最小闭环完整跑通**

只要你把这条线跑顺，后面再做更多类别、更多样本、更多功能，才有基础。
