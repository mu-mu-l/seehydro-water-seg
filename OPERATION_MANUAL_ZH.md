# SeeHydro 傻瓜式操作手册

这份手册不是讲概念，是讲你实际要怎么干。

目标很明确：

1. 下载某一段渠道附近的底图
2. 按线路裁剪
3. 切成小图
4. 人工标注水面
5. 把标注整理成训练数据
6. 训练模型
7. 推理
8. 回拼结果
9. 提取中心线和估算水面宽度

这份手册会把每一步分成三类：

- `需要人做`
- `需要代码做`
- `怎么判断这一步做对了`


## 1. 先记住业务边界

这个项目当前最成熟、最稳的方向不是“全要素工程参数自动提取”，而是：

- 渠道水面识别
- 估算水面宽度辅助分析

所以你现在的结果应该理解成：

- 先做出一版能用的水面识别模型
- 再输出中心线和估算水面宽度
- 后续再做人工复核和业务校核

不要一开始就把当前结果当成正式设计参数或测绘成果。


## 2. 整体分工

### 2.1 需要人做

- 选试验段
- 准备或绘制线路文件
- 决定下载区域 `bbox`
- 决定缓冲区宽度 `buffer`
- 用 Labelme 做人工标注
- 判断标注质量是否过关
- 判断模型结果是否符合业务常识

### 2.2 需要代码做

- 下载在线底图
- 裁剪影像
- 切片
- 把 Labelme 标注转换成训练数据
- 检查训练数据
- 训练模型
- 批量推理
- 按切片索引回拼整图掩膜
- 从掩膜提取中心线和估算水面宽度

### 2.3 需要人和代码配合做

- 线路文件准备
- 小范围试验段选择
- 标注样本迭代
- 模型结果业务校核


## 3. 先准备什么

你至少要准备这几样东西：

1. 一个试验段
2. 一个线路文件
3. 一个可用的天地图密钥
4. Labelme
5. Python 环境

建议目录先准备成这样：

```text
/root/SeeHydro/
  raw_geotiff/
  data/route/
  data/tiles/
  data/seg_water/images/
  data/seg_water/masks/
  labelme_work/
  models/seg_water/
  outputs/
```


## 4. 第一步：准备线路文件

### 4.1 需要人做

- 你要确定自己关心的是哪一段渠道
- 你要准备一条沿渠道走向的大致中心线

线路文件可以来自：

- 现成 GeoJSON / Shapefile
- 奥维画出来的 shp
- QGIS 自己画出来的 GeoJSON / shp

最推荐的是：

- 画一条线
- 不是点
- 不是面

### 4.2 需要代码做

代码不会替你决定画哪一段线，但能读取：

- `.geojson`
- `.shp`

### 4.3 怎么判断做对了

满足下面几条就够用了：

- 线大体沿着渠道走
- 文件能在 QGIS/奥维里正常打开
- shp 配套文件齐全
  典型包括 `.shp`、`.shx`、`.dbf`
- 最好带 `.prj`

建议保存成：

```text
data/route/snbd_centerline.geojson
```


## 5. 第二步：下载底图

### 5.1 需要人做

- 决定一个小范围 `bbox`
- 确认这个范围里确实有你关心的渠道

第一次不要下太大，先下一个很小的试验块。

### 5.2 需要代码做

用天地图密钥下载在线底图并拼接成 GeoTIFF。

```bash
cd /root/SeeHydro
export TDT_KEY=你的天地图密钥

python -m seehydro.cli download tiles \
  --bbox 114.35,38.20,114.39,38.23 \
  --provider tianditu_img \
  --zoom 18 \
  --output-dir raw_geotiff
```

### 5.3 怎么判断做对了

看这几件事：

- `raw_geotiff/` 里生成了 `.tif`
- 图能正常打开
- 能看清渠道水面
- 影像范围没偏得离谱


## 6. 第三步：按线路裁剪

### 6.1 需要人做

- 你要决定裁多宽

一般建议：

- 只做水面分割起步：`500m` 到 `1000m`
- 想多保留一点周边环境：`2000m`

### 6.2 需要代码做

```bash
python -m seehydro.cli preprocess clip \
  --input raw_geotiff \
  --route data/route/snbd_centerline.geojson \
  --buffer 1000 \
  --output data/clipped
```

### 6.3 怎么判断做对了

- `data/clipped/` 里有裁好的 `.tif`
- 打开后能看到渠道在图里
- 背景明显比原图少
- 没把渠道裁掉


## 7. 第四步：切片

### 7.1 需要人做

- 你要决定切片大小和重叠比例

建议起步：

- `size=512`
- `overlap=0.25`

### 7.2 需要代码做

```bash
python -m seehydro.cli preprocess tile \
  --input data/clipped \
  --size 512 \
  --overlap 0.25 \
  --output data/tiles
```

### 7.3 怎么判断做对了

- `data/tiles/` 下出现大量切片 `.tif`
- 有 `tile_index.csv`
- 随机打开几张切片能看到渠道


## 8. 第五步：人工标注

### 8.1 需要人做

这是当前最关键的人工作业。

你要做的是：

1. 从 `data/tiles/` 里挑一小批图复制到 `labelme_work/`
2. 用 Labelme 打开这些图
3. 只标水面
4. 标签统一写成：

```text
water
```

### 8.2 需要代码做

代码不替你标注，这一步必须人做。

### 8.3 怎么判断做对了

- `labelme_work/` 里有 `.tif`
- 同目录里有对应 `.json`
- 标签名统一是 `water`
- 水面边界没有画得太粗糙

第一次建议只标：

- `5` 到 `10` 张

不要一开始就标很多。


## 9. 第六步：把标注整理成训练数据

### 9.1 需要人做

- 确认 `labelme_work/` 里已经有图和对应的 json

### 9.2 需要代码做

现在已经补成一条命令，不需要再手动跑两个脚本。

```bash
python -m seehydro.cli train prepare-seg-data \
  --labelme-dir labelme_work \
  --output-root data/seg_water \
  --water-label water \
  --num-classes 2
```

这条命令会自动做：

1. 把 Labelme JSON 转成：
   - `data/seg_water/images/*.tif`
   - `data/seg_water/masks/*.tif`
2. 自动检查：
   - 图像和掩膜是否配对
   - 尺寸是否一致
   - 掩膜类别值是否合法

### 9.3 怎么判断做对了

- `data/seg_water/images/` 里有图
- `data/seg_water/masks/` 里有掩膜
- 命令最后显示“数据集检查通过”


## 10. 第七步：训练模型

### 10.1 需要人做

- 确认训练数据已经通过检查

### 10.2 需要代码做

```bash
python -m seehydro.cli train segmentation \
  --config configs/segmentation_binary_water.yaml
```

### 10.3 怎么判断做对了

- 训练过程正常跑完
- `models/seg_water/seg_best.pth` 生成了


## 11. 第八步：推理

### 11.1 需要人做

- 确认推理时用的模型配置和训练配置一致

### 11.2 需要代码做

```bash
python -m seehydro.cli infer \
  --input data/tiles \
  --config configs/segmentation_binary_water.yaml \
  --model-seg models/seg_water/seg_best.pth \
  --output outputs/infer
```

### 11.3 怎么判断做对了

- `outputs/infer/segmentation/` 下有切片级预测掩膜
- 如果输入目录有 `tile_index.csv`
  还会生成：
  - `outputs/infer/merged/*_merged_mask.tif`


## 12. 第九步：提取中心线和估算水面宽度

### 12.1 需要人做

- 你要知道这里提取的是“估算水面宽度”
- 不是正式设计断面宽度

### 12.2 需要代码做

```bash
python -m seehydro.cli extract \
  --input outputs/infer/merged \
  --output outputs/extraction \
  --sample-interval 50
```

注意：

- `extract` 现在只处理 `*_merged_mask.tif`

### 12.3 怎么判断做对了

你应该看到这些结果：

- `outputs/extraction/vectors/*_water_mask.geojson`
- `outputs/extraction/vectors/*_centerline.geojson`
- `outputs/extraction/vectors/*_width_profile.geojson`
- `outputs/extraction/vectors/*_berm_mask.geojson`（如果有马道类别）
- `outputs/extraction/vectors/*_berm_width_profile.geojson`（如果能提取到马道宽度）
- `outputs/extraction/reports/*_summary.csv`
- `outputs/extraction/reports/*_summary.xlsx`
- `outputs/extraction/summary.json`

你要重点看：

- 掩膜面和中心线是不是明显错位
- 宽度采样点是不是大体落在水面中轴附近
- 如果某张图没有提出来中心线，至少 `water_mask` 面结果还在
- 报表里的宽度是“估算值”，不是正式设计值


## 13. 第十步：导出结果

### 13.1 需要人做

- 你决定导出成什么格式

### 13.2 需要代码做

```bash
python -m seehydro.cli export \
  --input outputs/extraction \
  --format shapefile \
  --report outputs/export_reports
```

也可以直接传：

```bash
--input outputs/extraction/vectors
```

### 13.3 怎么判断做对了

- 导出了目标格式文件
- 如果传了 `--report`
  还会生成汇总报表
- `shapefile` 字段名不会因为过长而被导坏
- `report` 现在统一是同一套列结构：
  `类别 / 子类 / 数量 / 指标项 / 指标值 / 单位 / 备注`


## 14. 一条最常用的完整业务链

你当前最现实的一条业务链是：

1. 人选试验段
2. 人准备线路文件
3. 代码下载底图
4. 代码按线路裁剪
5. 代码切片
6. 人挑图并做 Labelme 标注
7. 代码整理训练数据
8. 代码训练模型
9. 代码推理并回拼
10. 代码提取中心线和估算水面宽度
11. 人做业务判断和人工复核


## 15. 一条最省事的流水线命令

如果你想把“下载、裁剪、切片、标注后整理训练数据、推理、提辅助结果”尽量串起来，可以用：

```bash
python -m seehydro.cli pipeline quickstart \
  --bbox 114.35,38.20,114.39,38.23 \
  --provider tianditu_img \
  --api-key 你的天地图密钥 \
  --route data/route/snbd_centerline.geojson \
  --config configs/segmentation_binary_water.yaml \
  --labelme-dir labelme_work \
  --seg-output-root data/seg_water \
  --model-seg models/seg_water/seg_best.pth \
  --workspace outputs/pipeline_run
```

但要记住：

- 标注本身还是你手做
- 代码不能替你画标签
- 这条流水线现在会尽量输出和单独执行 `extract` 一致的结果：
  `water_mask / centerline / width_profile / berm_mask / berm_width_profile / summary.json`


## 16. 最容易踩的坑

### 16.1 人工环节最容易出错的地方

- 线路文件画偏
- 标签名不统一
- 标注太粗糙
- 标了普通 png/jpg，却又想继承 GeoTIFF 空间信息

### 16.2 代码环节最容易出错的地方

- 推理没带训练配置
- 把原始影像目录直接拿去 `extract`
- 目录里混了很多不该处理的 tif
- 模型权重和配置不匹配
- 看到宽度结果就直接当正式设计参数


## 17. 你现在最应该怎么干

如果你现在是第一次真正动手，建议只做下面这些：

1. 先准备一条线路
2. 先下载一个小 bbox 的天地图底图
3. 先裁剪
4. 先切片
5. 先标 5 到 10 张
6. 先整理成训练数据
7. 先训出第一版模型
8. 先看推理和估算水面宽度结果是不是像样

不要一开始就追求：

- 全线
- 全要素
- 高精度正式成果


## 18. 一句话总结

这个项目现在最稳的打法不是“全自动工程参数生产”，而是：

- 人工选段
- 人工标注
- 代码完成数据处理、训练、推理和基础辅助分析
- 最终由人做业务判断和人工复核

如果只记一句：

```text
人负责判断和标注，代码负责批处理和重复劳动。
```
