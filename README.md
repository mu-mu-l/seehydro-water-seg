# SeeHydro Water Seg

面向零基础用户的渠道水面分割训练项目。

这个仓库当前最适合先做一件事：

- 用你自己的遥感影像
- 自己做水面标注
- 把标注转换成训练数据
- 用脚本训练一个最基础的水面分割模型

先不要一上来做桥梁检测、闸门识别、参数提取全流程。  
最稳的路线是先把 `二分类水面分割` 跑通。

## 1. 你最终要做成什么

你最后会得到一个训练好的模型权重文件，比如：

- `models/seg_water/seg_best.pth`

这个模型的目标很简单：

- 输入一张影像
- 输出哪里是水面，哪里不是水面

类别只有两类：

- `0 = background`
- `1 = canal_water`

## 2. 你必须人工做的事

这部分不能指望脚本自动完成：

1. 你自己准备原始影像
2. 你自己决定先识别什么
3. 你自己做标注
4. 你自己检查标注质量

对新手来说，最合理的起步就是：

- 先只做 `水面`
- 不做桥梁检测
- 不做 5 类分割
- 不做太大数据量

## 3. 需要什么软件

### 必装软件

1. `Labelme`
   用来做标注。

2. `Python 3.10`
   用来运行项目脚本。

3. `Git`
   用来拉代码和更新仓库。

### Labelme 是干什么的

它是一个标注工具。  
你在图上把水面区域圈出来，保存后会得到一个 `json` 文件。

你不用一开始理解深度学习，只要理解成：

- 图像 = 题目
- 标注 = 标准答案
- 模型 = 学做题的人

## 4. 零基础完整流程

整个流程是：

1. 准备影像
2. 用 Labelme 标注
3. 把 Labelme 的 `json` 转成 `masks`
4. 检查数据格式
5. 开始训练

## 5. 第一步：准备影像

先准备一小批影像，不要太多。

建议第一批只准备：

- `20 到 50 张`

要求：

- 尽量是同类区域
- 尽量画面清晰
- 尽量都是同一类分辨率
- 先别拿特别大的全线数据来折腾

你可以先建一个工作目录，比如：

```text
labelme_work/
```

把你准备标注的影像放进去：

```text
labelme_work/
  a001.png
  a002.png
  a003.png
```

如果你的图像是 `jpg`、`png` 也没关系。  
标注阶段先能用就行，后面转换脚本会帮你整理成训练数据。

## 6. 第二步：用 Labelme 做标注

### 你要标什么

只标一个标签：

- `water`

不要写成别的名字，比如：

- `Water`
- `canal`
- `river`
- `水面`

先统一只用：

- `water`

### 怎么标

1. 打开 `Labelme`
2. 点 `Open Dir`
3. 选择你的 `labelme_work/`
4. 打开一张图
5. 选择 `Create Polygons`
6. 沿着水面边缘一点一点画
7. 闭合后输入标签名：`water`
8. 保存

### 标注原则

你只记住一条：

- 水面就标成 `water`
- 不是水面的都不标

先不要纠结太复杂的类别。

### 标完后会得到什么

例如你标了一张 `a001.png`，通常会得到：

```text
labelme_work/
  a001.png
  a001.json
```

这个 `json` 还不能直接训练，它只是标注描述文件。  
还要转换成掩膜图。

## 7. 第三步：把 Labelme JSON 转成训练数据

项目里已经有脚本：

- `scripts/convert_labelme_to_masks.py`

它会把 Labelme 的 `json` 批量转换成：

```text
data/seg_water/images/
data/seg_water/masks/
```

### 运行命令

先进入项目目录并激活环境：

```bash
cd /root/SeeHydro
source .venv/bin/activate
export PYTHONPATH=/root/SeeHydro/src
```

然后运行转换：

```bash
python scripts/convert_labelme_to_masks.py \
  --input-dir labelme_work \
  --output-root data/seg_water
```

### 转换后会得到什么

例如：

```text
data/seg_water/images/a001.tif
data/seg_water/masks/a001.tif
data/seg_water/images/a002.tif
data/seg_water/masks/a002.tif
```

这里的要求是：

- 图像和掩膜文件名必须一样
- 掩膜是单通道
- 掩膜值应该只有 `0` 和 `1`

含义是：

- `0 = 背景`
- `1 = 水面`

## 8. 第四步：检查数据是不是能训练

项目里已经有检查脚本：

- `scripts/validate_seg_dataset.py`

它会检查：

- 图像和掩膜是不是一一对应
- 尺寸是不是一致
- 掩膜是不是单通道
- 类别值有没有越界

### 运行命令

```bash
python scripts/validate_seg_dataset.py \
  --image-dir data/seg_water/images \
  --mask-dir data/seg_water/masks \
  --num-classes 2
```

### 通过时会看到

类似：

```text
影像数量: 20
掩膜数量: 20
有效配对: 20
掩膜类别值统计: {0: 20, 1: 20}
数据集检查通过
```

如果这一步没通过，不要训练。  
先修数据。

## 9. 第五步：开始训练

项目里已经准备好一个适合新手起步的配置：

- `configs/segmentation_binary_water.yaml`

这份配置默认是：

- 二分类
- `Unet`
- `resnet18`
- `input_size = 256`

### 训练命令

```bash
python -m seehydro.cli train segmentation \
  --config configs/segmentation_binary_water.yaml
```

### 训练结果会放在哪里

默认会输出到：

```text
models/seg_water/seg_best.pth
```

## 10. 你第一次训练时应该怎么做

不要一上来就追求最好效果。  
第一次训练的目标不是“做出最强模型”，而是：

- 确认数据没问题
- 确认训练流程能跑
- 确认模型能学到东西

最稳的第一次做法：

1. 先标 `20 到 50 张`
2. 跑数据检查
3. 开始训练
4. 看训练是否能正常结束
5. 再补更多数据

## 11. 新手最容易犯的错

### 错误 1：标签名不统一

必须统一用：

- `water`

### 错误 2：图像和掩膜文件名对不上

必须像这样：

- `images/a001.tif`
- `masks/a001.tif`

### 错误 3：掩膜里不止 0 和 1

这个会直接影响训练。

### 错误 4：一开始就做太多类别

不要一开始做：

- 水面
- 边坡
- 马道
- 道路
- 桥梁
- 闸门

先只做水面。

### 错误 5：一开始就做太多数据

先用少量数据跑通，再扩大规模。

## 12. 你每天实际怎么操作

如果你完全按傻瓜式做，日常流程就是：

1. 往 `labelme_work/` 放新图片
2. 打开 Labelme 标 `water`
3. 保存出 `json`
4. 运行转换脚本
5. 运行检查脚本
6. 开始训练

## 13. 一套完整命令抄这里

### 进入项目环境

```bash
cd /root/SeeHydro
source .venv/bin/activate
export PYTHONPATH=/root/SeeHydro/src
```

### Labelme 标完后转换

```bash
python scripts/convert_labelme_to_masks.py \
  --input-dir labelme_work \
  --output-root data/seg_water
```

### 检查数据

```bash
python scripts/validate_seg_dataset.py \
  --image-dir data/seg_water/images \
  --mask-dir data/seg_water/masks \
  --num-classes 2
```

### 开始训练

```bash
python -m seehydro.cli train segmentation \
  --config configs/segmentation_binary_water.yaml
```

## 14. 现在最推荐你做什么

如果你是第一次接触深度学习训练，最推荐的顺序是：

1. 先准备 `3 张` 图
2. 先标这 `3 张`
3. 先跑转换脚本
4. 先跑检查脚本
5. 确认没问题后，再扩到 `20 到 50 张`
6. 再正式训练

不要跳步骤。  
只要你把这条线跑顺，后面再扩展到更多数据就容易很多。
