# YOLO融合Deepseek的高压输电线缺陷检测系统   |高压输电线 缺陷检测 YOLOv11 DeepSeek 电力巡检 9183期

## 系统概述

本系统基于 **YOLOv11 深度学习模型**，融合 **DeepSeek、Pytorch、SpringBoot、Vue、MySQL** 等技术，构建了一套支持**图片、视频、摄像头**三种检测方式的高压输电线缺陷检测平台。系统可快速识别绝缘子、瓷瓶等关键部件的缺陷，适配电力线路维护、故障检测及智能电网应用场景。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8f4a3003ae0741d0a66c50c39494791f.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9470223b1f074a3e93666215c08e67a6.png)

## 系统核心信息

| 项目 | 详细说明 |
| :--- | :--- |
| **检测类别** | 涵盖绝缘子破损、瓷瓶缺陷等关键部件缺陷 |
| **核心技术** | YOLOv11 + DeepSeek + Pytorch + SpringBoot + Vue3 |
| **检测模式** | 图片检测、视频检测、摄像头实时检测 |
| **系统架构** | 前后端分离（Vue3 + SpringBoot + Flask + MySQL） |
| **应用价值** | 1. **电力巡检**：支撑高压输电线关键部件的自动化缺陷检测，替代人工登塔检查。<br>2. **故障诊断**：识别绝缘子破损、瓷瓶缺陷等隐患，辅助运维决策。<br>3. **智能电网**：为电力系统运行管理提供视觉识别能力。<br>4. **AI 分析**：结合 DeepSeek 大语言模型生成分析报告与维护建议。 |


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c00a3ca6b05d4eda80528b45831fad2e.png)

## 技术栈

| 层次 | 技术选型 |
| :--- | :--- |
| **前端** | Vue3、Element-Plus、TypeScript |
| **后端** | SpringBoot、MyBatis-Plus、Flask |
| **深度学习** | YOLOv11、Pytorch |
| **数据库** | MySQL |
| **视频处理** | FFmpeg |
| **AI 分析** | DeepSeek |


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/774a95dfcd6049168caf689d896dbe8d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f8b91f0fe5bc491a8b01eca6ad334fd4.png)

## 系统功能

**1. 多方式检测**
- **图片检测**：单张图片上传识别
- **视频检测**：上传视频，逐帧分析并输出标注视频
- **摄像头实时检测**：连接摄像头，实时输出检测结果

**2. 高压输电线缺陷识别**
- 可识别绝缘子破损、瓷瓶缺陷等关键部件
- 使用 YOLOv11 模型，支持多种权重的目标检测
- 检测精度高、速度快

**3. 实时监控与预测**
- 支持通过摄像头进行实时监测
- 自动识别输电线缺陷并触发预警

**4. 模型训练与自定义**
- 用户可以使用自定义数据集重新训练模型
- 生成新的权重文件，支持 YOLOv11 的训练

**5. 用户管理**
- 支持分用户管理
- 管理员可查看所有检测记录，普通用户只看自己的记录

**6. 简洁美观的 UI**
- 采用 Vue3 前端框架
- 支持主题颜色、布局和组件大小的个性化定制
- 兼容深色模式与色弱模式
- 检测结果可导出 PDF 报告
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/905200f447034128b8634af9af1fa057.png)

## 快速开始（代码示例）

**1. 数据集配置文件 (`data.yaml`)**
```yaml
train: ./dataset/images/train
val: ./dataset/images/val
nc: 2
names: ['insulator_damage', 'porcelain_defect']
```


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a59c5de12a454c258d7cd5b07c79a755.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4d1e4bc421634f3cacb203de678c76d6.png)

**2. YOLOv11 训练代码**
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

**3. Flask 推理与 AI 分析服务**
```python
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO('runs/detect/train/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': model.names[int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist()
        })
    # 调用 DeepSeek 生成缺陷分析建议
    ai_suggestion = analyze_with_llm(detections)
    return jsonify({'detections': detections, 'suggestion': ai_suggestion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**4. Vue3 前端检测页面**
```vue
<template>
  <div class="detector-container">
    <el-upload action="/api/detection/image" :on-success="handleSuccess">
      <el-button type="primary">上传图片</el-button>
    </el-upload>
    <div v-if="resultImage" class="result-area">
      <img :src="resultImage" alt="Detection Result" />
      <el-table :data="detections" border>
        <el-table-column prop="class" label="缺陷类别" />
        <el-table-column prop="confidence" label="置信度" />
      </el-table>
      <el-button @click="exportPDF">导出 PDF 报告</el-button>
    </div>
  </div>
</template>
```

## 标签
`高压输电线` `缺陷检测` `YOLOv11` `DeepSeek` `电力巡检` `目标检测` `深度学习` `计算机视觉`
