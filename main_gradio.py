import gradio as gr
from PIL import Image
from ultralytics import YOLO


def predict_image(img):
    # 转换PIL图像为RGB
    if img.mode != "RGB":
        img = img.convert('RGB')

    # 使用YOLOv8模型进行预测
    model = YOLO('runs/detect/train3/weights/best.pt')
    results = model.predict(source=img, conf=0.25, show=False)  ## yolo参数设置
    im_array = results[0].plot()

    # 转换结果为PIL图像并返回
    pil_img = Image.fromarray(im_array[..., ::-1])
    return pil_img


# 创建Gradio界面
interface = gr.Interface( 
    outputs='image',
    examples=["./image/T0001_XM_20110807100242_01.jpg"],
    title="Real-Time Object Detection with YOLOv8",
    description="Upload an image to detect objects using Yolov8"
)  # gradio参数设置

# 启动界面
interface.launch()