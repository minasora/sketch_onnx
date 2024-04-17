from PIL import Image
import numpy as np

def load_image(image_path):

    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img).transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 假设图像路径
image_path = "./15.jpg"
input_tensor = load_image(image_path)
import onnxruntime as ort

def infer(input_tensor, model_path):

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(input_name, output_name)
    result = session.run([output_name], {input_name: input_tensor})
    return result[0]


model_path = "./model.onnx"
output_tensor = infer(input_tensor, model_path)

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
def postprocess_and_visualize(output_tensor):
    # 转换 ONNX 输出张量为 PyTorch 张量
    output_tensor = torch.tensor(output_tensor)
    
    # 假设输出也是 1x3x256x256 格式的图像
    # 将张量转换为 PIL 图像
    to_pil_image = transforms.ToPILImage()
    output_image = to_pil_image(output_tensor.squeeze(0))  # 移除批次维度

    # 显示图像
    plt.imshow(output_image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()

# 使用该函数代替之前的 visualize_output 函数
postprocess_and_visualize(output_tensor)

