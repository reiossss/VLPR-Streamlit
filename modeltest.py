import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
from torchvision import models

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 定义字符索引
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


# 定义模型
class CCPDModel(nn.Module):
    def __init__(self):
        super(CCPDModel, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.base_model = models.resnet18(weights=weights)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 65 * 8)  # 8个字符，每个字符有65个类别

    def forward(self, x):
        x = self.base_model(x)
        return x.view(x.size(0), 8, 65)  # 将输出调整为 (batch_size, 8, 65)


# 图像处理来定位车牌
def locate_license_plate(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 边缘检测
    edged = cv2.Canny(blurred, 100, 200)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate_img = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])

        if width > height:
            aspect_ratio = width / height
        else:
            aspect_ratio = height / width

        if 2 < aspect_ratio < 6:  # 根据实际情况调整车牌的宽高比
            license_plate_img = four_point_transform(image, box)
            break

    # 显示分割结果
    if license_plate_img is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Detected License Plate")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.title("Segmented License Plate")
        plt.imshow(cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2RGB))
        plt.show()

    return license_plate_img


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# 定义预测函数
def predict(image_array, model):
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 2)
        preds = preds.squeeze().tolist()

    license_plate = provinces[preds[0]] + alphabets[preds[1]] + ''.join([ads[idx] for idx in preds[2:]])
    return license_plate


# 主函数
def main(image_path):
    # 加载车牌识别模型
    model = CCPDModel()
    model.load_state_dict(torch.load('ccpd_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # 定位和分割车牌
    license_plate_img = locate_license_plate(image_path)

    # 识别车牌
    if license_plate_img is not None:
        license_plate = predict(license_plate_img, model)
        print(f'Recognized License Plate: {license_plate}')
    else:
        print("未识别到车牌")


# 示例用法
if __name__ == "__main__":
    image_path = 'D:/car5.jpg'  # 替换为你的图像路径
    main(image_path)
