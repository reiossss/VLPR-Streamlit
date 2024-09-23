import streamlit as st
import cv2
import pymysql
import datetime
import torch
from PIL import Image
import numpy as np
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
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 65 * 8)

    def forward(self, x):
        x = self.base_model(x)
        return x.view(x.size(0), 8, 65)


# 图像处理来定位车牌
def locate_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

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

        if 2 < aspect_ratio < 6:
            return four_point_transform(image, box)
    return None


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxHeight = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
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


# Streamlit App
st.title("车辆管理系统")

# Load model
ccpd_model = CCPDModel()
ccpd_model.load_state_dict(torch.load('ccpd_model.pth', map_location=torch.device('cpu')))
ccpd_model.eval()

# Database connection
db = pymysql.connect(host='localhost', user='root', password='030818YXing', database='parking_system')
cursor = db.cursor()

# Main menu
menu = ["车辆进入模拟", "车辆外出模拟", "车辆信息查询"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "车辆进入模拟":
    st.subheader("车辆进入")
    uploaded_file = st.file_uploader("选择图片...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Selected Image.', use_column_width=True)
        license_plate_img = locate_license_plate(image)
        if license_plate_img is not None:
            license_plate_number = predict(license_plate_img, ccpd_model)
            st.write(f'车牌号: {license_plate_number}')
            st.write("车牌识别成功...")
            sql = f"INSERT INTO cars (license_plate, entry_time, status) VALUES ('{license_plate_number}', '{datetime.datetime.now()}', 'in')"
            try:
                cursor.execute(sql)
                db.commit()
                st.success("车辆已进入")
            except Exception as e:
                db.rollback()
                st.error(f"Error: {e}")

elif choice == "车辆外出模拟":
    st.subheader("车辆外出")
    uploaded_file = st.file_uploader("请选择图片...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Selected Image.', use_column_width=True)
        license_plate_img = locate_license_plate(image)
        if license_plate_img is not None:
            license_plate_number = predict(license_plate_img, ccpd_model)
            st.write(f'车牌号: {license_plate_number}')
            st.write("车牌识别成功...")
            sql = f"SELECT entry_time FROM cars WHERE license_plate='{license_plate_number}' AND status='in'"
            cursor.execute(sql)
            entry_time = cursor.fetchone()
            if entry_time:
                entry_time = entry_time[0]
                current_time = datetime.datetime.now()
                parked_hours = (current_time - entry_time).total_seconds() / 60
                fee = parked_hours * 5
                st.write(f"停车费: {fee:.2f} 元")
                sql = f"UPDATE cars SET exit_time='{current_time}', fee={fee}, status='out' WHERE license_plate='{license_plate_number}' AND status='in'"
                try:
                    cursor.execute(sql)
                    db.commit()
                    st.success("车辆已外出")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error: {e}")
            else:
                st.error("未找到车辆进入记录")

elif choice == "车辆信息查询":
    st.subheader("车辆信息查询")
    st.write("查询所有信息...")
    sql = "SELECT license_plate, entry_time, exit_time, fee, status FROM cars"
    cursor.execute(sql)
    results = cursor.fetchall()
    for row in results:
        st.write(f"车牌号: {row[0]}, 进入时间: {row[1]}, 外出时间: {row[2]}, 费用: {row[3]}, 状态: {row[4]}")

db.close()
