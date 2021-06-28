# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import platform
from PIL import Image
debug = False


class DigitsNet(nn.Module):
    def __init__(self):
        super(DigitsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.tns1 = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.tns2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=32)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, padding=1)
        self.gpool = nn.AvgPool2d(kernel_size=7)
        #self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        #x = self.tns1(self.drop(self.bn1(F.relu(self.conv1(x)))))
        #x = self.drop(self.bn2(F.relu(self.conv2(x))))
        x = self.tns1(self.bn1(F.relu(self.conv1(x))))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)
        #x = self.drop(self.bn3(F.relu(self.conv3(x))))
        #x = self.drop(self.bn4(F.relu(self.conv4(x))))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.tns2(self.pool2(x))
        #x = self.drop(self.bn5(F.relu(self.conv5(x))))
        #x = self.drop(self.bn6(F.relu(self.conv6(x))))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))

        x = self.conv7(x)
        x = self.gpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class OCR:
    def __initVars__(self):
        self.idx = [
            ' ', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л',
            'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ',
            'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
        self.idxA = ['0', '1']
        self.idxD = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.RealAnwers = [[2, 2, 1, 2, 4, 4, 4, 4, 1, 1, 4, 2, 2, 1, 2, 1, 4, 4, 2, 1,
                            4, 1, 2, 2, 2, 1, 2, 4, 1, 2, 4, 1, 1, 2, 4, 4, 2, 4, 1, 2,
                            2, 4, 8, 2, 1, 8, 1, 4, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [2, 1, 2, 4, 4, 4, 2, 2, 4, 1, 2, 2, 4, 1, 4, 1, 2, 4, 2, 2,
                            2, 1, 4, 2, 2, 2, 4, 2, 1, 1, 2, 2, 2, 4, 4, 2, 4, 2, 1, 2,
                            4, 4, 2, 1, 4, 2, 8, 4, 4, 2, 2, 4, 4, 1, 2, 1, 4, 2, 1, 4],
                           [2, 1, 4, 2, 2, 2, 1, 2, 1, 1, 4, 2, 1, 4, 2, 2, 4, 1, 1, 4,
                            1, 2, 4, 1, 4, 4, 2, 4, 1, 4, 2, 2, 1, 4, 1, 1, 2, 2, 2, 4,
                            4, 1, 2, 2, 4, 1, 4, 4, 1, 2, 4, 4, 4, 2, 4, 8, 2, 2, 8, 2],
                           [2, 4, 2, 4, 4, 2, 4, 2, 2, 1, 1, 2, 1, 1, 2, 2, 4, 2, 1, 2,
                            4, 1, 2, 2, 1, 1, 2, 1, 4, 2, 2, 4, 4, 1, 2, 1, 2, 4, 2, 2,
                            1, 1, 2, 4, 4, 1, 1, 2, 4, 1, 1, 4, 4, 2, 8, 1, 2, 2, 4, 4]]

    def __init__(self, file):
        if not os.path.isfile(file):
            raise Exception('No file found')
        self.__initVars__()
        if (platform.system()=='Windows'):
            pil_img = Image.open(file)
            self.img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        else:
            self.img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if self.img.shape[0] != 1700:
            self.img = cv2.resize(self.img, (1700, 2800))
        # img = cv2.fastNlMeansDenoising(img)# img = img[:,:,0]#-img[:,:,0]
        self.network = DigitsNet()
        self.network.load_state_dict(torch.load('data/digits.dict', map_location=torch.device('cpu')))
        self.network.eval()
        self.input_size = 224
        self.__preprocess__()
        self.__initFioCNN__()
        self.__initAnswCNN__()

    def __preprocess__(self):
        img_b = self.img  # cv2.blur(self.img, (2,2))
        _, thresh = cv2.threshold(img_b, 170, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)
        contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centersArray = np.ndarray(shape=(0, 2), dtype=float)
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.015 * cv2.arcLength(c, True), True)
            if cv2.contourArea(c) > 1000 and cv2.contourArea(c) < 1500 and len(approx) == 4:
                contours_poly = cv2.approxPolyDP(c, 3, True)
                centers, radius = cv2.minEnclosingCircle(contours_poly)
                centersArray = np.append(centersArray, [centers], axis=0)
                if debug:
                    print(centers, 'area=', cv2.contourArea(c))
                    cv2.putText(img, f"{round(cv2.contourArea(c))}", (int(centers[0]), int(centers[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.circle(self.img, (int(centers[0]), int(centers[1])), int(radius), (0, 0, 0), 2)
        assert (len(centersArray) == 6), len(centersArray)
        sorted = np.sort(centersArray.view('f8,f8'), order=['f1', 'f0'], axis=1).view(np.float64)
        for i in range(0, 5, 2):
            if sorted[i, 0] > sorted[i + 1, 0]:
                sorted[[i, i + 1]] = sorted[[i + 1, i]]
        needRotate = (sorted[2, 1] - sorted[4, 1]) / (sorted[0, 1] - sorted[4, 1]) > 0.6
        dx = np.array([(sorted[1, 0] - sorted[0, 0]) * 0.05, 0])
        src = np.array([sorted[0] - dx, sorted[1] + dx, sorted[4] - dx, sorted[5] + dx], dtype=np.float32)
        dst = np.array([[0, 1780], [1100, 1780], [0, 0], [1100, 0]], dtype=np.float32)
        pt = cv2.getPerspectiveTransform(src, dst)
        self.img = cv2.warpPerspective(self.img, pt, dsize=(1100, 1780))  # ,flags=cv2.INTER_NEAREST
        self.colorImg = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        # self.img = cv2.fastNlMeansDenoising(self.img)# ,None,10,10,7,21)
        _, self.pr_img = cv2.threshold(self.img, 225, 255, cv2.THRESH_BINARY_INV)
        if needRotate:
            self.colorImg = cv2.rotate(self.colorImg, cv2.ROTATE_180)
            self.img = cv2.rotate(self.img, cv2.ROTATE_180)
            self.pr_img = cv2.rotate(self.pr_img, cv2.ROTATE_180)

    def __initFioCNN__(self):
        self.num_classes = 33
        self.model_ft = torchvision.models.resnet50(pretrained=False)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(self.num_ftrs, self.num_classes)
        self.model_ft.load_state_dict(torch.load('data/comnist.dict', map_location=torch.device('cpu')))
        self.model_ft.eval()

    def __initAnswCNN__(self):
        self.AnswModel = torchvision.models.resnet18(pretrained=False)
        self.numansw_ftrs = self.AnswModel.fc.in_features
        self.AnswModel.fc = nn.Linear(self.numansw_ftrs, 2)
        self.AnswModel.load_state_dict(torch.load('data/answ.dat', map_location=torch.device('cpu')))
        self.AnswModel.eval()

    def remove_isolated_pixels(self, image):
        connectivity = 8
        output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)
        num_stats = output[0]
        labels = output[1]
        stats = output[2]
        new_image = image.copy()
        # print(num_stats)
        for label in range(num_stats):
            if stats[label, cv2.CC_STAT_AREA] <= 400:
                new_image[labels == label] = 0
        return new_image

    def getTeacherImg(self, n):
        x = round(394.8 + 50 * n)
        y = 400
        dx = 41
        dy = 68
        ROI = self.img[y:y + dy, x:x + dx]
        ROI = cv2.fastNlMeansDenoising(ROI)
        _, ROI = cv2.threshold(ROI, 235, 255, cv2.THRESH_BINARY_INV)
        old_size = ROI.shape[:2]  # old_size is in (height, width) format
        ratio = float(self.input_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        ROI = cv2.resize(ROI, (new_size[1], new_size[0]))
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        ROI = cv2.copyMakeBorder(ROI, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        ROI = self.remove_isolated_pixels(ROI)
        ROI = cv2.resize(ROI, (28, 28))
        return ROI

    def getLevelImg(self, n):
        deltax = 5
        deltay = 5
        x = round(284 + 58 * n) - deltax
        y = 562 - deltay
        ROI = self.pr_img[y:y + 42 + deltay, x:x + 38 + deltax]
        ROI = cv2.resize(ROI, (228, 228))
        return ROI

    def getAnswerBR(self, n):
        deltax = 5
        deltay = 5
        x1 = round(69 + 48.2 * (n % 20) + 48.5 * ((n // 10) % 2)) - 3 - deltax
        y1 = round(908 + 57.5 * (n // 20) + 72 * ((n // 80) % 3)) - 3 - deltay
        x2 = x1 + 38 + deltax
        y2 = y1 + 42 + deltay
        return (x1, x2, y1, y2)

    def getAnswerImg(self, n):
        br = self.getAnswerBR(n)
        ROI = self.pr_img[br[2]:br[3], br[0]:br[1]]
        ROI = cv2.fastNlMeansDenoising(ROI)  # ,None,10,10,7,21)
        ROI = cv2.resize(ROI, (228, 228))
        return ROI

    def getAnswerImg(self, n):
        deltax = 5
        deltay = 5
        x = round(69 + 48.2 * (n % 20) + 48.5 * ((n // 10) % 2)) - 3 - deltax
        y = round(908 + 57.5 * (n // 20) + 72 * ((n // 80) % 3)) - 3 - deltay
        ROI = self.pr_img[y:y + 42 + deltay, x:x + 38 + deltax]
        ROI = cv2.fastNlMeansDenoising(ROI)  # ,None,10,10,7,21)
        ROI = cv2.resize(ROI, (228, 228))
        return ROI

    def getFIOImg(self, n):
        deltax = 18
        deltay = 32
        x = round(183 + 51.5 * (n % 17)) - 4
        y = round(206 + 105 * (n // 17)) - 8
        dx = 39 + deltax
        dy = 59 + deltay
        # ROI = cv2.resize(self.pr_img[y:y+dy, x:x+dx],(c, self.input_size)) # cv2.bitwise_not(
        ROI = self.img[y:y + dy, x:x + dx]
        ROI = cv2.fastNlMeansDenoising(ROI)  # ,None,10,10,7,21)
        _, ROI = cv2.threshold(ROI, 235, 255, cv2.THRESH_BINARY_INV)
        old_size = ROI.shape[:2]  # old_size is in (height, width) format
        ratio = float(self.input_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        ROI = cv2.resize(ROI, (new_size[1], new_size[0]))
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        ROI = cv2.copyMakeBorder(ROI, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        ROI = self.remove_isolated_pixels(ROI)
        return ROI

    def getTeacher(self):
        res = ''
        for a in range(2):
            i = (self.getTeacherImg(a)) / 255
            i = (i - 0.1307) / 0.3081
            # cv2_imshow(255*np.moveaxis(i,(0,1,2),(2,0,1)))
            r = torch.FloatTensor(i).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                test_output = self.network(r)
                pred = self.idxD[test_output.argmax()]
            res += pred[0]
        return res

    def getFIO(self):
        res = ''
        for a in range(34):
            i = (self.getFIOImg(a)) / 255
            i = np.stack([(i - 0.485) / 0.229, (i - 0.456) / 0.224, (i - 0.406) / 0.225])
            # cv2_imshow(255*np.moveaxis(i,(0,1,2),(2,0,1)))
            r = torch.FloatTensor(i).unsqueeze(0)
            with torch.no_grad():
                test_output = self.model_ft(r)
                pred = self.idx[test_output.argmax()]
            res += pred[0]
        return res

    def getAnswer(self, i):
        s = ''
        for j in range(4):
            im = (self.getAnswerImg(80 * (i // 20) + i % 20 + j * 20)) / 255
            im = np.stack([(im - 0.485) / 0.229, (im - 0.456) / 0.224, (im - 0.406) / 0.225])
            r = torch.FloatTensor(im).unsqueeze(0)
            with torch.no_grad():
                test_output = self.AnswModel(r)
                pred = self.idxA[test_output.argmax()]
            s = pred[0] + s
        return int(s, 2)

    def getAnsw(self):
        res = ''
        for a in range(240):
            i = (self.getAnswerImg(a)) / 255
            i = np.stack([(i - 0.485) / 0.229, (i - 0.456) / 0.224, (i - 0.406) / 0.225])
            # cv2_imshow(255*np.moveaxis(i,(0,1,2),(2,0,1)))
            r = torch.FloatTensor(i).unsqueeze(0)
            with torch.no_grad():
                test_output = self.AnswModel(r)
                pred = self.idxA[test_output.argmax()]
            res += pred[0]
        return res

    def getLevel(self):
        '''
        возвращает номер уровня
        '''
        res = 0
        for im in range(4):
            i = (self.getLevelImg(im)) / 255
            i = np.stack([(i - 0.485) / 0.229, (i - 0.456) / 0.224, (i - 0.406) / 0.225])
            r = torch.FloatTensor(i).unsqueeze(0)
            with torch.no_grad():
                test_output = self.AnswModel(r)
                pred = self.idxA[test_output.argmax()]
                if pred == "1":
                    res = 1 + im
        return res

    def doCheck(self):
        sum = 0
        err = ""
        fio = self.getFIO()
        level = self.getLevel()
        for i in range(60):
            answ = self.getAnswer(i)
            if answ == self.RealAnwers[level - 1][i]:
                sum += 1
            else:
                msk = answ | self.RealAnwers[level - 1][i]
                mskt = self.RealAnwers[level - 1][i]
                for j in range(4):
                    if msk & 1 == 1:
                        col = (0, 0, 255) if mskt & 1 == 0 else (0, 255, 0)
                        br = self.getAnswerBR(80 * (i // 20) + i % 20 + j * 20)
                        self.colorImg = cv2.rectangle(self.colorImg, (br[0], br[2]), (br[1], br[3]), col, 2)
                    msk = msk >> 1
                    mskt = mskt >> 1
                err += f" {i + 1}"
                # print(f"wrong answer {answ} in {i+1}")
        return (fio, level, sum, err)
