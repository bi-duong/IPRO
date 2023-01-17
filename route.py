
import cv2
def daoanh(img):
    return 255-img
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def setBrighness(img, beta_value):
    return cv2.convertScaleAbs(img, beta = beta_value)

def bilateralBlur(img, diameter, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)

def gaussianBlur(img, kSize, sigmaX):
    return cv2.GaussianBlur(img, kSize, sigmaX)

def laplasianFilter(img):
    return cv2.Laplacian(grayscale(img), cv2.CV_8UC4, ksize=5)

def sobel(img, x, y):
    return cv2.Sobel(gaussianBlur(grayscale(img), (3, 3), 0), cv2.CV_64F, x, y, ksize=5)

def canny(img):
    return cv2.Canny(img, 100, 200)

def averagingBlur(img, kSize):
    return cv2.blur(img, kSize)

def medianBlur(img, kSize):
    return cv2.medianBlur(img, kSize)