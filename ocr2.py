import cv2
import numpy as np
import pytesseract

# Load the image
image = cv2.imread('image.png')

# Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Apply OCR
text = pytesseract.image_to_string(thresh, config='--psm 6 outputbase digits')

# Print the result
print(text)