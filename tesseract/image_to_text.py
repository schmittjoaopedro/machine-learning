import cv2
from pytesseract import pytesseract

imagePath = "images/billboard.jpeg"

# pytesseract.pytesseract.tesseract_cmd = ‘/usr/bin/tesseract’# Parameters: ‘-l eng’ for using the English language LSTM OCR Engine

config = ("-l eng — oem 1 — psm 3")  # Read image from disk

image = cv2.imread(imagePath, cv2.IMREAD_COLOR)  # Run tesseract OCR on image

text = pytesseract.image_to_string(image, config=config)  # Write recognized text to file

print(text)
