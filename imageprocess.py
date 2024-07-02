import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return opening

def load_digit_templates():
    templates = {}
    for i in range(10):
        template = cv2.imread(f'digit_templates/{i}.png', 0)
        _, template = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY_INV)
        templates[i] = template
    return templates

def match_digit(roi, templates):
    best_match = -1
    best_score = float('-inf')
    for digit, template in templates.items():
        resized_template = cv2.resize(template, (roi.shape[1], roi.shape[0]))
        result = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
        score = np.max(result)
        if score > best_score:
            best_score = score
            best_match = digit
    return best_match

def extract_numbers(image_path, templates):
    image = cv2.imread(image_path)
    processed = preprocess_image(image)
    
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    result = ""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 100:  # Filter out very small contours
            roi = processed[y:y+h, x:x+w]
            digit = match_digit(roi, templates)
            result += str(digit)
            
            # Draw bounding box and digit on the image for visualization
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, str(digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Detected Digits', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result

# Load digit templates
digit_templates = load_digit_templates()

# Use the function
image_path = 'image.png'
result = extract_numbers(image_path, digit_templates)
print(f"Extracted numbers: {result}")