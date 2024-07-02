from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from PIL import Image
import cv2
import io
import pytesseract
import time

# Configuration dictionary
config = {
    'url': '',  # Replace with the actual URL
}

# Initialize the webdriver
driver = webdriver.Chrome()  # Or any other driver

def load_url(url):
    """
    Loads a given URL using the provided webdriver instance.
    """
    driver.get(url)

def read_captcha_image():
    """
    Finds the captcha image element by its ID, takes a screenshot, and saves it.
    Returns the path to the saved image file.
    """
    captcha_img_element = driver.find_element(By.ID, 'captcha_image')  # Adjust the ID as necessary
    image_binary = captcha_img_element.screenshot_as_png
    img = Image.open(io.BytesIO(image_binary))
    image_path = "captcha.png"
    img.save(image_path)
    return image_path

def process_image_for_ocr(image_path):
    """
    Reads the image from the given path, converts it to grayscale, applies thresholding,
    and returns the processed image ready for OCR.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def save_page_content(filename="page_content.html"):
    """
    Saves the current page content to a specified file.
    """
    with open(filename, "w", encoding="utf-8") as file:
        file.write(driver.page_source)

max_retries = 5  # Maximum number of retries
attempts = 0  # Current attempt count

try:
    while attempts < max_retries:
        load_url(config['url'])
        time.sleep(5) 
        image_path = read_captcha_image()

        processed_image = process_image_for_ocr(image_path)
        text = pytesseract.image_to_string(processed_image, config='--psm 6 outputbase digits')
        time.sleep(5) 
        # Fill the form with the OCR result
        securitycode_input = driver.find_element(By.NAME, 'securitycode')
        securitycode_input.send_keys(text)

        # Submit the form
        submit_button = driver.find_element(By.XPATH, '//input[@type="submit" and @value="Submit"]')
        submit_button.click()

        # Wait for the page to load after submission
        time.sleep(10)  # Adjust based on the expected processing time

        # Check for the presence of <meta name="author"> tag to confirm successful submission
        meta_tags = driver.find_elements(By.XPATH, '//meta[@name="author"]')
        if len(meta_tags) > 0:
            print("Submission successful.")
            break
        else:
            print("Submission failed or the page did not load correctly. Retrying...")
            attempts += 1

except NoSuchElementException:
    print("An error occurred, trying again...")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    save_page_content()  # Save the page content before quitting
    driver.quit()