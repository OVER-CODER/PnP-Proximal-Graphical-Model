import cv2

# Load the image
image_path = "5.png"  # Change to your file path
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Crop the image to (0:767, 0:767)
cropped_image = image[0:767, 0:767]

# Save the cropped image as lossless PNG
cv2.imwrite("cropped_5.png", cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print("Cropped image saved as 'cropped_5.png'")
