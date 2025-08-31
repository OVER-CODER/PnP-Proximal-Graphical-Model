import cv2

# Load the image
image_path = "demosaic_dncnn_5.jpg"  # Change this to your file path
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Get image dimensions
height, width = image.shape[:2]

# Define cropping area (690x690 box from the center of the rightmost third)
box_size = 1436
left_start = - (int(2 * width / 3) + (width // 6) - (685 // 2) - 7418) # Centered in rightmost third
top_start = (height // 2) - (689 // 2) - 365  # Centered vertically

# Ensure cropping boundaries are within the image limits
left_start = max(0, left_start)
top_start = max(0, top_start)
right_end = min(left_start + box_size, width)
bottom_end = min(top_start + box_size, height)

# Crop the image
cropped_image = image[top_start:bottom_end, left_start:right_end]

# Save the cropped image with high quality
cv2.imwrite("cropped_u.png", cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print("Cropped image saved as 'cropped.png'")
