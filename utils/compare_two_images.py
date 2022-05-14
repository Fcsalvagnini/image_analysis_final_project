import cv2

name = "b106-8_t.png"

img_ours = cv2.imread(f"output_ours_co_basins/{name}")
img_falcao = cv2.imread(f"output_falcao_co_basins/{name}")

equal_pixels = (img_ours == img_falcao).sum()

print("Our image shape => ", img_ours.shape)
print("Falcao image shape => ", img_falcao.shape)
print("Total number of pixels => ", img_falcao.shape[0] * img_falcao.shape[1] * img_falcao.shape[2])
print("Total number of equal pixels => ", equal_pixels)