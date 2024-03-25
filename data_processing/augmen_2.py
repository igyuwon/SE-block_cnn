import os
import cv2
from imgaug import augmenters as iaa
import random

# 데이터 폴더 경로
image_folder_path = "/Workspace/akiec_orig"

# 증강된 데이터 저장 폴더 경로
output_folder_path = "/Workspace/akiec_aug"

resize = iaa.Resize({"height": 224, "width": 224})

# 회전 각도 목록
rotation_angles = [90, 180, 270]

# 이미지 증강을 위한 변환
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% 확률로 좌우 반전
    iaa.Affine(rotate=random.choice(rotation_angles)),  # 무작위 회전 각도 선택
    iaa.GaussianBlur(sigma=(0, 1.0)),  # 가우시안 블러 적용
])

# 이미지 데이터 증강 및 저장
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

image_files = os.listdir(image_folder_path)
num_images = len(image_files)
target_num_images = 500
augmented_count = 0

while augmented_count < target_num_images:
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        output_file_name = f"{augmented_count + 1}_akiec.jpg"
        output_path = os.path.join(output_folder_path, output_file_name)

        image = cv2.imread(image_path)
        image = resize.augment_image(image)
        augmented_image = augmenter.augment_image(image)

        cv2.imwrite(output_path, augmented_image)

        augmented_count += 1
        if augmented_count >= target_num_images:
            break

        if augmented_count % 10 == 0:
            print(f"증강된 이미지 개수: {augmented_count}/{target_num_images}")

print("데이터 증강이 완료되었습니다.")
