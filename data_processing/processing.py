import os
from PIL import Image


# #500개 이상 데이터 전처리(이하도 같이 포함 되어있음)
# txt_file_path = "/Workspace/df.txt"
# image_folder_path = "/Workspace/process_images"
# output_folder_path = "/Workspace/df"

# with open(txt_file_path, "r") as txt_file:
#     for line in txt_file:
#         image_name = line.strip()
#         image_path = os.path.join(image_folder_path, f"{image_name}.jpg")
        
#         if os.path.exists(image_path):
#             image = Image.open(image_path)
#             output_file_path = os.path.join(output_folder_path,f"{input_count + 1}_df.jpg")
#             #output_file_path = os.path.join(output_folder_path,f"{image_name}_output.jpg")
#             image.save(output_file_path)
#             print(f"불러온 이미지 : {image_path}")
#         else:
#             print("파일 없음")

#위 500개 중 500개 이하 암만 추출하여 다른 폴더에 저장  
txt_file_path = "/Workspace/df.txt"
image_folder_path = "/Workspace/process_images"
output_folder_path = "/Workspace/df"

target_num_images = 115
input_count = 0
count = 0
while input_count < target_num_images:

    with open(txt_file_path, "r") as txt_file:
        for line in txt_file:
            image_name = line.strip()
            image_path = os.path.join(image_folder_path, f"{image_name}_output.jpg")
            count += 1
        
            if os.path.exists(image_path):
                image = Image.open(image_path)
                output_file_path = os.path.join(output_folder_path,f"{input_count + 1}_df.jpg")
                image.save(output_file_path)
                print(f"불러온 이미지 : {image_path}")
            else:
                print("파일 없음")
                
            input_count += 1
            if input_count >= target_num_images:
                break
print(count)
print("데이터 분류가 완료되었습니다.")

