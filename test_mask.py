import re

def sort_images(lst):
    pattern = re.compile(r"\d+(?=\.)(?!.*\d)")
    try:
        sorted_lst  = sorted(lst, key=lambda x: int(re.search(pattern, x).group()))
        return sorted_lst
    except:
        return lst

# 测试代码
# image_list = ["中文一.jpg", "中文一二.jpg", "中文一零.jpg", "中文零零.jpg"]
image_list = ["011.jpg", "010.jpg", "101.jpg", "000.jpg"]
print(sort_images(image_list))  # 应该输出 ["中文零零.jpg", "中文一.jpg", "中文一二.jpg", "中文一零.jpg"]
