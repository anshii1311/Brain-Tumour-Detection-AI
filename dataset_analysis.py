import os

base_dir = "dataset"   # because dataset folder is inside project folder
classes = os.listdir(base_dir)

for cls in classes:
    cls_path = os.path.join(base_dir, cls)
    print(cls, ":", len(os.listdir(cls_path)))
