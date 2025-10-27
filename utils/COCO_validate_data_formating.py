from pycocotools.coco import COCO
import os


val_ann_file = r"C:\Users\serhi\OneDrive\CD_DSST\Article_syntetic_data\Code\ModelTrainingProject\COCODataSet_EfficientDet\annotations\instances_val.json"
coco = COCO(val_ann_file)

print("Categories:", coco.cats)
print("Number of images:", len(coco.getImgIds()))
print("Number of annotations:", len(coco.anns))

# Check for invalid image_id or category_id
img_ids = set(coco.getImgIds())
cat_ids = set(coco.getCatIds())
for ann_id, ann in coco.anns.items():
    if ann['image_id'] not in img_ids:
        print(f"Invalid image_id {ann['image_id']} in annotation {ann_id}")
    if ann['category_id'] not in cat_ids:
        print(f"Invalid category_id {ann['category_id']} in annotation {ann_id}")

# Check for images without annotations
for img_id in img_ids:
    if not coco.getAnnIds(imgIds=[img_id]):
        print(f"Image {img_id} has no annotations")