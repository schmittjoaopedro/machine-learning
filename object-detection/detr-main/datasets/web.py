import json
import os
import matplotlib.pyplot as plt
import cv2

from PIL import Image

cv2_ctx = {}


def convert_image_to_jpg(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((800, 420))
    img.save(img_path.replace('.png', '.jpg'))
    os.remove(img_path)


"""
Images annotation structure:
{
    "info": {
        "description": "COCO 2017 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2017,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    },
    "licenses": [{
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }],
    "images": [{
        "id": 397133,
        "license": 1,
        "height": 427,
        "width": 640,
        "file_name": "000000397133.jpg",
        "date_captured": "2013-11-14 17:02:52",
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg"
    }],
    "annotations": [{
        "segmentation": [
            [239.97, 260.24, 222.04, 270.49, 200.0, 246.51, 203.22, 244.98, 226.23, 255.22, 239.97, 260.24]
        ],
        "area": 2765.1486499999996,
        "iscrowd": 0,
        "image_id": 397133,
        "bbox": [200.0, 244.98, 39.97, 25.26],
        "category_id": 18,
        "id": 1768
    }],
    "categories": [{
        "id": 18,
        "supercategory": "animal",
        "name": "dog"
    }]
}
"""


def load_annotations(file_path):
    # Create file if it does not exist
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump({
                "info": {
                    "description": "Schmitt Web Dataset",
                    "url": "https://schmittjoaopedro.github.io/",
                    "version": "1.0",
                    "year": 2024,
                    "contributor": "Joao Pedro Schmitt",
                    "date_created": "2024/06/26"
                },
                "licenses": [{
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }],
                "images": [],
                "annotations": [],
                "categories": [{
                    "id": 1,
                    "supercategory": "component",
                    "name": "button"
                }, {
                    "id": 2,
                    "supercategory": "component",
                    "name": "filter"
                }, {
                    "id": 3,
                    "supercategory": "component",
                    "name": "checkbox"
                }, {
                    "id": 4,
                    "supercategory": "component",
                    "name": "data_picker"
                }, {
                    "id": 5,
                    "supercategory": "component",
                    "name": "dialog"
                }, {
                    "id": 6,
                    "supercategory": "component",
                    "name": "expansion"
                }, {
                    "id": 7,
                    "supercategory": "component",
                    "name": "input"
                }, {
                    "id": 8,
                    "supercategory": "component",
                    "name": "navigation_icon"
                }, {
                    "id": 9,
                    "supercategory": "component",
                    "name": "list"
                }, {
                    "id": 10,
                    "supercategory": "component",
                    "name": "menu"
                }, {
                    "id": 11,
                    "supercategory": "component",
                    "name": "paginator"
                }, {
                    "id": 12,
                    "supercategory": "component",
                    "name": "progress"
                }, {
                    "id": 13,
                    "supercategory": "component",
                    "name": "radio"
                }, {
                    "id": 14,
                    "supercategory": "component",
                    "name": "table"
                }, {
                    "id": 15,
                    "supercategory": "component",
                    "name": "video"
                }, {
                    "id": 16,
                    "supercategory": "component",
                    "name": "image"
                }, {
                    "id": 17,
                    "supercategory": "component",
                    "name": "text_content"
                }, {
                    "id": 18,
                    "supercategory": "component",
                    "name": "text_title"
                }, {
                    "id": 19,
                    "supercategory": "component",
                    "name": "tab"
                }, {
                    "id": 20,
                    "supercategory": "component",
                    "name": "scroll"
                }, {
                    "id": 21,
                    "supercategory": "component",
                    "name": "link"
                }]
            }, file)

    with open(file_path, 'r') as file:
        return json.load(file)


def plot_image(image_path, annotations_path, image_id):
    with open(annotations_path, 'r') as file:
        json_obj = json.load(file)
        print("Loaded")

        image = [img for img in json_obj["images"] if img["id"] == int(image_id)][0]
        annotations = [ann for ann in json_obj["annotations"] if ann["image_id"] == image["id"]]
        categories = {cat["id"]: cat["name"] for cat in json_obj["categories"]}

        # Plot image
        img = Image.open(f'{image_path}/{image["file_name"]}')
        plt.imshow(img)

        # Plot bounding boxes for image
        for ann in annotations:
            bbox = ann["bbox"]
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=2))
            plt.text(bbox[0], bbox[1], categories[ann["category_id"]], fontsize=12, color='red')

        plt.show()


def input_labels_to_image(annotations_path, image_path):
    cv2_ctx['ix'] = 0
    cv2_ctx['iy'] = 0
    cv2_ctx['drawing'] = False
    cv2_ctx['original_img'] = cv2.imread(image_path)
    cv2_ctx['img'] = cv2_ctx['original_img'].copy()
    annotations_object = load_annotations(annotations_path)

    image_name = image_path.split('/')[-1].split('.')[0]
    image_id = int(image_name)

    def draw_rectangle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2_ctx['drawing'] = True
            cv2_ctx['ix'], cv2_ctx['iy'] = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if cv2_ctx['drawing'] is True:
                cv2_ctx['img'] = cv2_ctx['original_img'].copy()
                cv2.rectangle(cv2_ctx['img'], (cv2_ctx['ix'], cv2_ctx['iy']), (x, y), (0, 255, 0), 0)

        elif event == cv2.EVENT_LBUTTONUP:
            cv2_ctx['drawing'] = False
            cv2_ctx['img'] = cv2_ctx['original_img'].copy()

            options = '\n'.join([f"{cat['id']}: {cat['name']}" for cat in annotations_object['categories']])
            label = input(f"{options}\nEnter label (-1 cancel, 0 finish): ")

            if label == '0':
                annotations_object['images'].append({
                    "id": image_id,
                    "license": 1,
                    "height": cv2_ctx['original_img'].shape[0],
                    "width": cv2_ctx['original_img'].shape[1],
                    "file_name": f"{image_name}.jpg",
                    "date_captured": "2024-06-26 17:02:52",
                    "coco_url": f"http://localhost:8000/{image_name}.jpg",
                    "flickr_url": f"http://localhost:8000/{image_name}.jpg"
                })
                # Save annotations
                with open(annotations_path, 'w') as file:
                    json.dump(annotations_object, file)
            elif label != '-1':
                cv2.rectangle(cv2_ctx['img'], (cv2_ctx['ix'], cv2_ctx['iy']), (x, y), (0, 255, 0), 0)
                bounding_box = (cv2_ctx['ix'], cv2_ctx['iy'], x - cv2_ctx['ix'], y - cv2_ctx['iy'])
                annotations_object['annotations'].append({
                    "segmentation": [],
                    "area": bounding_box[2] * bounding_box[3],
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": bounding_box,
                    "category_id": int(label),
                    "id": len(annotations_object['annotations']) + 1
                })
                print(f"Label: {label}")
                print(f"Bounding Box (x, y, width, height): {bounding_box}")

            cv2_ctx['original_img'] = cv2_ctx['img'].copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)
    while 1:
        cv2.imshow("image", cv2_ctx['img'])
        # Exist on ESC key
        if cv2.waitKey(20) & 0xFF == 27:
            break


#split = 'train'
split = 'val'
image_id = input("Enter image id: ")
option = input("What you want to do? (1: Convert image to jpg, 2: Input labels to image, 3: Plot image): ")
if option == '1':
    convert_image_to_jpg(f'data_web/{split}2017/{image_id}.png')
elif option == '2':
    input_labels_to_image(f'data_web/annotations/instances_{split}2017.json', f'data_web/{split}2017/{image_id}.jpg')
elif option == '3':
    plot_image(f'data_web/{split}2017', f'data_web/annotations/instances_{split}2017.json', image_id)
