import os
import shutil
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import nltk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from IPython.display import display
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

class Preprocess:
    def __init__(
        self,
        images_dir,
        annotations_dir,
        class_id,
        plot_images=True,
        convert_to_yolo=True,
        move_images=True,
        display_df=True,
        create_annotations_dataframe = True
    ) -> None:
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.class_id = class_id
        self.methods = []
        if plot_images:
            self.methods.append(self.plot_multiple_images)
        if convert_to_yolo:
            self.methods.append(self.convert_and_save_yolo)
        if move_images:
            self.methods.append(self.move_images_to_dirs)
        if display_df:
            self.methods.append(self.display_dataframe)
        if create_annotations_dataframe:
            self.methods.append(self.create_annotations_dataframe)


    def __call__(self):
        for method in self.methods:
            method()

    def plot_image_with_bbox(self, ax, image_path, annotation_path):
        image = Image.open(image_path)
        ax.imshow(image)
        ax.axis('off')

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, label, color='yellow', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    def plot_multiple_images(self, num_images=16):
        images = sorted(os.listdir(self.images_dir))[:num_images]
        annotations = sorted(os.listdir(self.annotations_dir))[:num_images]
        
        num_cols = 4
        num_rows = (num_images + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        
        for ax, img, ann in zip(axes, images, annotations):
            image_path = os.path.join(self.images_dir, img)
            annotation_path = os.path.join(self.annotations_dir, ann)
            self.plot_image_with_bbox(ax, image_path, annotation_path)
        
        for ax in axes[len(images):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def create_annotations_dataframe(self):
        data_dict = {
            'filename': [],
            'label': [],
            'class_id': [],
            'width': [],
            'height': [],
            'bboxes': []
        }

        for annotation_file in os.listdir(self.annotations_dir):
            annotation_path = os.path.join(self.annotations_dir, annotation_file)
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            filename = root.find('filename').text
            for obj in root.findall("object"):
                label = obj.find("name").text
                bbox = [
                    int(obj.find('bndbox/xmin').text),
                    int(obj.find('bndbox/ymin').text),
                    int(obj.find('bndbox/xmax').text),
                    int(obj.find('bndbox/ymax').text)
                ]
                size = root.find('size')
                
                data_dict['filename'].append(filename)
                data_dict['width'].append(int(size.find('width').text))
                data_dict['height'].append(int(size.find('height').text))
                data_dict['label'].append(label)
                data_dict['class_id'].append(self.class_id[label])
                data_dict['bboxes'].append(bbox)

        return pd.DataFrame(data_dict)

    def pascal_voc_to_yolo_bbox(self, bbox_array, w, h):
        x_min, y_min, x_max, y_max = bbox_array
        x_center = ((x_max + x_min) / 2) / w
        y_center = ((y_max + y_min) / 2) / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h
        return [x_center, y_center, width, height]

    def convert_to_yolo_format(self, df, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for index, row in df.iterrows():
            img_path = os.path.join(self.images_dir, row['filename'])
            img = Image.open(img_path)
            img_width, img_height = img.size

            bbox = row['bboxes']
            yolo_bbox = self.pascal_voc_to_yolo_bbox(bbox, img_width, img_height)
            label_path = os.path.join(save_dir, row['filename'].replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                f.write(f"{row['class_id']} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")

    def move_images(self, df, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for filename in df['filename']:
            img_path = os.path.join(self.images_dir, filename)
            save_path = os.path.join(save_dir, filename)
            shutil.copy(img_path, save_path)

    def convert_and_save_yolo(self):
        df_data = self.create_annotations_dataframe()
        train, test = train_test_split(df_data, test_size= .1,
                              stratify= df_data['label'],
                              random_state= 42)
        train, val = train_test_split(df_data, test_size= .2,
                              stratify= df_data['label'],
                              random_state= 42)

        self.convert_to_yolo_format(train, 'train/labels')
        self.convert_to_yolo_format(val, 'val/labels')
        self.convert_to_yolo_format(test, 'test/labels')

    def move_images_to_dirs(self):
        df_data = self.create_annotations_dataframe()
        train, test = train_test_split(df_data, test_size= .1,
                              stratify= df_data['label'],
                              random_state= 42)
        train, val = train_test_split(df_data, test_size= .2,
                              stratify= df_data['label'],
                              random_state= 42)

        os.makedirs('train/images', exist_ok=True)
        os.makedirs('val/images', exist_ok=True)
        os.makedirs('test/images', exist_ok=True)

        self.move_images(train, 'train/images')
        self.move_images(val, 'val/images')
        self.move_images(test, 'test/images')

    def display_dataframe(self):
        df_data = self.create_annotations_dataframe()
        self.show_df(df_data)

    def show_df(self, df_train):
        print('shape'.center(30,'_'))
        display(df_train)

        print('head'.center(30,'_'))
        display(df_train.head().style.background_gradient(cmap='Blues').applymap(self.background_color))

        print('tail'.center(30,'_'))
        display(df_train.tail().style.background_gradient(cmap='Blues').applymap(self.background_color))

        print('info'.center(30,'_')+'\n')
        display(df_train.info())

        print('describe_continuous'.center(30,'_'))
        display(df_train.describe().T.style.background_gradient(cmap = 'Blues'))

        print('describe_categorical'.center(30,'_'))
        display(df_train.describe(include='object').T.applymap(self.background_color))

        print('null_values_percent'.center(30,'_'))
        display((df_train.isna().sum() / len(df_train) * 100).sort_values(ascending=False))

    def background_color(self, value):
        if isinstance(value, str):
            return 'background-color: #a6c0ed'
        return ''

# Example usage
images_dir = "/content/drive/MyDrive/datasets/images"
annotations_dir = "/content/drive/MyDrive/datasets/annotations"
class_id = {
    "with_mask": 0,
    "mask_weared_incorrect": 1,
    "without_mask": 2
}

#preprocessor = Preprocess(images_dir, annotations_dir, class_id)
#preprocessor()
