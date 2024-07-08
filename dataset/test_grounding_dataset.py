import os
from torchvision import transforms
from grounding_dataset import grounding_dataset, grounding_dataset_bbox

ann_file = ["../test_set.json"]
image_root = "../test_image_set"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def test_grounding_dataset():
    dataset = grounding_dataset(ann_file, transform, image_root, max_words=30, mode='train')
    print("Testing grounding_dataset:")
    print(f"Dataset size: {len(dataset)}")

    image, caption, img_id = dataset[0]
    print(f"Image size: {image.size()}")
    print(f"Caption: {caption}")
    print(f"Image ID: {img_id}")

def test_grounding_dataset_bbox():
    config = {
        'image_res': 224,
        'careful_hflip': True,
        'refcoco_data': "path/to/your/refcoco_data"
    }
    dataset = grounding_dataset_bbox(ann_file, transform, image_root, max_words=30, mode='train', config=config)
    print("Testing grounding_dataset_bbox:")
    print(f"Dataset size: {len(dataset)}")

    image, caption, target_bbox = dataset[0]
    print(f"Image size: {image.size()}")
    print(f"Caption: {caption}")
    print(f"Target BBox: {target_bbox}")

if __name__ == "__main__":
    test_grounding_dataset()
    test_grounding_dataset_bbox()
