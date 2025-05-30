import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os 
from Res_NET_Dataset import CustomObjectDetectionDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Res_NET_trainer import get_faster_rcnn_model
from mypkg.Image_generator.ImageGenerator import add_cifar_images_with_progress
 
# Model hyperparameters
NUM_CLASSES = 2  # Update this based on your dataset

# --- 5. Evaluation (Simplified) and Visualization ---
def visualize_predictions(model, dataset, num_images_to_show=3, score_threshold=0.5):
    model.eval() # Set model to evaluation mode
    plt.figure(figsize=(15, 5 * num_images_to_show))

    # Get a few random images from the dataset for visualization
    indices = np.random.choice(len(dataset), num_images_to_show, replace=False)

    for i, idx in enumerate(indices):
        image, target = dataset[idx]

        # Convert PIL image to tensor and normalize to [0, 1]
        if isinstance(image, Image.Image):
            img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        else:
            img_tensor = image

        original_img = Image.fromarray(img_tensor.mul(255).permute(1, 2, 0).byte().numpy())

        with torch.no_grad():
            prediction = model([img_tensor.to(DEVICE)])

        ax = plt.subplot(num_images_to_show, 1, i + 1)
        ax.imshow(image)
        ax.set_title(f"Image {idx} - Predictions (Score > {score_threshold})")
        ax.axis('off')

        # Draw ground truth boxes (optional, for comparison)
        for box, label in zip(target['boxes'], target['labels']):
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(box[0], box[1]-5, f"GT: {label.item()}", color='g', fontsize=8)


        # Draw predicted boxes
        for element in range(len(prediction[0]['boxes'])):
            score = prediction[0]['scores'][element].item()
            if score > score_threshold:
                box = prediction[0]['boxes'][element].cpu().numpy()
                label = prediction[0]['labels'][element].item()
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(box[0], box[1]-5, f"Class: {label} (Score: {score:.2f})", color='r', fontsize=8)

    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate the model and load weights
    model = get_faster_rcnn_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("faster_rcnn_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    # Generate 50 new images in a custom folder
    generated_folder = "Data\RES-net-evaluation"
    os.makedirs(generated_folder, exist_ok=True)
    add_cifar_images_with_progress(num_images=10, first_image_index=0, output_folder=generated_folder)

    # Create a dataset for the generated images
    evaluation_dataset = CustomObjectDetectionDataset(generated_folder)

    # Visualize predictions on the generated images
    visualize_predictions(model, evaluation_dataset, num_images_to_show=3, score_threshold=0.5)