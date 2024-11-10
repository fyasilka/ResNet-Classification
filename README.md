# ResNet-Classification
Image Similarity Search using ResNet18
This project demonstrates how to use a pre-trained ResNet18 model from PyTorch to extract features from images, perform clustering, and search for similar images based on their visual content.

Requirements
To run this code, you will need the following libraries installed:

Python 3.x
PyTorch
TorchVision
NumPy
Pandas
Scikit-learn
Matplotlib
You can install these dependencies by running:


- pip install torch torchvision numpy pandas scikit-learn matplotlib

# Usage
Place your images in a folder named 'Folder path'.
Run the main script to process the images, extract features, perform clustering, and find similar images.

# Example
Here’s an example of how to use the script:


if __name__ == "__main__":
   
    # Specify the path to your image folder
    image_folder = 'Folder path' 
    # Resize images to match ResNet18 input size
    resized_size = (224, 224) 
    images, image_paths = load_and_transform_local_images(image_folder, resize_to=resized_size)
    features = extract_features_with_resnet18(images)
    feature_df = create_feature_dataset(features, image_paths)
    clustered_df = perform_clustering(feature_df)

    # Find similar images for the first image in the dataset
    similar_filenames = find_similar_images(clustered_df, 0)
    visualize_results(clustered_df, similar_filenames)

# Contributing
If you would like to contribute to this project, feel free to submit a pull request or open an issue.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
