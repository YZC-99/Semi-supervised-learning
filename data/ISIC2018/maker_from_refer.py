import pandas as pd
import os


def split_dataset():
    # Paths to the CSV files
    all_info = "/home/gu721/yzc/Semi-supervised-learning/data/ISIC2018/all_dataset.csv"

    train_refer = "/home/gu721/yzc/Semi-supervised-learning/data/ISIC2018/refer/training.csv"
    val_refer = "/home/gu721/yzc/Semi-supervised-learning/data/ISIC2018/refer/validation.csv"
    test_refer = "/home/gu721/yzc/Semi-supervised-learning/data/ISIC2018/refer/testing.csv"

    # Read the all_info CSV file
    all_data = pd.read_csv(all_info)

    # Read the reference CSV files and extract the 'image' column
    train_images = pd.read_csv(train_refer)['image'].tolist()
    val_images = pd.read_csv(val_refer)['image'].tolist()
    test_images = pd.read_csv(test_refer)['image'].tolist()

    # Map 'image_id' in all_data to match 'image' in reference files
    # Assuming they are directly comparable; if not, adjust accordingly

    # Filter all_data based on the images in each reference list
    train_data = all_data[all_data['image_id'].isin(train_images)]
    val_data = all_data[all_data['image_id'].isin(val_images)]
    test_data = all_data[all_data['image_id'].isin(test_images)]

    # Get the directory of the all_info file to save the new CSV files
    save_dir = os.path.dirname(all_info)

    # Save the datasets to CSV files in the same directory as all_info
    train_data.to_csv(os.path.join(save_dir, 'train_dataset.csv'), index=False)
    val_data.to_csv(os.path.join(save_dir, 'val_dataset.csv'), index=False)
    test_data.to_csv(os.path.join(save_dir, 'test_dataset.csv'), index=False)

    print("Datasets have been split and saved successfully.")
    print(f"Train dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(val_data)}")
    print(f"Test dataset size: {len(test_data)}")


if __name__ == "__main__":
    split_dataset()
