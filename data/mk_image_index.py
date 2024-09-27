import os
import pandas as pd

def isic2018():
    # CSV file path
    csv_path = "/home/gu721/yzc/Semi-supervised-learning/data/ISIC2018/HAM10000_metadata.csv"
    train_all_info = pd.read_csv(csv_path)

    # Extract relevant columns
    image_id = train_all_info.iloc[:, 1].values
    lesion_id = train_all_info.iloc[:, 0].str.replace('HAM_', '').values  # Remove 'HAM_' from lesion_id
    dx = train_all_info.iloc[:, 2].values
    age = train_all_info.iloc[:, 4].values
    sex = train_all_info.iloc[:, 5].values
    localization = train_all_info.iloc[:, 6].values

    # Define the categories for dx and localization
    dx_categories = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    localization_categories = ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand',
                               'lower extremity', 'neck', 'scalp', 'trunk', 'unknown', 'upper extremity']

    # Create a new DataFrame
    new_data = []

    for i in range(len(image_id)):
        row = [image_id[i]]

        # dx one-hot encoding (7 categories from 3rd to 9th columns)
        dx_row = [1 if dx[i] == category else 0 for category in dx_categories]
        row.extend(dx_row)

        # Append lesion_id after dx_categories
        row.append(lesion_id[i])

        # Age column (10th column)
        row.append(age[i])

        # Sex column (11th column) - male is 1, otherwise 0
        row.append(1 if sex[i] == 'male' else 0)

        # Localization encoding (15 categories, 12th column) - assign corresponding index if it matches
        loc_index = localization_categories.index(localization[i]) + 1 if localization[
                                                                              i] in localization_categories else 0
        row.append(loc_index)

        new_data.append(row)

    # Column names for the new DataFrame
    columns = ['image_id'] + dx_categories + ['lesion_id', 'age', 'sex', 'localization']

    # Create DataFrame and save it
    new_df = pd.DataFrame(new_data, columns=columns)

    # Define the new CSV file path
    new_csv_path = os.path.join(os.path.dirname(csv_path), 'train_dataset.csv')

    # Save to CSV
    new_df.to_csv(new_csv_path, index=False)

    print(f"New dataset saved at: {new_csv_path}")


def cxr8():
    # CSV file path
    csv_path = "/home/gu721/yzc/Semi-supervised-learning/data/cxr8/Data_Entry_2017_v2020.csv"
    all_info = pd.read_csv(csv_path)

    # Extract necessary columns
    image_id = all_info.iloc[:, 0].values
    finding_labels = all_info.iloc[:, 1].values
    patient_id = all_info.iloc[:, 3].values
    patient_age = all_info.iloc[:, 4].values
    patient_gender = all_info.iloc[:, 5].values
    view_position = all_info.iloc[:, 6].values

    # Define the 14 diseases
    diseases = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    # Initialize a list to store one-hot encoded disease labels
    disease_labels = []

    for labels in finding_labels:
        # Initialize a list of zeros for one-hot encoding
        one_hot = [0] * len(diseases)
        # Split the labels by '|'
        label_list = labels.split('|')
        # For each label, if it is in the disease list, set the corresponding position to 1
        for idx, disease in enumerate(diseases):
            if disease in label_list:
                one_hot[idx] = 1
        disease_labels.append(one_hot)

    # Convert the list of lists into a DataFrame
    disease_df = pd.DataFrame(disease_labels, columns=diseases)

    # Map patient_gender: M -> 1, F -> 0
    patient_gender_encoded = [1 if gender == 'M' else 0 for gender in patient_gender]

    # Map view_position: AP -> 1, PA -> 0
    view_position_encoded = [1 if position == 'AP' else 0 for position in view_position]

    # Create the new DataFrame
    new_df = pd.DataFrame()
    new_df['image_id'] = image_id
    for disease in diseases:
        new_df[disease] = disease_df[disease]
    new_df['patient_id'] = patient_id
    new_df['patient_age'] = patient_age
    new_df['patient_gender'] = patient_gender_encoded
    new_df['view_position'] = view_position_encoded

    # Save the new DataFrame to CSV in the same directory as the original CSV
    output_csv_path = os.path.join(os.path.dirname(csv_path), 'all_dataset.csv')
    new_df.to_csv(output_csv_path, index=False)
    print(f"New dataset saved to {output_csv_path}")

if __name__ == '__main__':
    cxr8()
