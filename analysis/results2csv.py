import os
import re
import csv

import argparse

def test2csv(save_as = "metrics.csv"):
    # Folder path
    root_path = '../temp/test_log/report'

    # List files in the folder
    file_names = os.listdir(root_path)

    # Define CSV file name
    target_path = "./out/csv/"
    csv_file_name = save_as
    csv_file_name = os.path.join(target_path, csv_file_name)
    fieldnames = [ "epoch", "model", "accuracy", "precision", "recall", "f1-score", "fold", "weight","filename"]

    # Write data to CSV
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each file in the folder
        for filename in file_names:
            file_path = os.path.join(root_path, filename)

            if ".txt" in file_path and ".jpg" not in file_path:
                # Split the filename by '_' and '-'
                parts = filename.split('_')
                num_ep = parts[2].split('-')[0]
                fold = parts[3].split('-')[0]
                model = parts[1]
                weight = parts[2].split('-')[1]

                print(f'Epoch: {num_ep}, model: {model}, fold: {fold}, which: {weight}')

                print(file_path)
                # Open the file in read mode
                with open(file_path, 'r') as file:
                    # Read the entire contents of the file into a string
                    file_contents = file.read()

                # Extracting accuracy, precision, recall, and f1-score
                accuracy = re.search(r'accuracy\s+([\d.]+)', file_contents).group(1)
                precision, recall, f1_score = re.findall(r'(\d+\.\d+)', file_contents)[-3:]

                # Write data to CSV 
                writer.writerow({
                    "epoch": num_ep,
                    "model": model,
                    "weight": weight,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1_score,
                    "fold": fold,
                    "filename": filename
                })

    print("Data has been written to", csv_file_name)


def valid_limit(value):
    try:
        start, end = map(int, value.split('-'))
        if start < 0 or end < start:
            raise argparse.ArgumentTypeError("Invalid limit format. Must be 'start-end' with start >= 0 and end >= start.")
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid limit format. Must be 'start-end' with start and end as integers.")

def valid_size(value):
    try:
        size = int(value)
        if size <= 0:
            raise argparse.ArgumentTypeError("Invalid size. Must be a positive integer.")
        return size
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid size. Must be a positive integer.")

def main():
    parser = argparse.ArgumentParser(description='Process images with specified parameters.')
    
    # parser.add_argument('--sex', type=str, default='A', choices=['M', 'F', 'A'], help='Sex of the subjects (M/F) or all (A)')
    parser.add_argument('--save_as', type=str, default='metrics.csv', help='The filename to save the csv file')
    # parser.add_argument('--size', type=str, default='All', help='Number of subjects to include in your dataset. All if all')
    args = parser.parse_args()

    test2csv(save_as=args.save_as)

if __name__ == "__main__":
    main()
