#!/usr/bin/env python3
"""
Dataset Balancer for COSMICA

Creates balanced train/validation/test splits from existing dataset.
Maintains proportional class distribution across splits using a greedy algorithm.

Usage: python balancer.py
"""

import os
import shutil
import random
from collections import defaultdict


def main():
    # 1. Collect user input parameters
    dataset_path = input("Enter path to dataset: ").strip()

    # Specify the percentage split for train / test / val, e.g. 70 20 10
    ratio_train = float(input("Percentage for TRAIN: ").strip())
    ratio_test = float(input("Percentage for TEST: ").strip())
    ratio_val = float(input("Percentage for VAL: ").strip())

    # Verify that the sum equals 100
    ratio_sum = ratio_train + ratio_test + ratio_val
    if abs(ratio_sum - 100.0) > 1e-3:
        print("Sum of percentages must equal 100!")
        return

    # For convenience, convert percentages to fractions (0..1)
    ratio_train /= 100.0
    ratio_test /= 100.0
    ratio_val /= 100.0

    # 2. Collect all annotations from existing dataset splits
    #    Folder structure: dataset/{train,test,valid}/ with subfolders images/ and labels/
    splits = ["train", "test", "valid"]  # folder names in the original dataset
    annotations = []

    for split in splits:
        image_dir = os.path.join(dataset_path, split, "images")
        label_dir = os.path.join(dataset_path, split, "labels")
        print(f"Scanning: {image_dir}, {label_dir}")
        
        # Skip if directories do not exist
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Skipping non-existent directories for split {split}")
            continue

        # Scan all txt files in label_dir
        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue

            # Full path to the annotation file
            label_path = os.path.join(label_dir, label_file)
            # Assume image name matches label name with .jpg extension (adjust if needed)
            base_name = os.path.splitext(label_file)[0]
            
            # Try different image extensions
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                image_file = base_name + ext
                potential_path = os.path.join(image_dir, image_file)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break

            # If no image found, skip this record
            if image_path is None:
                print(f"Image not found for {label_file}, skipping...")
                continue

            # Parse classes from txt (YOLO format: class_id x_center y_center width height)
            # If the file is empty we treat it as background only
            classes = set()
            with open(label_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 1:
                        continue
                    class_id = parts[0]
                    classes.add(class_id)
            
            # If annotation is empty, treat as background (no objects)
            if not classes:
                classes.add("background")

            # Add annotation to our list
            annotations.append({
                "image_path": image_path,
                "label_path": label_path,
                "classes": classes,
                "original_split": split
            })

    # 3. Shuffle to avoid ordering bias and get stats
    random.shuffle(annotations)

    total_ann = len(annotations)
    if total_ann == 0:
        print("No annotations found. Exiting.")
        return

    print(f"Found {total_ann} total annotations")

    # 4. Gather all classes (including background)
    all_classes = set()
    for ann in annotations:
        all_classes.update(ann["classes"])

    print(f"Found classes: {sorted(all_classes)}")

    # 5. Determine quotas for each split
    need_train = int(total_ann * ratio_train)
    need_test = int(total_ann * ratio_test)
    need_val = total_ann - need_train - need_test  # for compensation of rounding

    train_list = []
    test_list = []
    val_list = []

    # Helpers to count class occurrences per split
    class_counts_train = defaultdict(int)
    class_counts_test = defaultdict(int)
    class_counts_val = defaultdict(int)

    for ann in annotations:
        # When all quotas are filled, add to the smallest split to compensate
        if len(train_list) >= need_train and len(test_list) >= need_test and len(val_list) >= need_val:
            sizes = {"train": len(train_list), "test": len(test_list), "val": len(val_list)}
            min_split = min(sizes, key=sizes.get)
            if min_split == "train":
                train_list.append(ann)
                for c in ann["classes"]:
                    class_counts_train[c] += 1
            elif min_split == "test":
                test_list.append(ann)
                for c in ann["classes"]:
                    class_counts_test[c] += 1
            else:
                val_list.append(ann)
                for c in ann["classes"]:
                    class_counts_val[c] += 1
            continue

        # Evaluate class balance score for each split
        score_train = score_test = score_val = 0
        if len(train_list) < need_train:
            for c in ann["classes"]:
                score_train += class_counts_train[c]
        else:
            score_train = float("inf")
        if len(test_list) < need_test:
            for c in ann["classes"]:
                score_test += class_counts_test[c]
        else:
            score_test = float("inf")
        if len(val_list) < need_val:
            for c in ann["classes"]:
                score_val += class_counts_val[c]
        else:
            score_val = float("inf")

        # Choose split with the lowest balance score
        min_score = min(score_train, score_test, score_val)
        if min_score == float("inf"):
            sizes = {"train": len(train_list), "test": len(test_list), "val": len(val_list)}
            min_split = min(sizes, key=sizes.get)
            if min_split == "train":
                train_list.append(ann)
                for c in ann["classes"]:
                    class_counts_train[c] += 1
            elif min_split == "test":
                test_list.append(ann)
                for c in ann["classes"]:
                    class_counts_test[c] += 1
            else:
                val_list.append(ann)
                for c in ann["classes"]:
                    class_counts_val[c] += 1
        else:
            if min_score == score_train:
                train_list.append(ann)
                for c in ann["classes"]:
                    class_counts_train[c] += 1
            elif min_score == score_test:
                test_list.append(ann)
                for c in ann["classes"]:
                    class_counts_test[c] += 1
            else:
                val_list.append(ann)
                for c in ann["classes"]:
                    class_counts_val[c] += 1

    # Output final distribution
    print(f"\nFinal distribution:")
    print(f"TRAIN: {len(train_list)} ({len(train_list)/total_ann*100:.1f}%)")
    print(f"TEST:  {len(test_list)} ({len(test_list)/total_ann*100:.1f}%)")
    print(f"VAL:   {len(val_list)} ({len(val_list)/total_ann*100:.1f}%)")
    print(f"TOTAL: {total_ann}")

    # Show class distribution per split
    print(f"\nClass distribution:")
    for class_name in sorted(all_classes):
        train_count = class_counts_train[class_name]
        test_count = class_counts_test[class_name]
        val_count = class_counts_val[class_name]
        total_class = train_count + test_count + val_count
        print(f"  {class_name}: TRAIN={train_count}, TEST={test_count}, VAL={val_count}, TOTAL={total_class}")

    # 6. Create a new folder for the balanced dataset
    new_dataset_path = dataset_path + "_balanced"
    os.makedirs(new_dataset_path, exist_ok=True)
    for split in ["train", "test", "valid"]:
        os.makedirs(os.path.join(new_dataset_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(new_dataset_path, split, "labels"), exist_ok=True)

    def copy_file(src, dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        base_name = os.path.basename(src)
        dst = os.path.join(dst_dir, base_name)
        shutil.copy2(src, dst)

    def copy_split(ann_list, split_name):
        for ann in ann_list:
            image_src = ann["image_path"]
            label_src = ann["label_path"]
            image_dst_dir = os.path.join(new_dataset_path, split_name, "images")
            label_dst_dir = os.path.join(new_dataset_path, split_name, "labels")
            copy_file(image_src, image_dst_dir)
            copy_file(label_src, label_dst_dir)

    copy_split(train_list, "train")
    copy_split(test_list, "test")
    copy_split(val_list, "valid")

    print(f"Balanced version of the dataset saved to: {new_dataset_path}")


if __name__ == "__main__":
    main()
