import os
from collections import defaultdict

def main():
    # 1. Ask user for dataset path
    dataset_path = input("Enter path to dataset: ").strip()

    # Define split names
    splits = ["train", "test", "valid"]

    # 2. Prepare structures for statistics
    # class_counts[split][class_id] = number of objects of class `class_id` in split `split`
    class_counts = {split: defaultdict(int) for split in splits}
    # total_per_split[split] = total number of objects (all classes) in this split
    total_per_split = {split: 0 for split in splits}
    # all_classes - set of all encountered classes
    all_classes = set()

    # 3. Read all label files
    for split in splits:
        label_dir = os.path.join(dataset_path, split, "labels")
        if not os.path.exists(label_dir):
            print(f"Directory {label_dir} not found, skipping {split}")
            continue

        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        for label_file in label_files:
            file_path = os.path.join(label_dir, label_file)
            if not os.path.isfile(file_path):
                continue

            has_object = False  # Flag to mark if objects were found in file
            with open(file_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # If line has at least one token, take the first as class_id
                    if len(parts) < 1:
                        continue
                    class_id = parts[0]
                    all_classes.add(class_id)
                    class_counts[split][class_id] += 1
                    total_per_split[split] += 1
                    has_object = True
            # If no objects found in file, treat as background
            if not has_object:
                class_counts[split]["background"] += 1
                total_per_split[split] += 1
                all_classes.add("background")

    # Exit if no data found
    if sum(total_per_split.values()) == 0:
        print("No annotations found. Check finished.")
        return

    # 4. Build summary distribution table
    # For each class count objects in each split and convert to percentages
    print("\nCLASS DISTRIBUTION BY SPLIT (% of all objects of each class):")
    print("-----------------------------------------------------")

    # First, find out how many objects of each class (in all splits)
    class_totals = defaultdict(int)
    for split in splits:
        for c, cnt in class_counts[split].items():
            class_totals[c] += cnt

    # Calculate overall percentage composition of splits across all annotations
    overall_total = sum(total_per_split.values())
    split_ratios = {}
    for split in splits:
        if overall_total > 0:
            split_ratios[split] = 100.0 * total_per_split[split] / overall_total
        else:
            split_ratios[split] = 0.0

    print(f"Total annotations: {overall_total}")
    print("Overall split ratios (for all classes combined):")
    for split in splits:
        print(f"  {split} = {split_ratios[split]:.2f}%")
    print("-----------------------------------------------------")

    # Print distribution for each class
    for c in sorted(all_classes):
        total_c = class_totals[c]
        if total_c == 0:
            continue

        dist_train = 100.0 * class_counts["train"][c] / total_c if total_c else 0
        dist_test = 100.0 * class_counts["test"][c] / total_c if total_c else 0
        dist_val = 100.0 * class_counts["valid"][c] / total_c if total_c else 0

        print(
            f"Class {c}: Train={dist_train:.2f}% | Test={dist_test:.2f}% | Valid={dist_val:.2f}% (out of {total_c} objects)"
        )

    # 5. Balance assessment
    print("\nBALANCE ASSESSMENT:")
    deviation_sum = 0.0
    count_nonempty_classes = 0
    for c in sorted(all_classes):
        total_c = class_totals[c]
        if total_c == 0:
            continue

        pct_train_c = 100.0 * class_counts["train"][c] / total_c
        pct_test_c = 100.0 * class_counts["test"][c] / total_c
        pct_val_c = 100.0 * class_counts["valid"][c] / total_c

        dev_train = abs(pct_train_c - split_ratios["train"])
        dev_test = abs(pct_test_c - split_ratios["test"])
        dev_val = abs(pct_val_c - split_ratios["valid"])

        avg_dev = (dev_train + dev_test + dev_val) / 3.0
        deviation_sum += avg_dev
        count_nonempty_classes += 1

    if count_nonempty_classes > 0:
        average_deviation = deviation_sum / count_nonempty_classes
    else:
        average_deviation = 0.0

    print(f"Average deviation by class from overall ratios: {average_deviation:.2f} p.p.")
    if average_deviation < 5:
        print("Dataset is well balanced (low deviation).")
    elif average_deviation < 15:
        print("Dataset is acceptably balanced (moderate deviation).")
    else:
        print("Dataset is poorly balanced (high deviation).")

if __name__ == "__main__":
    main()
