from pathlib import Path
import argparse

import cv2
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path("/mnt/d/Datasets/FashionAI/round1_fashionAI_attributes_test_a")
DEFAULT_CSV = DEFAULT_ROOT / "Tests" / "round1_fashionAI_attributes_answer_a.csv"


def label_to_class_index(label: str) -> int:
    """
    Convert FashionAI label string like 'nynnn' into class index.
    The position of 'y' is the selected class.
    """
    label = str(label).strip()

    if "y" not in label:
        return -1

    return label.index("y")


def load_annotations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["image_path", "attribute_name", "label"],
    )

    df["class_index"] = df["label"].apply(label_to_class_index)
    return df


def visualize_sample(root: Path, csv_path: Path, attribute: str, index: int, output_path: Path):
    df = load_annotations(csv_path)

    print("CSV shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nAvailable attributes:")
    print(df["attribute_name"].drop_duplicates().to_list())

    sub = df[df["attribute_name"] == attribute]

    if sub.empty:
        raise ValueError(f"No samples found for attribute: {attribute}")

    row = sub.iloc[index % len(sub)]

    image_path = root / row["image_path"]

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    title = (
        f"Attribute: {row['attribute_name']}\n"
        f"Label vector: {row['label']} | Class index: {row['class_index']}\n"
        f"Image: {image_path.name}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 7))
    plt.imshow(image)
    plt.title(title, fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("\nSelected row:")
    print(row)
    print(f"\nSaved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--attribute", default="collar_design_labels")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="outputs/fashionai_collar_sample.png")
    args = parser.parse_args()

    visualize_sample(
        root=Path(args.root),
        csv_path=Path(args.csv),
        attribute=args.attribute,
        index=args.index,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
