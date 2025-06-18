import zipfile
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

zip_folder = r'C:\Facultate\Licenta\ECP dataset'
extract_root = r'C:\Facultate\Licenta\ECP dataset'

USE_PARALLEL_EXTRACTION = True

day_zip_files = [
    "ECP_day_img_train", "ECP_day_img_val",
    "ECP_day_labels_train", "ECP_day_labels_val"
]

night_zip_files = [
    "ECP_night_img_train", "ECP_night_img_val",
    "ECP_night_labels_train", "ECP_night_labels_val"
]

max_per_city_train = 150
max_per_city_val = 25

city_image_counts_train = defaultdict(int)
city_image_counts_val = defaultdict(int)

def extract_day_zip(zip_path, modality, split):
    images_out_dir = os.path.join(extract_root, "day", split, "images")
    labels_out_dir = os.path.join(extract_root, "day", split, "labels")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = []
        for member in zip_ref.infolist():
            if member.is_dir():
                continue
            parts = member.filename.split('/')
            if len(parts) < 5:
                continue
            city = parts[4].lower()

            if modality == 'img':
                if split == 'train' and city_image_counts_train[city] >= max_per_city_train:
                    continue
                if split == 'val' and city_image_counts_val[city] >= max_per_city_val:
                    continue

            members.append(member)

        print(f"📦 Extracting {len(members)} files from {os.path.basename(zip_path)}")

        def extract_member(member):
            parts = member.filename.split('/')
            filename = os.path.basename(member.filename)
            city = parts[4].lower()

            with zip_ref.open(member) as source:
                data = source.read()

            if modality == 'img':
                if split == 'train':
                    if city_image_counts_train[city] >= max_per_city_train:
                        return None
                    city_image_counts_train[city] += 1
                else:
                    if city_image_counts_val[city] >= max_per_city_val:
                        return None
                    city_image_counts_val[city] += 1

                dest_path = os.path.join(images_out_dir, filename)
                with open(dest_path, 'wb') as f:
                    f.write(data)
            else:
                if split == 'train' and city_image_counts_train[city] == 0:
                    return None
                if split == 'val' and city_image_counts_val[city] == 0:
                    return None

                dest_path = os.path.join(labels_out_dir, filename)
                with open(dest_path, 'wb') as f:
                    f.write(data)

            return filename

        if USE_PARALLEL_EXTRACTION:
            list(tqdm(ThreadPoolExecutor().map(extract_member, members), total=len(members)))
        else:
            for member in tqdm(members):
                extract_member(member)

        print(f"✅ Done extracting day {modality} {split} from {os.path.basename(zip_path)}")

def extract_night_zip(zip_path, modality, split):
    images_out_dir = os.path.join(extract_root, "night", split, "images")
    labels_out_dir = os.path.join(extract_root, "night", split, "labels")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = [m for m in zip_ref.infolist() if not m.is_dir() and len(m.filename.split('/')) >= 5]

        print(f"📦 Extracting {len(members)} files from {os.path.basename(zip_path)}")

        def extract_member(member):
            filename = os.path.basename(member.filename)
            dest_dir = images_out_dir if modality == 'img' else labels_out_dir
            dest_path = os.path.join(dest_dir, filename)

            with zip_ref.open(member) as source:
                data = source.read()

            with open(dest_path, 'wb') as f:
                f.write(data)

            return filename

        if USE_PARALLEL_EXTRACTION:
            list(tqdm(ThreadPoolExecutor().map(extract_member, members), total=len(members)))
        else:
            for member in tqdm(members):
                extract_member(member)

        print(f"✅ Done extracting night {modality} {split} from {os.path.basename(zip_path)}")

# Run extraction
for zip_name in day_zip_files:
    zip_path = os.path.join(zip_folder, zip_name + ".zip")
    modality = 'img' if 'img' in zip_name else 'labels'
    split = 'train' if 'train' in zip_name else 'val'
    extract_day_zip(zip_path, modality, split)

for zip_name in night_zip_files:
    zip_path = os.path.join(zip_folder, zip_name + ".zip")
    modality = 'img' if 'img' in zip_name else 'labels'
    split = 'train' if 'train' in zip_name else 'val'
    extract_night_zip(zip_path, modality, split)

print("\n🎉 All day and night data extracted successfully.")
