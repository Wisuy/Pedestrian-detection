import os
from PIL import Image

folder = r'C:\Facultate\Licenta\ECP dataset\day\train\images'
max_dim = 640

def resize_in_place(img_path, max_dim):
    with Image.open(img_path) as img:
        img_format = img.format
        img = img.convert("RGB")
        w, h = img.size

        if max(w, h) <= max_dim:
            return

        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        resized_img = img.resize(new_size, Image.LANCZOS)

        if img_format == "PNG":
            resized_img.save(img_path, format="PNG", optimize=True)
        else:
            resized_img.save(img_path, format="JPEG", quality=95, optimize=True)

count = 0
for filename in os.listdir(folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder, filename)
        try:
            resize_in_place(img_path, max_dim)
            count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"Done: Resized {count} images in-place in {folder}")
