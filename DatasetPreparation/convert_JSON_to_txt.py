import os
import json

base_label_dir = r'C:\Facultate\Licenta\ECP dataset\ECP\train\labels'

classes = {"pedestrian": 0}

for root, _, files in os.walk(base_label_dir):
    for file in files:
        if not file.endswith('.json'):
            continue
        
        json_path = os.path.join(root, file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_width = data.get("imagewidth", 1920)
        image_height = data.get("imageheight", 1024)
        yolo_lines = []
        
        for obj in data.get("children", []):
            identity = obj.get("identity")
            if identity not in classes:
                continue
            
            x0, y0 = obj["x0"], obj["y0"]
            x1, y1 = obj["x1"], obj["y1"]
            
            x_center = ((x0 + x1) / 2) / image_width
            y_center = ((y0 + y1) / 2) / image_height
            width = (x1 - x0) / image_width
            height = (y1 - y0) / image_height
            
            class_id = classes[identity]
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        if not yolo_lines:
            continue 
        
        txt_path = os.path.splitext(json_path)[0] + '.txt'
        
        with open(txt_path, 'w') as out_f:
            out_f.write('\n'.join(yolo_lines))
        
        print(f"Converted {json_path} -> {txt_path}")

print("Done converting all JSON annotations to YOLO format.")
