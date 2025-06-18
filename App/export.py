from ultralytics import YOLO

model = YOLO("model.pt")

# Export ONNX with simplify, dynamic shape, opset 17 and NMS baked in
model.export(format="onnx", simplify=True, dynamic=True, opset=17, nms=False)


#yolo export model=model.pt format=onnx opset=17 simplify=True dynamic=True nms=False
#>>