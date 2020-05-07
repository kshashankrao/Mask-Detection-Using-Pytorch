from torchvision import transforms
import numpy as np
import torch
import json
import PIL
import cv2
import os

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir,"output")

    config_path = os.path.join(root_dir,"config", "config.json")
    cfg_file = open(config_path)
    cfg = json.load(cfg_file)
    image_size = cfg["image_size"]

    face_model_path = os.path.join(root_dir, "models")
    prototxtPath = os.path.join(face_model_path, "deploy.prototxt")
    weightsPath = os.path.join(face_model_path,"res10_300x300_ssd_iter_140000.caffemodel")
    face_model = cv2.dnn.readNet(prototxtPath, weightsPath)

    model_path = os.path.join(output_dir, "model.pth")
    mask_model = torch.load(model_path)

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    transform = data_transforms['test']
    idx_to_class = {0: 'with_mask', 1: 'without_mask'}

    cap = cv2.VideoCapture(0)

    while(True):
        
        ret, frame = cap.read()
        frame = cv2.resize(frame, (400,400))
        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_model.setInput(blob)
        detections = face_model.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.80:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = frame[startY:endY, startX:endX]

                image = PIL.Image.fromarray(frame)
                image_tensor = transform(image)

                if torch.cuda.is_available():
                    image_tensor = image_tensor.view(1, 3, image_size, image_size).cuda()
                else:
                    image_tensor = image_tensor.view(1, 3, image_size, image_size)
                
                with torch.no_grad():
                    mask_model.eval()
                    out = mask_model(image_tensor)
                    ps = torch.exp(out)
                    topk, topclass = ps.topk(1, dim=1)
                    print("Output class :  ", idx_to_class[topclass.cpu().numpy()[0][0]])
                
                cv2.rectangle(frame, (startX - 3, startY - 3), (endX + 3, endY + 3), (0, 255, 0), 2)
                cv2.imshow('frame',frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
	

if __name__ == "__main__":
    main()

    
