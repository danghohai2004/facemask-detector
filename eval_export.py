import torch
from ultralytics import YOLO
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[0]

def export(model_name:str) -> None:
    model_path = BASE_PATH / model_name
    model = YOLO(model_path)
    model.export(format="engine", half=True)

def evaluate(data_yaml_name:str, model_name:str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    model_path = BASE_PATH / model_name
    datapath = BASE_PATH / "dataset" / data_yaml_name

    model = YOLO(model_path).to(device)

    metrics = model.val(data=datapath)

    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("mAP75:", metrics.box.map75)

    print("Precision:", metrics.box.p)
    print("Recall:", metrics.box.r)

if __name__ == "__main__":
    export(model_name = "best.pt")
