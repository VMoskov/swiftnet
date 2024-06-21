import matplotlib.pyplot as plt
import pandas as pd
import importlib.util
from pathlib import Path
import random
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, required=True)

translations = {'Cesta': 'Road', 'Pločnik': 'Sidewalk', 'Zgrada': 'Building', 'Zid': 'Wall', 'Ograda': 'Fence', 
                'Stup': 'Pole', 'Semafor': 'Traffic light', 'Prometni znak': 'Traffic sign', 'Vegetacija': 'Vegetation',
                'Prirodni teren': 'Terrain', 'Nebo': 'Sky', 'Osoba': 'Person', 'Vozač': 'Rider', 'Automobil': 'Car',
                'Kamion': 'Truck', 'Autobus': 'Bus', 'Vlak': 'Train', 'Motocikl': 'Motorcycle', 'Bicikl': 'Bicycle'}


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_miou_per_epoch(log_file, train_type):
    with open(log_file, 'r') as f:
        miou_epochs = [float(i) for i in re.findall(r'IoU mean class accuracy -> TP / \(TP\+FN\+FP\) = (\d+\.\d+)', f.read())]

    if train_type == 'supervised':
        elements_to_remove = random.sample(miou_epochs, 7)  # we remove 7 random elements due to the fact that we had to resume training so we got more epochs than we wanted
        for element in elements_to_remove:
            miou_epochs.remove(element)
    else:
        train_type = f'{log_file.parent.parent.parent.stem}/{train_type}'

    epochs = list(range(4, 4 * len(miou_epochs) + 1, 4))

    output_dir = Path(train_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.plot(epochs, miou_epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title('Mean IoU per epoch')
    plt.savefig(output_dir / 'mean_iou_per_epoch.png', format='png')
    plt.close()


def plot_iou_per_class(log_file, train_type):
    _, best_epoch, best_miou = util.read_last_and_best_epoch(log_file.parent)
    class_iou = {}
    with open(log_file, 'r') as f:
        for line in f:
            if f'Epoch: {best_epoch} /' in line:
                while 'Errors' not in line:
                    line = next(f)
                line = next(f)
                while 'IoU accuracy' in line:
                    class_name, iou = re.findall(r'(\w+) IoU accuracy = (\d+\.\d+) %', line)[0]
                    class_iou[class_name] = float(iou)
                    line = next(f)
                break

    df = pd.DataFrame({'Klasa': [f'{key} ($\\mathit{{eng.}} {value})$' for key, value in translations.items()],
                       'IoU (%)': class_iou.values()})


    fig, ax = plt.subplots(figsize=(8, 12))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=["#ffffff", "#ffffff"])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust table scale

    row_colors = ['#f0f0f0', '#d9d9d9']
    for i in range(len(df)):
        color = row_colors[i % 2]
        table[(i+1, 0)].set_facecolor(color)  # Alternating grey background for class column
        table[(i+1, 1)].set_facecolor(color)  # Alternating grey background for IoU column

    header_color = "#4f81bd"
    table[(0, 0)].set_facecolor(header_color)  # Blue background for header
    table[(0, 1)].set_facecolor(header_color)

    table[(0, 0)].set_text_props(color='white')
    table[(0, 1)].set_text_props(color='white')

    table[(0, 0)].set_text_props(ha='center', va='center')
    table[(0, 1)].set_text_props(ha='center', va='center')

    fig.tight_layout()
    if train_type != 'supervised':
        train_type = f'{log_file.parent.parent.parent.stem}/{train_type}'
    
    output_dir = Path(train_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'iou_per_class.png', format='png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    util_path = Path(__file__).parent.parent.parent.parent / 'models' / 'util.py'
    util = import_module(util_path)

    log_file = Path(args.log_file)
    train_type = log_file.parent.parent.stem

    plot_miou_per_epoch(log_file, train_type)
    plot_iou_per_class(log_file, train_type)