import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import pandas as pd
import random
import re


translations = {'Cesta': 'Road', 'Pločnik': 'Sidewalk', 'Zgrada': 'Building', 'Zid': 'Wall', 'Ograda': 'Fence', 
                'Stup': 'Pole', 'Semafor': 'Traffic light', 'Prometni znak': 'Traffic sign', 'Vegetacija': 'Vegetation',
                'Prirodni teren': 'Terrain', 'Nebo': 'Sky', 'Osoba': 'Person', 'Vozač': 'Rider', 'Automobil': 'Car',
                'Kamion': 'Truck', 'Autobus': 'Bus', 'Vlak': 'Train', 'Motocikl': 'Motorcycle', 'Bicikl': 'Bicycle'}


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_miou_per_epoch(log_files, title, pair=False):
    values = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            miou_epochs = [float(i) for i in re.findall(r'IoU mean class accuracy -> TP / \(TP\+FN\+FP\) = (\d+\.\d+)', f.read())]

            if log_file.parent.parent.stem == 'supervised':
                elements_to_remove = random.sample(miou_epochs, 7)  # We remove 7 random elements due to the fact that we had to resume training so we got more epochs than we wanted
                for element in elements_to_remove:
                    miou_epochs.remove(element)

        values.append(miou_epochs)
    
    for miou_epochs, log_file in zip(values, log_files):
        epochs = list(range(4, 4 * len(miou_epochs) + 1, 4))
        if log_file.parent.parent.stem == 'supervised':
            train_type = 'supervised'
        elif pair:
            train_type = f'{log_file.parent.parent.stem} phase'
        else:
            train_type = f'{log_file.parent.parent.parent.stem}'.replace('_', ' ')
        plt.plot(epochs, miou_epochs, label=train_type)
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title('Mean IoU per epoch')
    plt.legend()
    plt.savefig(f'{title}_mean_iou_per_epoch.png', format='png')
    plt.close()


def miou_per_epoch():
    log_files = list(models_dir.glob('**/log.txt'))

    first_phase = [log_file for log_file in log_files if log_file.parent.parent.stem == 'first']
    second_phase = [log_file for log_file in log_files if log_file.parent.parent.stem in ['supervised', 'second']]

    plot_miou_per_epoch(first_phase, title='first_phase')
    plot_miou_per_epoch(second_phase, title='second_phase')

    pairs = [[log1, log2] for log1 in first_phase for log2 in second_phase 
             if log1.parent.parent.parent.stem == log2.parent.parent.parent.stem]
    for pair in pairs:
        plot_miou_per_epoch(pair, title=f'{pair[0].parent.parent.parent.stem}_both_phases', pair=True)


def plot_iou_per_class(log_file, train_type, class_iou_combined):
    _, best_epoch, best_miou = util.read_last_and_best_epoch(log_file.parent)
    class_iou = []
    with open(log_file, 'r') as f:
        for line in f:
            if f'Epoch: {best_epoch} /' in line:
                while 'Errors' not in line:
                    line = next(f)
                line = next(f)
                while 'IoU accuracy' in line:
                    iou = re.findall(r'IoU accuracy = (\d+\.\d+) %', line)[0]
                    class_iou.append(float(iou))
                    line = next(f)
                break

    for key, iou in zip(translations.keys(), class_iou):
        class_iou_combined[key][train_type] = iou
    class_iou_combined['mIoU'][train_type] = best_miou


def iou_per_class():
    log_files = list(models_dir.glob('**/log.txt'))
    class_iou_combined = {key: {} for key in translations.keys()}
    class_iou_combined['mIoU'] = {}

    for log_file in log_files:
        if log_file.parent.parent.stem == 'supervised':
            train_type = 'supervised'
        else:
            train_type = f'{log_file.parent.parent.parent.stem} - {log_file.parent.parent.stem} phase'.replace('_', ' ').replace(' percent', '%')
        
        plot_iou_per_class(log_file, train_type, class_iou_combined)

    df = pd.DataFrame({'Klasa': [f'{key} ($\\mathit{{eng.}} {value})$' for key, value in translations.items()] + ['mIoU']})
    for train_type in set(class_iou_combined[next(iter(class_iou_combined))].keys()):
        df[train_type] = [class_iou_combined[key].get(train_type, None) for key in class_iou_combined.keys()]
    
    desired_order = ['Klasa', 'supervised', '50% - first phase', '50% - second phase', '25% - first phase', '25% - second phase']
    df = df[desired_order]

    fig, ax = plt.subplots(figsize=(12, 18))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')

    col_widths = [0.3] + [0.2] * (len(df.columns) - 1)  # Wider first column and narrower others

    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     cellLoc='center', 
                     loc='center', 
                     colColours=["#ffffff"] * df.shape[1], 
                     colWidths=col_widths)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust table scale

    row_colors = ['#f0f0f0', '#d9d9d9']
    for i in range(len(df)):
        color = row_colors[i % 2]
        for j in range(len(df.columns)):
            table[(i+1, j)].set_facecolor(color)

    header_color = "#4f81bd"
    for j in range(len(df.columns)):
        table[(0, j)].set_facecolor(header_color)
        table[(0, j)].set_text_props(color='white', ha='center', va='center')

    fig.tight_layout()
    plt.savefig('iou_per_class.png', format='png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    util_path = Path(__file__).parent.parent.parent.parent / 'models' / 'util.py'
    util = import_module(util_path)

    models_dir = Path(__file__).parent.parent.parent.parent / 'weights'

    miou_per_epoch()
    iou_per_class()