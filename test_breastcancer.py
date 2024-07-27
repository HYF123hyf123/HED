import torch
import torchvision.transforms as transforms
import torchvision.models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models
import pandas as pd
from torch.autograd import Variable
from torchvision.models import resnet50  
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from train_FGVC_Aircraft_ResNet50 import Network_Wrapper
from train_FGVC_Aircraft_ResNet50 import Features
from utils import *
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AA = '**'
"""1"""
model = torch.load(f'./results/models/model_{AA}.pth')
model = model.to(device)
model.eval()

test_loss = 0
correct = 0
correct_com = 0
correct_com2 = 0
total = 0
idx = 0
use_cuda = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss()

data_transforms = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

"""2"""
test_dataset = ImageFolder(root=f'./data/Data_{AA}/test', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


results = pd.DataFrame(columns=['file', 'prediction'])


with torch.no_grad():
    results = []
    for batch_idx, (inputs, targets) in enumerate(test_loader):

        img_path, _ = test_loader.dataset.samples[batch_idx]
        img_name = os.path.basename(img_path)
        print(img_name)

        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_concat, map1, map2, map3 = model(inputs)

        p1 = model.state_dict()['classifier3.1.weight']
        p2 = model.state_dict()['classifier3.4.weight']
        att_map_3 = map_generate(map3, output_3, p1, p2)

        p1 = model.state_dict()['classifier2.1.weight']
        p2 = model.state_dict()['classifier2.4.weight']
        att_map_2 = map_generate(map2, output_2, p1, p2)

        p1 = model.state_dict()['classifier1.1.weight']
        p2 = model.state_dict()['classifier1.4.weight']
        att_map_1 = map_generate(map1, output_1, p1, p2)

        inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)
        output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = model(inputs_ATT)

        # print(output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT)

        outputs_com2 = output_1 + output_2 + output_3 + output_concat
        outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        _, predicted_com2 = torch.max(outputs_com2.data, 1)

        results.append({
            'file': img_name,
            'predicted': predicted.cpu().numpy(),
            'predicted_com': predicted_com.cpu().numpy(),
            'predicted_com2': predicted_com2.cpu().numpy(),
            'true_label': targets.data.cpu().numpy()
        })

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_com += predicted_com.eq(targets.data).cpu().sum()
        correct_com2 += predicted_com2.eq(targets.data).cpu().sum()

        if batch_idx % 5 == 0:
            print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1),
                100. * float(correct_com) / total, correct_com, total))
    """3"""
    with open(f'./results/result_{AA}/pre_{AA}.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['filename','predicted', 'predicted_com', 'predicted_com2', 'true_label']
        writer.writerow(header)
        for result in results:
            row = [str(result['file']), result['predicted'], result['predicted_com'], result['predicted_com2'], result['true_label']]
            writer.writerow(row)


    y_true = np.array([res['true_label'] for res in results])
    y_pred = np.array([res['predicted'] for res in results])


    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')


    # fpr, tpr, _ = roc_curve(y_true, y_pred)
    # roc_auc = auc(fpr, tpr)


    df = pd.DataFrame({
        'Precision': [precision],
        'Recall': [recall],
        'Accuracy': [accuracy],
        'F1-Score': [f1],
        # 'AUC': [roc_auc]
    })
    """4"""
    df.to_excel(f'./results/result_{AA}/evaluation_index_{AA}.xlsx')


    plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

