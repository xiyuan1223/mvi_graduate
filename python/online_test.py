
from config_2d import opt
import os
import torch as t
import models
from online_test_dataset import dataset_2d
from torch.utils.data import DataLoader
from torchnet import meter

from tqdm import tqdm
import numpy as np
import fire
@t.no_grad()  # pytorch>=0.5
def test(load_model_path = "checkpoints/resnet34_1203_03_05_14.pth",test_data="test_data",**kwargs):
    opt._parse(kwargs)
    confusion_matrix = meter.ConfusionMeter(2)
    # configure model
    model = getattr(models, opt.model)().eval()

    model.load(load_model_path)
    model.eval()
    # model = t.nn.DataParallel(model, device_ids=opt.device_ids)  # 声明所有可用设备
    model.to(opt.device)  # 模型放在主设备
    # data
    test_data = dataset_2d(test_data, train=False)
    test_dataloader = DataLoader(test_data, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []
    user_pos_count = {}
    user_prob_rec = {}
    user_img_count = test_dataloader.dataset.user_img_count
    all_score = []
    for ii, (data, label) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)  # 数据放在主设备
        score = model(input)
        # update user_pos_count
        item_predict = t.argmax(score, dim=1).item()
        probility = t.softmax(score,dim=1)[0][item_predict].item()
        all_score.append(probility)
    probility = sum(all_score)/len(all_score)
    return probility
   


if __name__ == "__main__":
    fire.Fire(test)