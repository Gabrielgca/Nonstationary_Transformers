from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, Informer, Autoformer
from ns_models import ns_Transformer, ns_Informer, ns_Autoformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_loader import Dataset_VVUser
from sklearn.preprocessing import StandardScaler
import copy

import os
import time
import csv

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'ns_Transformer': ns_Transformer,
            'ns_Informer': ns_Informer,
            'ns_Autoformer': ns_Autoformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, loso=False, sub_sampling=False):
        if loso:
            self.train_with_loso(setting, sub_sampling)
            return
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                # print("batch_y shape: ", batch_y.shape)
                # print("batch_x shape: ", batch_x.shape)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def train_with_loso(self, setting, sub_sampling=False):
        dataset = Dataset_VVUser(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag='train',
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            timeenc=0 if self.args.embed != 'timeF' else 1,
            freq=self.args.freq
        )
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if dataset.number_of_participants is None:
            raise ValueError("Dataset_VVUser must have 'number_of_participants' attribute for LOSO training.")
        
        participant_count = dataset.number_of_participants

        all_mse = []
        all_mae = []

        print("Total participants for LOSO:", participant_count)
        for participant_id in range(participant_count):
            print(f"Starting LOSO training for participant ID: {participant_id}")
            train_data = copy.deepcopy(dataset)

            print("Total windows in dataset:", len(dataset.windows))
            train_data.windows = [w for w in train_data.windows if w[-1] != participant_id]  # Exclude current participant

            print("train_data in dataset:", len(train_data.windows))
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=True
            )
            # dataset de teste: apenas test_id
            test_data = copy.deepcopy(dataset)
            test_data.windows = [w for w in test_data.windows
                                if w[-1] == participant_id]  # Include only current participant
            
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=1,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=False
            )

            print("test_data in dataset:", len(test_data.windows))

            # Standardize data based on training set
            # CHECK SCALER TO FIND DIFF BETWEEN SIN/COS AND EUCLIDEAN
            scaler = StandardScaler()
            
            all_train_x = []
            # print("train_data.windows length:", len(train_data.windows))
            for (x, y, x_mark, y_mark, pid) in train_data.windows:
                # print("Fitting scaler - participant ID:", pid, "x shape:", x.shape)
                all_train_x.append(x)  # x tem shape (seq_len, 1)

            all_train_x = np.vstack(all_train_x)   # shape final: (num_windows * seq_len, 1)
            # print("Fitting scaler on:", all_train_x.shape)
            scaler.fit(all_train_x)

            # print("Fitting scaler for participant ID:", participant_id)
            # print("Training data shape for scaler fitting:", all_train_x.shape)
            # scaler.fit(train_data_array)

            # Scaling training and testing data
            for i in range(len(train_data.windows)):
                x, y, x_mark, y_mark, pid = train_data.windows[i]
                x = scaler.transform(x)
                y = scaler.transform(y)
                train_data.windows[i] = (x, y, x_mark, y_mark, pid)

            for i in range(len(test_data.windows)):
                x, y, x_mark, y_mark, pid = test_data.windows[i]
                x = scaler.transform(x)
                y = scaler.transform(y)
                test_data.windows[i] = (x, y, x_mark, y_mark, pid)

            self.model = self._build_model().to(self.device)

            model_optim = self._select_optimizer()
            criterion = self._select_criterion()
        
            time_now = time.time()

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)



            if self.args.use_amp:
                scaler = torch.amp.GradScaler()

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                        # break  # for testing purposes, remove this break for full training

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Loss (MSE) on participant ID {}: {:.4f}".format(participant_id, test_loss))
                all_mse.append(test_loss)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, test_loss))
                early_stopping(test_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

            mae, mse = self.eval_model(self.model, scaler, test_loader, dataset, self.device, setting, participant_id, sub_sampling=sub_sampling)

            print("###################### mae", mae)
            print("###################### mse", mse)
            all_mae.append(mae)
            all_mse.append(mse)
            # break  # for testing purposes, remove this break for full LOSO training
        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        print("LOSO Training completed.")
        print("Average MAE across participants: ", np.mean(all_mae))
        print("Average MSE across participants: ", np.mean(all_mse))

        # # result save
        folder_path = './results/' + setting + '/' + 'sub_sampling_' + str(sub_sampling) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('Sub_sampling: ' + str(sub_sampling) + '\n')
        f.write('Average MAE: ' + str(np.mean(all_mae)) + '\n')
        f.write('Average MSE: ' + str(np.mean(all_mse)) + '\n')
        f.write('\n')
        f.write('\n')
        f.close()

        return self.model


    def train_epoch(self, model, loader, opt, loss_fn, device):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)   # [B,L,1]
            y = y.to(device)   # [B]
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    def eval_model(self, model, scaler, loader, dataset, device, setting, participant_id, sub_sampling=False):
        model.eval()
        tot = 0.0

        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                pred_raw  = scaler.inverse_transform(pred[0,:,:])
                true_raw  = scaler.inverse_transform(true[0,:,:])


                pred_radians = np.atan2(pred_raw[:, 0], pred_raw[:, 1])
                true_radians = np.atan2(true_raw[:, 0], true_raw[:, 1])

                pred_degree = np.rad2deg(pred_radians)
                true_degree  = np.rad2deg(true_radians)

                preds.append(pred_degree.reshape(1,-1))
                trues.append(true_degree.reshape(1,-1))

                iteration = 100
                if sub_sampling:
                    iteration = 21
                if i % iteration == 0:

                    # batch_x_mark: (B, seq_len, 2) = [standard_scaled_time, minmax_scaled_time]
                    # batch_y_mark: (B, label_len+pred_len, 2)

                    t_x  = batch_x_mark[0, :, 0].detach().cpu().numpy().reshape(-1, 1)
                    t_y  = batch_y_mark[0, :, 0].detach().cpu().numpy().reshape(-1, 1)

                    # # inverse-transform using the time scaler *for this participant*
                    # t_x = dataset.time_standard_scaler[participant_id].inverse_transform(t_x_scaled).flatten()
                    # t_y = dataset.time_standard_scaler[participant_id].inverse_transform(t_y_scaled).flatten()

                    # create full time axis for GT and PD
                    t_gt = np.concatenate((t_x, t_y[self.args.label_len:]))

                    input = batch_x.detach().cpu().numpy()
                    # input = batch_x.detach().cpu().numpy()
                    # print("input.shape ", input.shape)
                    input_raw = scaler.inverse_transform(input[0,:,:])


                    input_radians = np.atan2(input_raw[:, 0], input_raw[:, 1])


                    # norm = np.sqrt(input[0, :, 0]**2 + input[0, :, 1]**2)
                    # print(norm)
                    input_degree = np.rad2deg(input_radians)


                    # flatten sequences for CSV
                    input_flat = input_degree.flatten()
                    true_flat = true_degree.flatten()
                    pred_flat = pred_degree.flatten()

                    row = {
                        "participant_id": participant_id,
                        "iteration": i,
                        "sub_sampling": sub_sampling,

                        "input_len": len(input_flat),
                        "true_len": len(true_flat),
                        "pred_len": len(pred_flat),

                        "input": input_flat.tolist(),
                        "true": true_flat.tolist(),
                        "pred": pred_flat.tolist(),

                        "t_input": t_x.tolist(),
                        "t_true": t_gt.tolist(),   # aligned with true/pred_len
                        "t_pred": t_gt.tolist(),
                        }

                    csv_path = os.path.join(folder_path, f'participant_{participant_id}_results.csv')
                    # write or append the CSV
                    file_exists = os.path.isfile(csv_path)

                    with open(csv_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=row.keys())

                        # only write header if file is new
                        if not file_exists:
                            writer.writeheader()

                        writer.writerow(row)

                    gt = np.concatenate((input_degree, true_degree), axis=0)
                    pd = np.concatenate((input_degree, pred_degree), axis=0)
                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, t_gt, os.path.join(folder_path, 'part_' + str(participant_id) + '_' + str(i) + '.pdf'))
                    # break  # for testing purposes, remove this break for full evaluation
            
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # # result save
            # folder_path = './results/' + setting + '/'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            # f = open("result.txt", 'a')
            # f.write(setting + "  \n")
            # f.write('mse:{}, mae:{}'.format(mse, mae))
            # f.write('\n')
            # f.write('\n')
            # f.close()
        return mae, mse

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i == 1 or i == 2 or i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
