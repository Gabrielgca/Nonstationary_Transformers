from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, Informer, Autoformer
from ns_models import ns_Transformer, ns_Informer, ns_Autoformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_t
from utils.metrics import metric
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

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
                # break  # DEBUGGING ONLY RUN 1 BATCH PER EPOCH
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
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
                # break  # DEBUGGING ONLY RUN 1 BATCH PER EPOCH

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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        preds_abrupt = []
        trues_abrupt = []

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
                # print("i: ", i)

                if i % (self.args.seq_len + self.args.pred_len) == 0:
                    # 2) sliding window feature extraction
                    
                    dbscan_params = {
                                    'HUFL': [0.8, 3],
                                    'HULL': [0.3, 2],
                                    'MUFL': [0.3, 2],
                                    'MULL': [0.8, 3],
                                    'LUFL': [0.8, 3],
                                    'LULL': [1.0, 3],
                                    'OT':   [0.5, 3],
                                    
                                    }
                    found_vars = []
                    found_vars_indexes = []
                    keys_list = list(dbscan_params.keys())
                    for params in dbscan_params:
                        var_idx = keys_list.index(params)
                        if self.args.features == 'S':
                            var_idx = -1

                        # print("var_idx", var_idx)
                        eps = dbscan_params[params][0]
                        min_samples = dbscan_params[params][1]
                        
                        # print("Testing eps: ", eps, " min_samples: ", min_samples, "for variable: ", params)

                        
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, var_idx], true[0, :, var_idx]), axis=0)
                        
                        # print("y shape ", gt)
                        outlier_samples_all = []

                        w = gt[self.args.seq_len: self.args.seq_len + self.args.pred_len]

                        # print("w shape: ", w.shape)
                        w_diff = np.diff(w)
                        w_diff_abs = np.abs(w_diff)
                        # print("w_diff_abs shape: ", w_diff_abs.shape)
                        scaler = StandardScaler()
                        X = scaler.fit_transform(w_diff_abs.reshape(-1, 1))
                        
                        cl = DBSCAN(eps=eps, min_samples=min_samples)   
                        labels = cl.fit_predict(X) 

                        outlier_windows_idx = np.where(labels == -1)[0]

                        if len(outlier_windows_idx) > 0:
                            found_vars.append(params)
                            found_vars_indexes.append(var_idx)
                            for idx in outlier_windows_idx:
                                global_index = self.args.seq_len + idx

                                outlier_samples_all.append([global_index, i])
                        # print("## i: ", i)
                        
                        outlier_samples = [x[0] for x in outlier_samples_all]

                        outlier_samples = np.unique(outlier_samples)
                        # print("Number of abrupt changes: " ,len(outlier_samples))

                        if len(outlier_samples) > 0:
                            preds_abrupt.append(pred)
                            trues_abrupt.append(true)

                            for var_idx in found_vars_indexes:
                                if self.args.features == 'S':
                                    var_name = 'OT'
                                else:
                                    var_name = found_vars[found_vars_indexes.index(var_idx)]
                                # print("var_name", var_name)
                                gt = np.concatenate((input[0, :, var_idx], true[0, :, var_idx]), axis=0)
                                pd = np.concatenate((input[0, :, var_idx], pred[0, :, var_idx]), axis=0)

                                t = np.arange(i,i+self.args.seq_len + self.args.pred_len)
                                visual_t(gt, pd, t, outlier_samples, var_name, os.path.join(folder_path, str(i) + '_' + var_name + '.pdf'))
                        else:
                            continue
                            # print("No abrupt changes found")

                        # list_variables_abrupt, variable_idxs = test_data.get_list_variables_abrupt(i)

                    # print("is_abrupt:", eval(is_abrupt.values[0]), "dbscan_start:", eval(dbscan_start.values[0]))

                    # if len(list_variables_abrupt) > 0 :
                    #     preds_abrupt.append(pred)
                    #     trues_abrupt.append(true)

                        
                        # found_vars = []
                        # for var in list_variables_abrupt:
                        #     if var in variable_idxs:
                        #         found_vars.append(variable_idxs[var])

                        # for var_idx in found_vars:

                        #     var_name = [key for key, value in variable_idxs.items() if value == var_idx][0]

                        #     input = batch_x.detach().cpu().numpy()
                        #     gt = np.concatenate((input[0, :, var_idx], true[0, :, var_idx]), axis=0)
                        #     pd = np.concatenate((input[0, :, var_idx], pred[0, :, var_idx]), axis=0)

                        #     t = np.arange(i,i+self.args.seq_len + self.args.pred_len)
                        #     visual_t(gt, pd, t, os.path.join(folder_path, str(i) + '_' + var_name + '.pdf'))
                

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

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


        preds_abrupt = np.array(preds_abrupt)
        trues_abrupt = np.array(trues_abrupt)
        print('test shape:', preds_abrupt.shape, trues_abrupt.shape)
        preds_abrupt = preds_abrupt.reshape(-1, preds_abrupt.shape[-2], preds_abrupt.shape[-1])
        trues_abrupt = trues_abrupt.reshape(-1, trues_abrupt.shape[-2], trues_abrupt.shape[-1])
        print('test shape:', preds_abrupt.shape, trues_abrupt.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae_abrupt, mse_abrupt, rmse, mape, mspe = metric(preds_abrupt, trues_abrupt)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('mse_abrupt:{}, mae_abrupt:{}'.format(mse_abrupt, mae_abrupt))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\nmse_abrupt:{}, mae_abrupt:{}'.format(mse_abrupt, mae_abrupt))
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
