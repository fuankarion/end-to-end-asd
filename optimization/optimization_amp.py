import os
import torch

from torch.cuda.amp import autocast
from models.graph_layouts import generate_av_mask
from sklearn.metrics import average_precision_score

from models.graph_layouts import generate_temporal_video_center_mask, generate_temporal_video_mask


def optimize_easee(model, dataloader_train, data_loader_val,
                               device, criterion, optimizer, scheduler,
                               num_epochs, spatial_ctx_size, time_len,
                               a_weight=0.2, v_weight=0.5, models_out=None,
                               log=None):

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        outs_train = _train_model_amp_avl(model, dataloader_train, optimizer,
                                                    criterion, device, spatial_ctx_size, time_len,
                                                    a_weight, v_weight)
        outs_val = _test_model_graph_losses(model, data_loader_val, criterion,
                                            device, spatial_ctx_size, time_len)
        scheduler.step()

        train_loss, ta_loss, tv_loss, train_ap = outs_train
        val_loss, va_loss, vv_loss, val_ap, val_tap, val_cap = outs_val

        if models_out is not None and epoch > num_epochs-10:  # just save last 10 epochs
            model_target = os.path.join(models_out, str(epoch+1)+'.pth')
            print('save model to ', model_target)
            torch.save(model.state_dict(), model_target)

        if log is not None:
            log.writeDataLog([epoch+1, train_loss, ta_loss, tv_loss,
                             train_ap, val_loss, va_loss, vv_loss, val_ap, val_tap, val_cap])

    return model


def _train_model_amp_avl(model, dataloader, optimizer, criterion,
                                   device, ctx_size, time_len, a_weight,
                                   v_weight):
    model.train()
    softmax_layer = torch.nn.Softmax(dim=1)

    pred_lst = []
    label_lst = []

    pred_time_lst = []
    label_time_lst = []

    pred_center_lst = []
    label_center_lst = []

    running_loss_g = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0

    audio_size = dataloader.dataset.get_audio_size()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Iterate over data
    for idx, dl in enumerate(dataloader):
        print('\t Train iter {:d}/{:d} {:.4f}'.format(idx, len(dataloader), running_loss_g/(idx+1)), end='\r')

        graph_data = dl
        graph_data = graph_data.to(device)
        targets = graph_data.y

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # TODO inneficient here
            audio_mask, video_mask = generate_av_mask(ctx_size, graph_data.x.size(0))
            temporal_video_mask = generate_temporal_video_mask(ctx_size, graph_data.x.size(0))
            center_mask = generate_temporal_video_center_mask(ctx_size, graph_data.x.size(0), time_len)

            with autocast(True):
                outputs, audio_out, video_out = model(graph_data, ctx_size, audio_size)
                aux_loss_a = criterion(audio_out, targets[audio_mask])
                aux_loss_v = criterion(video_out, targets[video_mask])
                loss_graph = criterion(outputs, targets)
                loss = a_weight*aux_loss_a + v_weight*aux_loss_v + loss_graph

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.set_grad_enabled(False):
            label_lst.extend(targets[video_mask].cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(outputs[video_mask]).cpu().numpy()[:, 1].tolist())

            label_time_lst.extend(targets[temporal_video_mask].cpu().numpy().tolist())
            pred_time_lst.extend(softmax_layer(outputs[temporal_video_mask]).cpu().numpy()[:, 1].tolist())

            label_center_lst.extend(targets[center_mask].cpu().numpy().tolist())
            pred_center_lst.extend(softmax_layer(outputs[center_mask]).cpu().numpy()[:, 1].tolist())

        # statistics
        running_loss_g += loss_graph.item()
        running_loss_a += aux_loss_a.item()
        running_loss_v += aux_loss_v.item()
        if idx == len(dataloader)-2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_time_ap = average_precision_score(label_time_lst, pred_time_lst)
    epoch_center_ap = average_precision_score(label_center_lst, pred_center_lst)
    print('Train Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, VmAP: {:.4f}, TVmAP: {:.4f}, CVmAP: {:.4f}'.format(
        epoch_loss_g, epoch_loss_a, epoch_loss_v, epoch_ap, epoch_time_ap, epoch_center_ap))
    return epoch_loss_g, epoch_loss_a, epoch_loss_v, epoch_ap


def _test_model_graph_losses(model, dataloader, criterion, device, ctx_size, time_len):
    model.eval()
    softmax_layer = torch.nn.Softmax(dim=1)

    pred_lst = []
    label_lst = []

    pred_time_lst = []
    label_time_lst = []

    pred_center_lst = []
    label_center_lst = []

    running_loss_g = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0

    audio_size = dataloader.dataset.get_audio_size()

    # Iterate over data
    for idx, dl in enumerate(dataloader):
        print('\t Val iter {:d}/{:d} {:.4f}'.format(idx,
              len(dataloader), running_loss_g/(idx+1)), end='\r')

        graph_data = dl
        graph_data = graph_data.to(device)
        targets = graph_data.y

        with torch.set_grad_enabled(False):
            # TODO inneficient here
            audio_mask, video_mask = generate_av_mask(
                ctx_size, graph_data.x.size(0))
            temporal_video_mask = generate_temporal_video_mask(
                ctx_size, graph_data.x.size(0))
            center_mask = generate_temporal_video_center_mask(
                ctx_size, graph_data.x.size(0), time_len)

            outputs, audio_out, video_out = model(
                graph_data, ctx_size, audio_size)
            loss_graph = criterion(outputs, targets)
            aux_loss_a = criterion(audio_out, targets[audio_mask])
            aux_loss_v = criterion(video_out, targets[video_mask])

            label_lst.extend(targets[video_mask].cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(
                outputs[video_mask]).cpu().numpy()[:, 1].tolist())

            label_time_lst.extend(
                targets[temporal_video_mask].cpu().numpy().tolist())
            pred_time_lst.extend(softmax_layer(
                outputs[temporal_video_mask]).cpu().numpy()[:, 1].tolist())

            label_center_lst.extend(
                targets[center_mask].cpu().numpy().tolist())
            pred_center_lst.extend(softmax_layer(
                outputs[center_mask]).cpu().numpy()[:, 1].tolist())

        # statistics
        running_loss_g += loss_graph.item()
        running_loss_a += aux_loss_a.item()
        running_loss_v += aux_loss_v.item()

        if idx == len(dataloader)-2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_time_ap = average_precision_score(label_time_lst, pred_time_lst)
    epoch_center_ap = average_precision_score(
        label_center_lst, pred_center_lst)
    print('Val Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, VmAP: {:.4f}, TVmAP: {:.4f}, CVmAP: {:.4f}'.format(
        epoch_loss_g, epoch_loss_a, epoch_loss_v, epoch_ap, epoch_time_ap, epoch_center_ap))
    return epoch_loss_g, epoch_loss_a, epoch_loss_v, epoch_ap, epoch_time_ap, epoch_center_ap
