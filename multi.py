def train_model(model, data_loaders, optimizer, alpha, beta, device="cpu", num_epochs=500):
    since = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4096
    temperature = 0.37#0.37 best now
    base_temperature = 0.07 #0.1 best now
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        #adjust_learning_rate(optimizer, epoch)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects_img = 0
            running_corrects_txt = 0
            # Iterate over data.
            for idx,(imgs, txts, labels) in enumerate(data_loaders[phase]):
                #warmup_learning_rate(epoch, idx, len(data_loaders[phase]), optimizer)
                # imgs = imgs.to(device)
                # txts = txts.to(device)
                # labels = labels.to(device)
                if torch.sum(imgs != imgs)>1 or torch.sum(txts != txts)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()


                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    view1_feature, view2_feature, view1_predict, view2_predict = model(imgs, txts)

                    if (epoch < 500):
                        loss = calc_loss1(view1_feature, view2_feature, view1_predict, view2_predict, labels, labels, alpha, beta, device, temperature, base_temperature, batch_size)
                    else:
                        loss = calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels, labels, alpha, beta)

                    img_preds = view1_predict
                    txt_preds = view2_predict

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects_img += torch.sum(torch.argmax(img_preds, dim=1) == torch.argmax(labels, dim=1))
                running_corrects_txt += torch.sum(torch.argmax(txt_preds, dim=1) == torch.argmax(labels, dim=1))

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            # epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
            # epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
            t_imgs, t_txts, t_labels = [], [], []
            with torch.no_grad():
                for imgs, txts, labels in data_loaders['test']:
                    if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                            labels = labels.cuda()
                    if (epoch < 500):
                        t_view1_feature, t_view2_feature, _, _ = model(imgs, txts)

                    else:
                        _, _, t_view1_feature, t_view2_feature = model(imgs, txts)
                    t_view1_feature, t_view2_feature = normalize(t_view1_feature, 2), normalize(t_view2_feature, 2)
                    t_imgs.append(t_view1_feature.cpu().numpy())
                    t_txts.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_imgs = np.concatenate(t_imgs)
            t_txts = np.concatenate(t_txts)
            t_labels = np.concatenate(t_labels).argmax(1)
            if epoch < 500:
                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels, dist_method='COS')
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels, dist_method='COS')
            else:
                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels, dist_method='COS')
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels, dist_method='COS')
            print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}  Average: {:.4f}  Best_acc: {:.4f}'.format(phase, epoch_loss, img2text, txt2img, (img2text + txt2img) / 2, best_acc))

            # deep copy the model
            if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
                best_acc = (img2text + txt2img) / 2.
                torch.save(model, './model_cl')
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                test_img_acc_history.append(img2text)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history

def normalize(x, power = 2):
    norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
    out = x.div(norm)
    return out

def calc_loss1(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta, device, temperature, base_temperature, batch_size):
    #print(labels_1.shape)
    view1_feature = normalize(view1_feature, 2)
    view2_feature = normalize(view2_feature, 2)
    #term3 = ((view1_feature - view2_feature) ** 2).sum(1).sqrt().mean()
    labels_1 = torch.argmax(labels_1, dim = 1)
    labels_2 = torch.argmax(labels_2, dim = 1)
    batch_size = labels_1.shape[0]
    #print(labels_1.shape, labels_2.shape)
    labels_1 = labels_1.contiguous().view(-1, 1)
    labels_2 = labels_2.contiguous().view(-1, 1)
    mask_img = torch.eq(labels_1, labels_1.T).float().to(device)
    mask_txt = torch.eq(labels_2, labels_2.T).float().to(device)
    mask_img2txt = torch.eq(labels_1, labels_2.T).float().to(device)
    mask_txt2img = torch.eq(labels_2, labels_1.T).float().to(device)
    #print("here1")
    img_contrast = torch.div(
        torch.matmul(view1_feature, view1_feature.T),
        temperature)
    txt_contrast = torch.div(
        torch.matmul(view2_feature, view2_feature.T),
        temperature)
    img2txt_contrast = torch.div(
        torch.matmul(view1_feature, view2_feature.T),
        temperature)
    txt2img_contrast = torch.div(
        torch.matmul(view2_feature, view1_feature.T),
        temperature)
    #print("here2")
    #print("img_contrast")
    #print(img_contrast, txt_contrast)
    logits_img_max, _ = torch.max(img_contrast, dim=1, keepdim=True)
    logits1 = img_contrast - logits_img_max.detach()
    logits_txt_max, _ = torch.max(txt_contrast, dim=1, keepdim=True)
    logits2 = txt_contrast - logits_txt_max.detach()
    logits_img2txt_max, _ = torch.max(img2txt_contrast, dim=1, keepdim=True)
    logits3 = img2txt_contrast - logits_img2txt_max.detach()
    logits_txt2img_max, _ = torch.max(txt2img_contrast, dim=1, keepdim=True)
    logits4 = txt2img_contrast - logits_txt2img_max.detach()
    #print("here3")
    contrast_count = 1
    anchor_count = 1
    # tile mask
    mask_img = mask_img.repeat(anchor_count, contrast_count)
    mask_txt = mask_txt.repeat(anchor_count, contrast_count)
    mask_img2txt = mask_img2txt.repeat(anchor_count, contrast_count)
    mask_txt2img = mask_txt2img.repeat(anchor_count, contrast_count)
    #print(mask_img.shape,view1_feature.shape[1])
    #print(mask_img)
    logits_mask_1 = torch.scatter(
        torch.ones_like(mask_img),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    logits_mask_2 = torch.scatter(
        torch.ones_like(mask_txt),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    logits_mask_3 = torch.scatter(
        torch.ones_like(mask_img2txt),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    logits_mask_4 = torch.scatter(
        torch.ones_like(mask_txt2img),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    #print("here4")
    mask_img = mask_img * logits_mask_1
    mask_txt = mask_txt * logits_mask_2
    mask_img2txt = mask_img2txt * logits_mask_3
    mask_txt2img = mask_txt2img * logits_mask_4
    # compute log_prob
    exp_logits1 = torch.exp(logits1) * logits_mask_1
    exp_logits2 = torch.exp(logits2) * logits_mask_2
    exp_logits3 = torch.exp(logits3) * logits_mask_3
    exp_logits4 = torch.exp(logits4) * logits_mask_4
    #print("exp_logits1")
    #print(exp_logits1,exp_logits2)
    log_prob1 = logits1 - torch.log(exp_logits1.sum(1, keepdim=True))
    log_prob2 = logits2 - torch.log(exp_logits2.sum(1, keepdim=True))
    log_prob3 = logits3 - torch.log(exp_logits3.sum(1, keepdim=True))
    log_prob4 = logits4 - torch.log(exp_logits4.sum(1, keepdim=True))
    #print("log_prob3")
    #print(log_prob3,log_prob3)
    # compute mean of log-likelihood over positive
    mean_log_prob_pos1 = (mask_img * log_prob1).sum(1) / mask_img.sum(1)
    mean_log_prob_pos2 = (mask_txt * log_prob2).sum(1) / mask_txt.sum(1)
    mean_log_prob_pos3 = (mask_img2txt * log_prob3).sum(1) / mask_img2txt.sum(1)
    mean_log_prob_pos4 = (mask_txt2img * log_prob4).sum(1) / mask_txt2img.sum(1)
    #print("mean_log_prob_pos1")
    #print(mean_log_prob_pos1)
    # loss
    loss1 = - (temperature / base_temperature) * mean_log_prob_pos1
    loss2 = - (temperature / base_temperature) * mean_log_prob_pos2
    loss3 = - (temperature / base_temperature) * mean_log_prob_pos3
    loss4 = - (temperature / base_temperature) * mean_log_prob_pos4
    #print("loss1")
    #print(loss1)
    loss1 = loss1.view(1, batch_size).mean()
    loss2 = loss2.view(1, batch_size).mean()
    loss3 = loss3.view(1, batch_size).mean()
    loss4 = loss4.view(1, batch_size).mean()
    #print("loss1")
    #print(loss1)
    return 0.0001 * loss1 + 0.0001 * loss2 + loss3 + loss4
