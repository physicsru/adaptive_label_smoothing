def fx_calc_map_label(image, text, label, k = 0, dist_method='COS'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(numcases):
      if label[i] == label[order[j]]:
          res += [j]
          break
  rank = [1, 5, 10]
  acc = [sum([_ < r for _ in res]) / len(res) for r in rank]
  return acc[0]

def __getitem__(self, index):
        img = cv2.imread('/home/w00536717/hammer/DSCMR-master/data/flickr30k_images/flickr30k_images/'+ self.name[index])
        print('/home/w00536717/hammer/DSCMR-master/data/flickr30k_images/flickr30k_images/'+ self.name[index])
        img1 = cv2.rotate(img, cv2.ROTATE_180)
        img2 = cv2.GaussianBlur(img,(3,3),0)
        img3 = cv2.GaussianBlur(img,(1,1),0)
        img4 = cv2.GaussianBlur(img,(5,5),0)
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        img1 = cv2.resize(img1, self.dim, interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2, self.dim, interpolation = cv2.INTER_AREA)
        img3 = cv2.resize(img3, self.dim, interpolation = cv2.INTER_AREA)
        img4 = cv2.resize(img4, self.dim, interpolation = cv2.INTER_AREA)
        out = np.concatenate((img, img1), axis=0)
        out = np.concatenate((out, img2), axis=0)
        out = np.concatenate((out, img3), axis=0)
        out = np.concatenate((out, img4), axis=0)
        out = np.array([img,img1,img2,img3,img4][:],dtype=np.float32)#torch.FloatTensor([img,img1,img2,img3,img4])
        return torch.from_numpy(out).float(), self.label_img[index]
      
      
def calc_loss2(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta, device, temperature, base_temperature, batch_size):
    #print(labels_1.shape)
    view1_feature = normalize(view1_feature, 2)
    view2_feature = normalize(view2_feature, 2)
    #print("view1_feature, view2_feature")
    #print(view1_feature, view2_feature)
   
    img2txt_contrast = torch.div(
        torch.matmul(view1_feature, view2_feature.T),
        temperature)
    txt2img_contrast = torch.div(
        torch.matmul(view2_feature, view1_feature.T),
        temperature)
    #print("img2txt_contrast, txt2img_contrast")
    #print(img2txt_contrast, txt2img_contrast)
    logits_img2txt_max, _ = torch.max(img2txt_contrast, dim=1, keepdim=True)
    logits3 = img2txt_contrast - logits_img2txt_max.detach()
    logits_txt2img_max, _ = torch.max(txt2img_contrast, dim=1, keepdim=True)
    logits4 = txt2img_contrast - logits_txt2img_max.detach()
    #print("logits3, logits4")
    #print(logits3, logits4)
    batch_s = logits3.shape[0]
    
    exp_logits3 = torch.exp(logits3)
    exp_logits4 = torch.exp(logits4)
    
    #print("exp_logits1")
    #print(exp_logits3, exp_logits4)
#     log_prob3 = logits3 - torch.log(exp_logits3.sum(1, keepdim=True))
#     log_prob4 = logits4 - torch.log(exp_logits4.sum(1, keepdim=True))
    logits3 = logits3 - torch.log(exp_logits3.sum(1, keepdim=True))
    logits4 = logits4 - torch.log(exp_logits4.sum(1, keepdim=True))
    #print("log_prob3")
    #print(log_prob3,log_prob3)
    # compute mean of log-likelihood over positive
    del exp_logits3
    del exp_logits4
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    index1 = torch.unsqueeze(torch.from_numpy(np.arange(batch_s)), 0).to(device)
    index2 = torch.unsqueeze(torch.from_numpy(np.arange(batch_s)), 0).to(device)
    mask_img2txt = torch.eye(batch_s, m=batch_s, out=None).to(device)
    mask_txt2img = torch.eye(batch_s, m=batch_s, out=None).to(device)
    mean_log_prob_pos3 = logits3.gather(0,index1)
    mean_log_prob_pos4 = logits4.gather(0,index2)
    
    del logits3
    del logits4
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    #mean_log_prob_pos3 = log_prob3.gather(0,index1)
    #mean_log_prob_pos4 = log_prob4.gather(0,index2)
#     mean_log_prob_pos3 = (mask_img2txt * log_prob3).sum(1) / mask_img2txt.sum(1)
#     mean_log_prob_pos4 = (mask_txt2img * log_prob4).sum(1) / mask_txt2img.sum(1)
    #print("mean_log_prob_pos1")
    #print(mean_log_prob_pos1)
    # loss
    loss3 = - (temperature / base_temperature) * mean_log_prob_pos3
    loss4 = - (temperature / base_temperature) * mean_log_prob_pos4
    #print("loss1")
    #print(loss1)
    loss3 = loss3.view(1, batch_s).mean()
    loss4 = loss4.view(1, batch_s).mean()
    #print("loss3 loss4")
    #print(loss3, loss4)
    return (loss3 + loss4)# + (loss1 + loss2) * 0.5
