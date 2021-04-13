

# mocking baseline 2

@timer
def run_cp(df,cp,doc_2_cluster_map,learning_rate,context_pred_loss_weight,num_epochs,batch_size,increments=10):
    """
    """
    train,test = get_train_test_attm(df,cp,doc_2_cluster_map)
        
    model, epoch_losses, opt, loss_func = train_model(train,
                                                      learning_rate=learning_rate,
                                                      context_pred_loss_weight=context_pred_loss_weight,
                                                      num_epochs=num_epochs,
                                                      batch_size=batch_size)
        
    inc_losses_test,precisions,recalls = incremental_train(model=model,
                                      data=test,
                                      opt=opt,
                                      loss_func=loss_func,
                                      context_pred_loss_weight=context_pred_loss_weight,
                                      increments=increments)
    
    return epoch_losses,inc_losses_test, precisions, recalls

@timer
def run_all_cps(df,cps,doc_2_cluster_map,learning_rate=0.01,context_pred_loss_weight=0.5,num_epochs=2,batch_size=1,increments=10):
    """
    """
    cp_scores = defaultdict(lambda : defaultdict(list))
    epoch_train_loss_cps = defaultdict()
    inc_test_loss_cps = defaultdict()
    
    for cp in cps:
        print("cluster_pair : %s" %str(cp))
        
        epoch_losses,inc_losses_test,precisions, recalls = run_cp(df,cp,doc_2_cluster_map,learning_rate,context_pred_loss_weight,num_epochs,batch_size,increments=increments)
        epoch_train_loss_cps[cp]= epoch_losses
        inc_test_loss_cps[cp] = inc_losses_test
        cp_scores[cp]["precision"] = precisions
        cp_scores[cp]["recall"] = recalls
    
    return epoch_train_loss_cps, inc_test_loss_cps, cp_scores

@timer
def train_model(data,learning_rate=0.01,context_pred_loss_weight=0.5,num_epochs=2,batch_size=1):
    """
    """
    cuda1 = torch.device('cuda:1')
    model = AttentionMT(embedding_size=768,verbose=True)
    model.to(cuda1)
    loss_func = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    bert_tokenizer = load_tokenizer()
    
    epoch_losses = defaultdict(lambda : defaultdict(list))
    
    for epoch in range(num_epochs):
        
        batch_nums = data.shape[0]/batch_size
        batch_total_loss = []
        batch_word_loss = []
        batch_rs_loss = []
        
        articles = data["text"].tolist()
        context_words = data["context_word"].tolist()
        class_labels = data["class_label"].tolist()
        word_labels = data["word_label"].tolist()
        which_cluster = data["which_cluster"].tolist()
        
        for batch_num, (article_batch, context_word_batch, class_label_batch, word_label_batch, which_cluster_batch) in enumerate(batch_text_gen(articles,
                                                                                                                                                 context_words,
                                                                                                                                                 class_labels,
                                                                                                                                                 word_labels,
                                                                                                                                                 which_cluster,
                                                                                                                                                 batch_size=batch_size)):
            
            bert_tokenized_words, bert_tokenized_word_to_predict, rec_labels, word_labels = tokenize_4bert_batch(article_batch,
                                                                                                                 context_word_batch, 
                                                                                                                 class_label_batch, 
                                                                                                                 word_label_batch, tokenizer=bert_tokenizer)
            opt.zero_grad() # reset all the gradient information
    
            y_pred, context_pred, attention_vector = model(bert_tokenized_words, bert_tokenized_word_to_predict)
            
            rec_loss = loss_func(y_pred,rec_labels)
            word_loss = loss_func(context_pred,word_labels)
            
            total_loss = rec_loss + (context_pred_loss_weight * word_loss)
            
            total_loss.backward()
            
            opt.step()
            
            batch_rs_loss.append(rec_loss.item())
            batch_word_loss.append(word_loss.item())
            batch_total_loss.append(total_loss)
            
            if batch_num % 100 == 0 and batch_num >=100:
                print("Epoch : %s | Batch : %s | Total Loss : %s | Rec Loss : %s | Word Loss : %s" % (str(epoch),str(batch_num),str(total_loss.item()),str(rec_loss.item()),str(word_loss.item())))
            
            
        epoch_losses[epoch]["rs_loss"].append(batch_rs_loss)
        epoch_losses[epoch]["word_loss"].append(batch_word_loss)
        epoch_losses[epoch]["total_loss"].append(batch_total_loss)
    
    return model, epoch_losses, opt, loss_func

@timer
def incremental_train(model,data,opt,loss_func,context_pred_loss_weight=0.5,increments=100,pred_batch_size=100):
    """
    * we train with one sample
    * recommend from the rest of the candidate pool
    * record recommendation
    * remove recommendation from candidate pool
    * use this recommendation to train again
    * repeat for n increments
    """
    
    articles = data["text"].tolist()
    context_words = data["context_word"].tolist()
    class_labels = data["class_label"].tolist()
    word_labels = data["word_label"].tolist()
        
    bert_tokenizer = load_tokenizer()
    
    bert_tokenized_words, bert_tokenized_word_to_predict, rec_labels, word_labels = tokenize_4bert_batch(articles,
                                                                                                         context_words, 
                                                                                                         class_labels, 
                                                                                                         word_labels, 
                                                                                                         tokenizer=bert_tokenizer)
    candidate_x1 = bert_tokenized_words
    candidate_x2 = bert_tokenized_word_to_predict
    candidate_y1 = rec_labels
    candidate_y2 = word_labels
    
    y1_preds = []
    y2_preds = []
    all_relevant = sum(class_labels)
    precisions_k = []
    recalls_k = []
    
    ttl = []
    rsl = []
    cwl = []
    
    for i in range(increments):
        print("Iter : %s" %str(i))
        predicted_probas_y1 = []
        predicted_probas_y2 = []
        attention_vectors = []
        with torch.no_grad(): # eval or recommendation mode
#             for i in range(candidate_x1.size(0)):
#                 pp_y1, pp_y2, at = model(candidate_x1[i,:].unsqueeze(0),candidate_x2[i,:].unsqueeze(0))
#                 predicted_probas_y1.append(pp_y1.unsqueeze(0))
#                 predicted_probas_y2.append(pp_y2.unsqueeze(0))
#                 attention_vectors.append(at)
            
            for batch_num, (cp_x1_batch, cp_x2_batch, y1_batch, y2_batch) in enumerate(batch_gen_pred(candidate_x1,candidate_x2,candidate_y1,candidate_y2,batch_size=100)):
                pp_y1, pp_y2, at = model(cp_x1_batch,cp_x2_batch)
                predicted_probas_y1.append(pp_y1)
                predicted_probas_y2.append(pp_y2)
                attention_vectors.append(at)
        
#         predicted_probas_y1 = torch.stack(predicted_probas_y1)
#         predicted_probas_y2 = torch.stack(predicted_probas_y2)
        
        predicted_probas_y1 = torch.cat(predicted_probas_y1,dim=0)
        predicted_probas_y2 = torch.cat(predicted_probas_y2,dim=0)
        
#         attention_vectors = torch.cat(attention_vectors,dim=0)
        
        # get argmax of predicted_probas_y1
        rec_item_ind = torch.topk(predicted_probas_y1, k=1, dim=0)[1].squeeze()
        rec_item_y1_label = candidate_y1[rec_item_ind.item()]
        
        y1_preds.append(rec_item_y1_label.item())
        # delete this index from tensors (not trivial as inn numpy)
        # need to subsample the tensor instead of deletion 
        
        rec_x1 = candidate_x1[rec_item_ind,:]
        word_x2 = candidate_x2[rec_item_ind,:]
        rec_label = rec_labels[rec_item_ind]
        word_label = word_labels[rec_item_ind]
        
        candidate_x1 = candidate_x1[torch.arange(candidate_x1.size(0)) != rec_item_ind.cpu(),:]
        candidate_x2 = candidate_x2[torch.arange(candidate_x2.size(0)) != rec_item_ind.cpu(),:]
        candidate_y1 = candidate_y1[torch.arange(candidate_y1.size(0)) != rec_item_ind.cpu()]
        candidate_y2 = candidate_y2[torch.arange(candidate_y2.size(0)) != rec_item_ind.cpu()]
        
        _,_,_,rec_loss, word_loss, total_loss = partial_train_step(rec_x1.unsqueeze(0),word_x2.unsqueeze(0),rec_label,word_label.unsqueeze(0),model,opt,loss_func,context_pred_loss_weight=context_pred_loss_weight)
        
        # Recall @K
        recall = Counter(y1_preds[:i+1])[1.0]/all_relevant
        # Precision @K
        precision = (Counter(y1_preds[:i+1])[1.0])/len(y1_preds[:i+1])
        
        ttl.append(total_loss)
        rsl.append(rec_loss)
        cwl.append(word_loss)
        precisions_k.append(precision)
        recalls_k.append(recall)
    
    # calculate metrics 
    losses = defaultdict()
    losses["total_loss"] = ttl
    losses["rec_loss"] = rsl
    losses["word_loss"] = cwl
    
    return losses, precisions_k, recalls_k