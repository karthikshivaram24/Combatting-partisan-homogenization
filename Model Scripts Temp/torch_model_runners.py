import torch
import torch.nn as nn
import tqdm
import gc
import numpy as np
from torch_seeder import seed_all, seed_worker
from torch_utils import load_pickle, timer, PolarLoss
from torch_models import AttentionST, AttentionMT
from torch_datasets import CPDatasetST, CPDatasetMT
from torch.utils.data import DataLoader
from torch_evaluators import evaluate_st, evaluate_mt
from attention_utils import get_attention_weights, aggregate_attW




def combined_loss_step(model,text_x,x2,y1,w1lw,w2lw,l2,loss_func,loss_func2,opt,glove,cuda_device):
    """
    """
    rec_losses = []
    word_losses = []
    total_losses = []
    
    for label_cw in range(2):
        # expand the context words 
        context_word_embed = None
        if glove:
            context_word_embed = x2[:,label_cw]
        if not glove:
            context_word_embed = x2[:,label_cw,:]
        if label_cw == 0:
            word_labels = torch.Tensor([1]*x2.size()[0])
        if label_cw == 1:
            word_labels = torch.Tensor([0]*x2.size()[0])
        model.train()
        text_x,context_word_embed,y1,word_labels = text_x.to(cuda_device),context_word_embed.to(cuda_device),y1.to(cuda_device),word_labels.to(cuda_device)
        opt.zero_grad() # reset all the gradient information
        y_pred, context_pred,attcvec = model(text_x, context_word_embed)
        rec_loss = loss_func(y_pred,y1)
        word_loss1 = loss_func(context_pred.squeeze(),word_labels)
        word_loss2 = loss_func2(model.att_con_vec)
        total_loss = (1 - (w1lw + w2lw))*rec_loss + (w1lw * word_loss1) + (w2lw * word_loss2) + model.word_weights.weight.norm(p=2) * l2
        total_loss.backward()
        opt.step()
        
        word_loss = word_loss1.item() + word_loss2.item()
        
        rec_losses.append(rec_loss.item())
        word_losses.append(word_loss)
        total_losses.append(total_loss.item())
    
    return np.mean(rec_losses), np.mean(word_losses), np.mean(total_losses)
    
    

def polar_loss_step(model,text_x,y1,word_loss_w,l2,loss_func,loss_func2,opt,cuda_device):
    """
    """
    text_x,y1 = text_x.to(cuda_device),y1.to(cuda_device) 
    y_pred, context_pred,attcvec = model(text_x, None)
    word_loss = loss_func2(model.att_con_vec)
    rec_loss = loss_func(y_pred,y1)
    total_loss = (1 - word_loss_w)*rec_loss + (word_loss_w * word_loss) + model.word_weights.weight.norm(p=2) * l2
    opt.zero_grad() 
    total_loss.backward(retain_graph=True)
    opt.step()
    
    return rec_loss.item(), word_loss.item(), total_loss.item()


def mt_step(model,text_x,x2,y1,word_loss_w,l2,loss_func,opt,glove,cuda_device):
    """
    """
    rec_losses = []
    word_losses = []
    total_losses = []
    
    for label_cw in range(2):
        # expand the context words 
        context_word_embed = None
        if glove:
            context_word_embed = x2[:,label_cw]
        if not glove:
            context_word_embed = x2[:,label_cw,:]
        if label_cw == 0:
            word_labels = torch.Tensor([1]*x2.size()[0])
        if label_cw == 1:
            word_labels = torch.Tensor([0]*x2.size()[0])
        model.train()
        text_x,context_word_embed,y1,word_labels = text_x.to(cuda_device),context_word_embed.to(cuda_device),y1.to(cuda_device),word_labels.to(cuda_device)
        opt.zero_grad() # reset all the gradient information
        y_pred, context_pred,attcvec = model(text_x, context_word_embed)
        rec_loss = loss_func(y_pred,y1)
        word_loss = loss_func(context_pred.squeeze(),word_labels)
        total_loss = (1 - word_loss_w)*rec_loss + (word_loss_w * word_loss) + model.word_weights.weight.norm(p=2) * l2
        total_loss.backward()
        opt.step()
        
        rec_losses.append(rec_loss.item())
        word_losses.append(word_loss.item())
        total_losses.append(total_loss.item())
    
        
    return np.mean(rec_losses), np.mean(word_losses), np.mean(total_losses)





@timer
def run_singleTask_exp(train, 
                        test,
                        val,
                        lr,
                        epochs=2,
                        batch_size=8,
                        dropout=0.1,
                        cuda_device=torch.device('cuda:1'),
                        num_workers=1,
                        with_attention=True,
                        cp=None,
                        avg_type="binary",
                        glove=False,
                        extreme=False,
                        weight_decay=0.0):
    """
    """
    seed_all(42)
    
    weights_matrix = None
    
    if glove:
        if not extreme:
            weights_matrix = load_pickle("Data_4_AttM/glove_mat.pkl")
            weights_matrix = torch.from_numpy(weights_matrix)
        
        if extreme:
            weights_matrix = load_pickle("Data_4_AttM/glove_mat_extreme.pkl")
            weights_matrix = torch.from_numpy(weights_matrix)
    
    model = AttentionST(embedding_size=768,with_attention=with_attention,dropout = dropout,glove=glove,weights_matrix=weights_matrix)
    model.to(cuda_device)
    loss_func = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    
    epoch_losses = {}
    total_losses = []
    total_losses_val = []
    
    glove_vocab_path = ""
    
    if glove:
        if extreme:
            glove_vocab_path = "Data_4_AttM/vocab_target_glove_extreme.pkl"
        
        if not extreme:
            glove_vocab_path = "Data_4_AttM/vocab_target_glove.pkl"
            
    bert_hdf_file_path = None
    
    if extreme:
        bert_hdf_file_path = "/media/karthikshivaram/SABER_4TB/attm_bert_embeddings/token_bert_12_embeds_attm_extreme.hdf5"
    
    if not extreme:
        bert_hdf_file_path = "/media/karthikshivaram/SABER_4TB/attm_bert_embeddings/token_bert_12_embeds_attm.hdf5"
    
    train_dataset = CPDatasetST(train,hdf_file_path=bert_hdf_file_path,glove_vocab_path=glove_vocab_path,glove=glove)
    test_dataset = CPDatasetST(test,hdf_file_path=bert_hdf_file_path,glove_vocab_path=glove_vocab_path,glove=glove)
    val_dataset = CPDatasetST(val,hdf_file_path=bert_hdf_file_path,glove_vocab_path=glove_vocab_path,glove=glove)
    
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,worker_init_fn=seed_worker)
    test_dataloader = DataLoader(test_dataset,batch_size=100,num_workers=num_workers,shuffle=True,worker_init_fn=seed_worker)
    val_dataloader = DataLoader(val_dataset,batch_size=100,num_workers=num_workers,shuffle=True,worker_init_fn=seed_worker)
    
    for epoch in tqdm(range(epochs),total=epochs):
        batch_losses =  []
        # Training
        for batch_num, (x1,y1,t1,wc) in enumerate(train_dataloader):
            
            model.train()

            x1,y1 = x1.to(cuda_device),y1.to(cuda_device)
            
            opt.zero_grad() # reset all the gradient information
    
            y_pred = model(x1)
            
            if batch_size == 1:
                y1 = y1.view(1)
            
            if batch_size > 1:
                y1 = y1.squeeze()
            
            total_loss = loss_func(y_pred,y1.squeeze()) 
            
            total_loss.backward()
            
            opt.step()
            
            batch_losses.append(total_loss.item())
        
        # Validation
        batch_losses_val =  []
        for bn_v, (xv1,yv1,tv1,wcv) in enumerate(val_dataloader):
            
            model.eval()
            with torch.no_grad():
                xv1,yv1 = xv1.to(cuda_device),yv1.to(cuda_device)

                y_pred = model(xv1)
                
                if batch_size == 1:
                    yv1 = yv1.view(1)
            
                if batch_size > 1:
                    yv1 = yv1.squeeze()

                total_loss_val = loss_func(y_pred,yv1) 
                batch_losses_val.append(total_loss_val.item())
        
        avg_batch_loss = np.mean(batch_losses)
        avg_batch_loss_val = np.mean(batch_losses_val)
        total_losses.append(avg_batch_loss)
        total_losses_val.append(avg_batch_loss_val)

        
    epoch_losses["total_loss"] = total_losses
    epoch_losses["total_loss_val"] = total_losses_val
        
    scores_train = evaluate_st(model,train_dataloader,device=cuda_device,avg_type=avg_type)
    scores_val = evaluate_st(model,val_dataloader,device=cuda_device,avg_type=avg_type)
    scores_test = evaluate_st(model,test_dataloader,device=cuda_device,avg_type=avg_type)
    
    if with_attention:
        single_att_w_n = get_attention_weights(model,test,single=True)
        single_att_w_agg = aggregate_attW(single_att_w_n)
        ranks = sorted(list(single_att_w_agg.items()),key=lambda x:x[1],reverse=True)[:30]
        print("\nTop Ranked Words with Highest Attention Weights :\n")
        for r in ranks:
            print(r)
    
    x1 = None
    y1 = None
    total_loss = None
    opt = None
    

    del x1
    del y1

    gc.collect()
    torch.cuda.empty_cache()
    
    return model, epoch_losses, scores_train, scores_test, scores_val



@timer
def run_multiTask_exp(train,test,val,cw_embed_train,cw_embed_test,cw_embed_val,lr,word_loss_w,epochs=2,batch_size=8,dropout=0.1,cuda_device=torch.device('cuda:1'),
                num_workers=1,l2=0.05,avg_type="binary",glove=False,extreme=False,loss2=False,loss3=False,w1lw=0.5,w2lw=0.4,bad_term_embeds=None):
    """
    """
    seed_all(42)
    
    weights_matrix = None
    
    if glove:
        if not extreme:
            weights_matrix = load_pickle("Data_4_AttM/glove_mat.pkl")
            weights_matrix = torch.from_numpy(weights_matrix)
        
        if extreme:
            weights_matrix = load_pickle("Data_4_AttM/glove_mat_extreme.pkl")
            weights_matrix = torch.from_numpy(weights_matrix)
    
    if loss2 and bad_term_embeds != None:
        bad_term_embeds = bad_term_embeds.to(cuda_device)
    
    if loss3 and bad_term_embeds != None:
        bad_term_embeds = bad_term_embeds.to(cuda_device)
 
    
    model = AttentionMT(embedding_size=768,dropout=dropout,glove=glove,weights_matrix=weights_matrix,bad_embeds = bad_term_embeds,use_loss2=loss2)
    model.to(cuda_device)
    loss_func = nn.BCELoss()
    loss_func2 = None
    if loss2 or loss3:
        loss_func2 = PolarLoss(bad_term_embeds=bad_term_embeds)
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    
    epoch_losses = {}
    total_losses = []
    word_losses = []
    rs_losses = []
    total_losses_val = []
    word_losses_val = []
    rs_losses_val = []
    
    glove_vocab_path = ""
    
    if glove:
        if extreme:
            glove_vocab_path = "Data_4_AttM/vocab_target_glove_extreme.pkl"
        
        if not extreme:
            glove_vocab_path = "Data_4_AttM/vocab_target_glove.pkl"
            
    bert_hdf_file_path = None
    
    if extreme:
        bert_hdf_file_path = "/media/karthikshivaram/SABER_4TB/attm_bert_embeddings/token_bert_12_embeds_attm_extreme.hdf5"
    
    if not extreme:
        bert_hdf_file_path = "/media/karthikshivaram/SABER_4TB/attm_bert_embeddings/token_bert_12_embeds_attm.hdf5"
    
    train_dataset = CPDatasetMT(train,cw_embed_train,hdf_file_path=bert_hdf_file_path,glove_vocab_path=glove_vocab_path,glove=glove)
    test_dataset = CPDatasetMT(test,cw_embed_test,hdf_file_path=bert_hdf_file_path,glove_vocab_path=glove_vocab_path,glove=glove)
    val_dataset = CPDatasetMT(val,cw_embed_val,hdf_file_path=bert_hdf_file_path,glove_vocab_path=glove_vocab_path,glove=glove)
    
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers, shuffle=True,worker_init_fn=seed_worker)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,worker_init_fn=seed_worker)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True,worker_init_fn=seed_worker)
    
    model.train()
    for epoch in tqdm(range(epochs),total=epochs):
        
        batch_losses_total = []
        batch_losses_rec = []
        batch_losses_word = []
        
        for batch_num, (x1, x2, y1, t1, wc) in enumerate(train_dataloader):
            
            rec_loss = None
            word_loss = None
            total_loss = None
            
            if loss2:
                rec_loss, word_loss, total_loss = polar_loss_step(model,x1,y1,
                                                                  word_loss_w,l2,
                                                                  loss_func,loss_func2,
                                                                  opt,cuda_device)
            
            elif loss3:
                rec_loss, word_loss, total_loss = combined_loss_step(model,x1,x2,y1,
                                                                     w1lw,w2lw,l2,
                                                                     loss_func,loss_func2,
                                                                     opt,glove,cuda_device)
            
            elif not loss2 and not loss3:
                rec_loss, word_loss, total_loss = mt_step(model,x1,x2,y1,
                                                          word_loss_w,l2,
                                                          loss_func,opt,
                                                          glove,cuda_device)

                    
            batch_losses_rec.append(rec_loss)
            batch_losses_word.append(word_loss)
            batch_losses_total.append(total_loss)
                
            
        avg_batch_losses_total = np.mean(batch_losses_total)
        avg_batch_losses_rec = np.mean(batch_losses_rec)
        avg_batch_losses_word = np.mean(batch_losses_word)
        
        
        batch_losses_total_val = []
        batch_losses_rec_val = []
        batch_losses_word_val = []

        for bn_v, (x1v, x2v, y1v, t1v, wcv) in enumerate(val_dataloader):

            model.eval()
            with torch.no_grad():
                
                for label_cw in range(2):
                    
                    context_word_embed = None
                    if glove:
                        context_word_embed = x2v[:,label_cw]

                    if not glove:
                        context_word_embed = x2v[:,label_cw,:]
                        
                    if label_cw == 0:
                        word_labels = torch.Tensor([1]*x1v.size()[0])

                    if label_cw == 1:
                        word_labels = torch.Tensor([0]*x1v.size()[0])
            
                    x1v,context_word_embed,y1v,word_labels = x1v.to(cuda_device),context_word_embed.to(cuda_device),y1v.to(cuda_device),word_labels.to(cuda_device)

                    y_pred, context_pred,attcvec = model(x1v, context_word_embed)

                    if batch_size == 1:
                        y_pred = y_pred.view(1,1)
                        context_pred = context_pred.view(1,1)

                    rec_loss = loss_func(y_pred,y1v)
                    word_loss = None
                    total_loss = None
                    
                    if not loss2 and not loss3:
                        if context_pred.squeeze().size() != word_labels.size():
                            word_loss = loss_func(context_pred,word_labels.view(1,1))
                        else:  
                            word_loss = loss_func(context_pred.squeeze(),word_labels)
                        
                        total_loss = (1 - word_loss_w)*rec_loss + (word_loss_w * word_loss) + model.word_weights.weight.norm(p=2) * l2
                    
                    if loss2:
                        
                        word_loss = loss_func2(attcvec)

                        total_loss = (1 - word_loss_w)*rec_loss + (word_loss_w * word_loss)
                    
                    if loss3:
                        word_loss1 = loss_func(context_pred.squeeze(),word_labels)
                        word_loss2 = loss_func2(model.att_con_vec)
                        total_loss = (1 - (w1lw + w2lw))*rec_loss + (w1lw * word_loss1) + (w2lw * word_loss2)
                        word_loss = word_loss1 + word_loss2

                    batch_losses_total_val.append(rec_loss.item())
                    batch_losses_word_val.append(word_loss.item())
                    batch_losses_rec_val.append(total_loss.item())
        
        
        
        avg_batch_losses_total_val =np.mean(batch_losses_total_val)
        avg_batch_losses_rec_val = np.mean(batch_losses_rec_val)
        avg_batch_losses_word_val =np.mean(batch_losses_word_val)

        total_losses.append(avg_batch_losses_total)
        word_losses.append(avg_batch_losses_word)
        rs_losses.append(avg_batch_losses_rec)
        total_losses_val.append(avg_batch_losses_total_val)
        word_losses_val.append(avg_batch_losses_word_val)
        rs_losses_val.append(avg_batch_losses_rec_val)
    
    print("\n**** Last Loss vals *****")
    print("Rec loss : %s" %str(rec_loss.item()))
    print("Word loss : %s" %str(word_loss.item()))
    print("Total loss : %s" %str(total_loss.item()))
        
    epoch_losses["rs_loss"] = rs_losses
    epoch_losses["word_loss"] = word_losses
    epoch_losses["total_loss"] = total_losses
    epoch_losses["rs_loss_val"] = rs_losses_val
    epoch_losses["word_loss_val"] = word_losses_val
    epoch_losses["total_loss_val"] = total_losses_val

    scores_train = evaluate_mt(model,train_dataloader,device=cuda_device,avg_type=avg_type)
    scores_val = evaluate_mt(model,val_dataloader,device=cuda_device,avg_type=avg_type)
    scores_test = evaluate_mt(model,test_dataloader,device=cuda_device,avg_type=avg_type)
    
    multi_att_w_n = get_attention_weights(model,test,single=False)
    multi_att_w_agg = aggregate_attW(multi_att_w_n)
    ranks = sorted(list(multi_att_w_agg.items()),key=lambda x:x[1],reverse=True)[:30]
    print("\nTop Ranked Words with Highest Attention Weights :\n")
    for r in ranks:
        print(r)
    
    x1 = None
    x2 = None
    y1 = None
    y2 = None
    opt = None
    

    del x1
    del x2
    del y1
    del y2

    gc.collect()
    torch.cuda.empty_cache()
    
    return model, epoch_losses, scores_train, scores_test, scores_val