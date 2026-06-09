Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1218/1218 [00:01<00:00, 821.61 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 305/305 [00:00<00:00, 839.82 examples/s]
==Start model initialisation
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:12<00:00, 31.75it/s]
==Weights were successfully initialised
trainable params: 33,030,144 || all params: 4,055,498,240 || trainable%: 0.8145
==Старт обучения...
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
{'loss': '2.259', 'grad_norm': '0.7014', 'learning_rate': '9.706e-05', 'epoch': '0.06568'}                                                                                
{'loss': '1.777', 'grad_norm': '0.7872', 'learning_rate': '9.379e-05', 'epoch': '0.1314'}                                                                                 
{'loss': '1.695', 'grad_norm': '0.8574', 'learning_rate': '9.052e-05', 'epoch': '0.197'}                                                                                  
{'loss': '1.556', 'grad_norm': '0.845', 'learning_rate': '8.725e-05', 'epoch': '0.2627'}                                                                                  
{'loss': '1.487', 'grad_norm': '0.856', 'learning_rate': '8.399e-05', 'epoch': '0.3284'}                                                                                  
{'eval_loss': '1.476', 'eval_runtime': '469.3', 'eval_samples_per_second': '0.65', 'eval_steps_per_second': '0.65', 'epoch': '0.3284'}                                    
{'loss': '1.487', 'grad_norm': '0.9252', 'learning_rate': '8.072e-05', 'epoch': '0.3941'}                                                                                 
{'loss': '1.397', 'grad_norm': '0.9015', 'learning_rate': '7.745e-05', 'epoch': '0.4598'}                                                                                 
{'loss': '1.372', 'grad_norm': '1.085', 'learning_rate': '7.418e-05', 'epoch': '0.5255'}                                                                                  
{'loss': '1.347', 'grad_norm': '1.123', 'learning_rate': '7.092e-05', 'epoch': '0.5911'}                                                                                  
{'loss': '1.281', 'grad_norm': '1.311', 'learning_rate': '6.765e-05', 'epoch': '0.6568'}                                                                                  
{'eval_loss': '1.251', 'eval_runtime': '781.2', 'eval_samples_per_second': '0.39', 'eval_steps_per_second': '0.39', 'epoch': '0.6568'}                                    
{'loss': '1.291', 'grad_norm': '1.224', 'learning_rate': '6.438e-05', 'epoch': '0.7225'}                                                                                  
{'loss': '1.272', 'grad_norm': '1.36', 'learning_rate': '6.111e-05', 'epoch': '0.7882'}                                                                                   
{'loss': '1.21', 'grad_norm': '1.149', 'learning_rate': '5.784e-05', 'epoch': '0.8539'}                                                                                   
{'loss': '1.124', 'grad_norm': '1.321', 'learning_rate': '5.458e-05', 'epoch': '0.9195'}                                                                                  
{'loss': '1.116', 'grad_norm': '1.187', 'learning_rate': '5.131e-05', 'epoch': '0.9852'}                                                                                  
{'eval_loss': '1.097', 'eval_runtime': '687.9', 'eval_samples_per_second': '0.443', 'eval_steps_per_second': '0.443', 'epoch': '0.9852'}                                  
 50%|███████████████████████████████████████████████████████████████                                                               | 153/306 [4:26:32<6:48:12, 160.08s/it][timing] эпоха 1: 179.9 мин (10795 с)                                                                                                                                     
{'loss': '1.102', 'grad_norm': '1.089', 'learning_rate': '4.804e-05', 'epoch': '1.046'}                                                                                   
{'loss': '1.012', 'grad_norm': '1.165', 'learning_rate': '4.477e-05', 'epoch': '1.112'}                                                                                   
{'loss': '1.043', 'grad_norm': '1.411', 'learning_rate': '4.15e-05', 'epoch': '1.177'}                                                                                    
{'loss': '1.016', 'grad_norm': '1.386', 'learning_rate': '3.824e-05', 'epoch': '1.243'}                                                                                   
{'loss': '0.9065', 'grad_norm': '1.268', 'learning_rate': '3.497e-05', 'epoch': '1.309'}                                                                                  
{'eval_loss': '1.048', 'eval_runtime': '689.9', 'eval_samples_per_second': '0.442', 'eval_steps_per_second': '0.442', 'epoch': '1.309'}                                   
{'loss': '0.8856', 'grad_norm': '1.259', 'learning_rate': '3.17e-05', 'epoch': '1.374'}                                                                                   
{'loss': '1.094', 'grad_norm': '1.456', 'learning_rate': '2.843e-05', 'epoch': '1.44'}                                                                                    
{'loss': '1.016', 'grad_norm': '1.496', 'learning_rate': '2.516e-05', 'epoch': '1.506'}                                                                                   
{'loss': '0.9709', 'grad_norm': '1.227', 'learning_rate': '2.19e-05', 'epoch': '1.571'}                                                                                   
{'loss': '0.903', 'grad_norm': '1.314', 'learning_rate': '1.863e-05', 'epoch': '1.637'}                                                                                   
{'eval_loss': '1.025', 'eval_runtime': '682.1', 'eval_samples_per_second': '0.447', 'eval_steps_per_second': '0.447', 'epoch': '1.637'}                                   
{'loss': '0.9344', 'grad_norm': '1.252', 'learning_rate': '1.536e-05', 'epoch': '1.703'}                                                                                  
{'loss': '0.9379', 'grad_norm': '1.358', 'learning_rate': '1.209e-05', 'epoch': '1.768'}                                                                                  
{'loss': '0.9475', 'grad_norm': '1.407', 'learning_rate': '8.824e-06', 'epoch': '1.834'}                                                                                  
{'loss': '0.9733', 'grad_norm': '1.339', 'learning_rate': '5.556e-06', 'epoch': '1.9'}                                                                                    
{'loss': '0.9169', 'grad_norm': '1.335', 'learning_rate': '2.288e-06', 'epoch': '1.966'}                                                                                  
{'eval_loss': '1.015', 'eval_runtime': '628.8', 'eval_samples_per_second': '0.485', 'eval_steps_per_second': '0.485', 'epoch': '1.966'}                                   
{'eval_loss': '1.015', 'eval_runtime': '669.3', 'eval_samples_per_second': '0.456', 'eval_steps_per_second': '0.456', 'epoch': '2'}                                       
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 306/306 [17:16:30<00:00, 73.22s/it[timing] эпоха 2: 205.4 мин (12325 с)                                                                                                                                      
{'train_runtime': '6.219e+04', 'train_samples_per_second': '0.039', 'train_steps_per_second': '0.005', 'train_loss': '1.205', 'epoch': '2'}                               
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 306/306 [17:16:34<00:00, 73.22s/it]
=== ВРЕМЯ ОБУЧЕНИЯ ===
  Всего (train+eval+save): 385.4 мин (23125 с)
  Среднее на эпоху:        192.7 мин (11560 с)
  Всего optimizer-шагов:   306
  Среднее на шаг:          61.65 с (эффективный батч = 8 примеров)
  ~на микро-батч:          7.71 с (1 примера)
  ~на пример:              7.71 с
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 306/306 [17:16:34<00:00, 203.25s/it]
==Конец обучения, сохранение адаптера...