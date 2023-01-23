import logging, torch
import argparse, os
from tqdm import tqdm
from datasets import load_metric
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from utils import * 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y  %H:%M:%S',
                    level=logging.INFO)

class SkyeGODEL:

    def __init__(self, pretrained_model_name_or_path, cache_dir, output_dir, max_length):
        self.max_length = max_length
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_gpu = torch.cuda.device_count()
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path = self.pretrained_model_name_or_path,
                                                cache_dir = self.cache_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = self.pretrained_model_name_or_path,
                                                cache_dir = self.cache_dir)

        self.model =  AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path = self.pretrained_model_name_or_path, 
                                        from_tf=bool(".ckpt" in self.pretrained_model_name_or_path),
                                        config=self.config,
                                        cache_dir=self.cache_dir).to(self.device)

    def train(self, training_file, dev_file, batch_size, 
             gradient_accumulation_steps, num_train_epochs, learning_rate,
             weight_decay=0.0, warmup_steps=0,
             max_grad_norm=1.0):
        
        train_dataset = SkyeDataset(training_file, self.tokenizer, self.max_length)
        train_dataloader = createDataLoader(train_dataset, batch_size)

        val_dataset = SkyeDataset(dev_file, self.tokenizer, self.max_length)
        val_dataloader = createDataLoader(val_dataset, batch_size, eval=True)


        total_steps = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
       

        no_decay = ["bias", "LayerNorm.Weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", batch_size)
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", total_steps)

        progress_bar = tqdm(range(total_steps))
        completed_steps = 0
        global_steps = 0
        logging_steps = 500
        save_steps = 600
        tr_loss, logging_loss = 0.0, 0.0
        val_bleu = 0.0

        for _ in range(num_train_epochs):
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                global_steps += 1
                
                inputs = {"input_ids": batch['input_ids'].to(self.device), 
                          "attention_mask": batch['attention_mask'].to(self.device), 
                          "labels": batch['labels'].to(self.device)}

                outputs = self.model(**inputs)
                
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                tr_loss += loss.item()

                loss.backward()

                if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1
                
                if completed_steps >= total_steps:
                    break

                if step % logging_steps == 0 and step > 1:
                    logger.info(f" EVAL ERROR: {(tr_loss - logging_loss) / float(logging_steps)}")
                    logging_loss = tr_loss
                    progress_bar.update(logging_steps)

                if global_steps % save_steps == 0 and global_steps > 0:
                    rouge, bleu = self.evaluate_data(val_dataloader)
                    if bleu['bleu'] > val_bleu:
                        val_bleu = bleu['bleu']

                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(self.output_dir, f"{checkpoint_prefix}-{global_steps}")
                        if not os.path.exists(output_dir):
                            os.mkdir(output_dir)
                        model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)  
                        model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)
                        logger.info(f"Saving model checkpoint to {output_dir}")
                    else:
                        logger.info('Bleu metric did not improve')

        total_loss = tr_loss / completed_steps
        return completed_steps, total_loss

    def evaluate_data(self, val_dataloader):
               
        metric_rouge = load_metric('rouge')
        metric_bleu = load_metric('bleu')

        decoded_preds_all = []
        predictions, references = [], []
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            self.model.eval()
            with torch.no_grad():
                inputs = {"input_ids": batch['input_ids'].to(self.device), 
                          "attention_mask": batch['attention_mask'].to(self.device), 
                          "labels": batch['labels'].to(self.device)}
                          
                outs = self.model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=self.max_length)
            labels = inputs['labels']

            decoded_preds = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            predictions.extend(decoded_preds)

            decoded_labels = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
            references.extend(decoded_labels)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            metric_rouge.add_batch(predictions=decoded_preds, references=decoded_labels)

            _decoded_preds = [i.split() for i in decoded_preds]
            _decoded_labels = [[i.split()] for i in decoded_labels]

            decoded_preds_all.extend(_decoded_preds)
            metric_bleu.add_batch(predictions=_decoded_preds, references=_decoded_labels)

        result = metric_rouge.compute(use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        logger.info(f"ROUGE: {result}")
        result_bleu = metric_bleu.compute()
        logger.info(f"BLEU: {result_bleu}")

        return result, result_bleu

def parse_args():
    parser = argparse.ArgumentParser(description="Godel Finetuning")

    parser.add_argument('--training-file', dest='training_file', 
                        help='Path to training file', default = 'path/to/train/data')
    
    parser.add_argument('--dev-file', dest='dev_file', 
                        help='Path to dev file', default = 'path/to/val/data')

    parser.add_argument('--language-model', help='GODEL HuggingFace Model', default = 'microsoft/GODEL-v1_1-large-seq2seq')
    
    parser.add_argument('--output-dir', dest='output_dir',
                        help='Path to where output is stored', default='training/pretrained')

    parser.add_argument('--cache-dir', dest='cache_dir',
                        help='Path to where cache is stored', default='training/cache')

    parser.add_argument('--epochs',
                        help='Number of epochs', default = 15)
    
    parser.add_argument('--batch-size',
                        help='Batch size', default = 1)

    parser.add_argument('--lr',
                        help='Learning rate', default = 5e-4)

    parser.add_argument('--gradient-accumulation',
                        help='', default = 16)
    
    parser.add_argument('--max-length',
                        help='Maximum length of sequence per sentences', default = 512)

    args = parser.parse_args()
    return args

def training(training_file, dev_file, pretrained_model_name_or_path, output_dir, epochs, cache_dir,
            batch_size, learning_rate, gradient_accumulation, max_length):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    classifier = SkyeGODEL(pretrained_model_name_or_path, cache_dir, output_dir, max_length)
    classifier.train(training_file, dev_file, batch_size, gradient_accumulation,
                    epochs, learning_rate)

def main():
    args = parse_args()
    training(training_file=args.training_file, dev_file=args.dev_file,
            pretrained_model_name_or_path=args.language_model, output_dir=args.output_dir,
            epochs=int(args.epochs), cache_dir=args.cache_dir, batch_size=args.batch_size,
            learning_rate=float(args.lr), gradient_accumulation=int(args.gradient_accumulation),
            max_length=args.max_length)

if __name__ == "__main__":
    main()
