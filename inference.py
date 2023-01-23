import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
model = AutoModelForSeq2SeqLM.from_pretrained('training/pretrained/checkpoint-15000').to(device)

INSTRUCTIONS=''
KNOWLEDGE= ''' '''

def generate(instruction, context, knowledge):
    if knowledge != '':
        knowledge_tag = '[KNOWLEDGE] ' + knowledge
    context = ' EOS '.join(context)
    query = f"{instruction} [CONTEXT] {context} {knowledge_tag}"
    tokenized_text = tokenizer(f"{query}", return_tensors="pt")
    outputs = model.generate(tokenized_text['input_ids'].to(device), max_length=128, min_length=8, top_p=0.9, do_sample=True)
    msg = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return msg, knowledge


context = []
knowledge = KNOWLEDGE
instruction = INSTRUCTIONS

while(True):
    user_text = input('User >>> ')
    if(user_text == 'clear'):
        print('EPISODE DONE: TRUE. INITIALIZING NEW CONVERSATION')
        context = []
        knowledge = KNOWLEDGE['KNOWLEDGE_PROMPT']
        continue
        
    elif(user_text == 'exit'):
        print('Exiting...')
        break
        
    elif(user_text == 'knowledge'):
        print('current knowledge is : ', knowledge)
        user_text = input("Override (If you're not going to overwrite it, press Enter) >>> ")
        if(user_text != ''):
            knowledge = user_text
    else:
        if len(context) > 6:
            context = context[-4:]
        context.append(user_text)
        history_text = ' '.join(context[-2:])
        msg, knowledge = generate(instruction, context, knowledge)
        context.append(msg)
        print('GODEL >>> ' + msg)
