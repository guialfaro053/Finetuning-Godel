# Finetuning Godel

GODEL (Large-Scale Pre-Training for Goal-Directed Dialog) is an conversational agent that can be used for open-domain and task-oriented applications. It can be said that is an upgraded version of DialoGPT. Here is presented a method to finetune GODEL to respond generatively given a condition or context. The model is available on [HuggingFace🤗](https://huggingface.co/microsoft/GODEL-v1_1-large-seq2seq?text=Hey+my+name+is+Julien%21+How+are+you%3F).

--- 
The data was formatted as follows:

* **Instruction** - "Instruction: given a dialog context and a description of an AI assistant, you need to response emphatically."
* **Knowledge** - "My name is Mike and I am a virtual assistant that speaks Korean and English. I am 23 years old and I like partying."
* **Context** - [
            "Hello",
            "Hey! How are you doing today?",
            "I am okay. How old are you?",
            "I am 23 years old. How about you?"
            ...
            ]
