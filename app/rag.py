import os
import pandas as pd
import minsearch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv


prompt_template = """
You're a fitness insrtuctor. Answer the QUESTION based on the CONTEXT from our exercises database. 
Use only the facts from the CONTEXT when answering the QUESTION. Lastly, Answer should be translated to Traditional Chinese language.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
exercise_name: {exercise_name}
type_of_activity: {type_of_activity}
type_of_equipment: {type_of_equipment}
body_part: {body_part}
type: {type}
muscle_groups_activated: {muscle_groups_activated}
instructions: {instructions}
""".strip()

DATA_PATH = os.getenv("DATA_PATH", "../data/data.csv")
MODEL_PATH = os.getenv("MODEL_PATH","../model/Qwen3-1.7B")
class RAG:
#    def __init__(self, data_path= DATA_PATH, model_name = "Qwen/Qwen3-1.7B")->None:

    def __init__(self, data_path= DATA_PATH, model_path = MODEL_PATH)->None:

        load_dotenv()
        access_token = os.getenv("HF_ACCESS_TOKENS")
        self.ingess(data_path)
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )    

#        self.model = AutoModelForCausalLM.from_pretrained(
#            model_name,
#            torch_dtype="auto",
#            device_map="auto"
#        )    

    def ingess(self,data_path)->None:
        df =pd.read_csv(data_path)
        documents = df.to_dict(orient='records')
        self.index = minsearch.Index(
            text_fields= df.columns.tolist()[1:],
            keyword_fields= df.columns.tolist()[0]
        )
        self.index.fit(documents)

    def search(self, query):
        results=self.index.search(
            query='Push-up',
            filter_dict={},
            boost_dict={},
            num_results=10 )
        return results

    def build_prompt(self, query, search_results):
        context = ""
        
        for doc in search_results:
            context = context + entry_template.format(**doc) + "\n\n"

        prompt = prompt_template.format(question=query, context=context).strip()
        return prompt

    def summary(self, prompt):
        messages = [{"role": "user", "content": prompt}]    
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        index_content = self.tokenizer.decode([151667, 151668], skip_special_tokens=True).strip("\n")  
       
 
         
        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
    
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")    
        
        answer_data ={'think_logic': thinking_content,
                      'answer':   content, 
                      'input_pompt_token_size': len(model_inputs.input_ids[0]), 
                      'output_pompt_token_size': len(output_ids) 
                      }
        return answer_data
    
    def answer(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        #print(prompt)
        reply = self.summary(prompt)
        return reply

if __name__=='__main__':
    rag = RAG(data_path= '../data/data.csv', model_name = "Qwen/Qwen3-1.7B")
    question = 'Is the Lat Pulldown considered a strength training activity, and if so, why?'
    answer = rag.answer(question)
    print(answer)
