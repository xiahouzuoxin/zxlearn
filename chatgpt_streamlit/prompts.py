from datetime import datetime

SYSTEM_PROMPT = f'''
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2021-09
Current date: {datetime.now().strftime('%Y-%m-%d')}
'''

KNOWLEDGE_PROMPT = f'''
You can use the following knowledge to chat with me if you need:
```
<<KNOWLEDGE>>
```
'''

def construct_user_prompt(prompt, vector_retrieval=None):
    if vector_retrieval is None:
        return prompt
    
    knowledge = vector_retrieval.query([prompt], limit=2)
    knowledge_prompt = KNOWLEDGE_PROMPT.replace('<<KNOWLEDGE>>', '\n- '.join(knowledge['documents'][0]))
    final_prompt = f'''Question: {prompt}?\n{knowledge_prompt}'''
    # print(final_prompt)
    return final_prompt
