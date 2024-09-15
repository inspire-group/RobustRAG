from .helper import load_json,clean_str
import logging

logger = logging.getLogger('RRAG-main')

class DataUtils: # base class for dataset
    def __init__(self,data_path,top_k):
        self.data_path = data_path
        logger.info(f'Loading data from {data_path}......')
        self.data = load_json(data_path)
        logger.info(f'Total samples: {len(self.data)}')
        self.top_k = top_k

    def process_data_item(self,data_item,top_k=None,include_title=True,add_expanded_answer=True):
        # extract necessary information from raw json file
        top_k = self.top_k if top_k is None else top_k
        question = data_item['question']
        context = data_item['context'][:top_k] # retrieved passages
        answer = data_item['correct answer']
        if add_expanded_answer: # using additional equivalent answers written by GPT (GPT can make mistakes occasionally)
            answer += data_item['expanded answer']
        
        incorrect_answer = data_item.get('incorrect answer',[]) # used for running targeted attack
        incorrect_context = data_item.get('incorrect_context',[]) # used for running Poison attack

        if include_title: # include webpage title or not
            topk_content = [x['title'] + '\n' + x['text'] for x in context if ('text' in x) and ('title' in x)] 
        else:
            topk_content = [x['text'] for x in context if 'text' in x]

        return {
            'question':question,
            'answer':answer,
            'topk_content':topk_content,
            'incorrect_answer':incorrect_answer,
            'incorrect_context':incorrect_context
        }

    def wrap_prompt(self): 
        raise NotImplementedError

    def eval_response(self,response,data_item): # eval the correctness of QA
        answer = data_item['answer']
        response = clean_str(response)
        # if any answer is in the response, return true
        for ans in answer:
            if clean_str(ans) in response:
                logger.debug('correct!')
                return True 
        return False

    def eval_response_asr(self,response,data_item): # eval if the targeted attack succeeds
        incorrect_answer = data_item['incorrect_answer']
        response = clean_str(response)
        # if any answer is in the response, return true
        if clean_str(incorrect_answer) in response:
            logger.debug(f'Incorrect answer:\n{incorrect_answer}')
            logger.debug('Attack successed!')
            return True 
        return False

class RealtimeQA(DataUtils):
    # add supports for multiple-choice QA
    def __init__(self,data_path,top_k,as_multi_choices=True):
        super().__init__(data_path,top_k)
        self.as_multi_choices = as_multi_choices

    def process_data_item(self,data_item,top_k=None):
        ret = super().process_data_item(data_item,top_k)
        if self.as_multi_choices:
            choices = data_item['choices']
            choices_answer = data_item.get('choices answer')
            ret.update({
                'choices':choices,
                'choices_answer':choices_answer
                })
        return ret 

    def eval_response(self,response,data_item):
        if not self.as_multi_choices:
            return super().eval_response(response,data_item)
        else: # multiple choice questions
            mapping = {'0':'a.','1':'b.','2':'c.','3':'d.'}
            answer = data_item['choices_answer']
            answer = mapping[answer]
            if "Answer:" in response:
                response = response[response.index("Answer:") + len("Answer:"):]
            corr =  clean_str(response).startswith(answer)
            if corr:
              logger.debug('correct!')
            return corr

class Biogen(DataUtils):
    def __init__(self,data_path,top_k):
        super().__init__(data_path,top_k)

    def process_data_item(self,data_item,top_k=None):
        ret = super().process_data_item(data_item,top_k)
        ret.update({'long_gen':data_item.get('long_gen',False)}) # add a tag for long-form generation
        return ret 


NQ = RealtimeQA 
def load_data(dataset_name,top_k,data_path=None):
    data_path = data_path if data_path else f'data/{dataset_name.split("-")[0]}.json'
    if dataset_name == 'realtimeqa-mc':
        return RealtimeQA(data_path,top_k,as_multi_choices=True)
    elif dataset_name == 'realtimeqa':
        return RealtimeQA(data_path,top_k,as_multi_choices=False)
    elif dataset_name == 'open_nq':
        return NQ(data_path,top_k,as_multi_choices=False)
    elif dataset_name == 'biogen':
        return Biogen(data_path,top_k)
    else: # shouldn't happen
        if 'mc' in dataset_name:
            return RealtimeQA(data_path,top_k,as_multi_choices=True)
        else:
            return RealtimeQA(data_path,top_k,as_multi_choices=False)
