# adapted from https://github.com/hyintell/RetrievalQA/blob/main/google_search.py

import serpapi

serpapi_api_key = "" # unfortunately, we are currently using this paid third-party API from https://serpapi.com/
# Google itself provides APIs free for a number of queries per day https://developers.google.com/custom-search/v1/overview
import json

def call_search_engine(query):
    params = {
        "q": query,
        "engine": "google", # Set parameter to google to use the Google API engine
        # "location": "California, United States",
        "hl": "en", # Parameter defines the language to use for the Google search.
        "gl": "us", # Parameter defines the country to use for the Google search.
        "google_domain": "google.com",  # Parameter defines the Google domain to use.
        "api_key": serpapi_api_key,
        "num": 50
    }
    results = serpapi.search(**params)
    return results


def parse_google_research_results(results, retrieved_num=5):
    
    retrieved_docs = []
    # Answer box has higher priority
    #if 'answer_box' in results: 
    #    parsed_item = {}
    #    answer_box = results["answer_box"]
    #    if 'link' in answer_box:# avoid the case where the answer box only has answer but no linked webpage
    #        parsed_item["link"] = item['link']            
    #        if "title" in answer_box:
    #            parsed_item["title"] = answer_box["title"]
    #        if "snippet" in answer_box:
    #            parsed_item["text"] = answer_box["snippet"]
    #    retrieved_docs.append(parsed_item)

    if "organic_results" in results: # main search results
        items = results['organic_results']

        if len(items) < retrieved_num:
            retrieved_num = len(items)

        for idx in list(range(len(items)))[:retrieved_num]:
            item = items[idx]
            parsed_item = {}
            if "title" in item:
                parsed_item["title"] = item['title']
            if "snippet" in item:
                parsed_item["text"] = item['snippet'] # for QA tasks, we only use snippets
            if "link" in item:
                parsed_item["link"] = item['link']
                
            retrieved_docs.append(parsed_item)
        
    return retrieved_docs


def main():
    dataset = 'realtimeqa'
    input_file_path = f"{dataset}.json"
    with open(input_file_path, "r") as f:
        data = json.load(f)
    
    query_count = 0
    raw_results = []
    for item in data:
        query_count += 1

        query = item["question"]
        
        results = call_search_engine(query) # call api

        raw_results.append(dict(results)) # save raw results

        retrieved_docs = parse_google_research_results(results, retrieved_num=20) # parse results

        item["context"] = retrieved_docs 

    print(f"total query times: {query_count}")

    with open(f"{dataset}_gg.json", 'w') as g:
        g.write(json.dumps(data, indent=4))
    with open(f"{dataset}_gg_raw.json", 'w') as g:
        g.write(json.dumps(raw_results, indent=4))        




if __name__ == "__main__":
    main()

