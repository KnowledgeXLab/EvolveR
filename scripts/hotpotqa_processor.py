import json
import re
import os
import time
import random
from tqdm import tqdm
from openai import OpenAI

prompt_teacher = """
You are a top-notch intelligent reasoning expert, adept at restoring solution paths from given answers and documents in reverse. Your task is to simulate a full reasoning trajectory for answering the question below, based on the provided documents and answer. You must reason step-by-step as if you do not yet know the final answer, even though it is given for supervision. 

In <think> blocks, do not reference or confirm the final answer directly. Instead, reason like a human—understand the task, recall prior knowledge, evaluate the need for experience or external information, and gradually infer the answer.

The reasoning trajectory must follow the **exact format below**. If the retrieved **experience alone is sufficient to answer the question**, you may skip the <search_knowledge> and <information> steps.

**Output Format:**

<think> ... </think>

<search_experience>

- Retrieve 2–3 relevant abstract experience principles, using structured triple format.

- For each principle, add a short description of its purpose.

</search_experience>

<think> Explain what you plan to do after retrieving experience. Decide whether you still need to retrieve knowledge. </think>

[IF experience is enough:]

<think>

- List the principles you are applying, include their triple form and description.

- Explain briefly how each principle contributes to your reasoning.

- Continue with reasoning based on these principles and conclude with your final judgment.

</think>

<answer>...</answer>

[ELSE:]

<search_knowledge>

- Generate one or more natural language search queries that would help retrieve the provided documents.

</search_knowledge>

<information>

{{relevant_document}}

</information>

<think> Reflect on retrieved information. </think>

<think>

- List the principles you are applying, include their triple form and description.

- Explain how each principle guides the reasoning process using the retrieved information.

- Summarize your reasoning path and justify the answer.

</think>

<answer>...</answer>

### Please use the following inputs:

[Query]

{{query}}

[Relevant Documents]

{{relevant_document}}

[Answer]

{{answer}}

Please begin generating the reasoning trajectory:
"""

# System prompt (for multi-turn dialogue format)
prompt_multi = "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information or get new experience principles. After reasoning, you can search for past experiences by <search_experience> query </search_experience> to get relevant past experience principles (may be guilding or warning principles) and it will return the top searched results between <experience> and </experience>. You can use these principles which you think is helpful to help you answer the question. If you find you lack some knowledge, you can call a search engine by <search_knowledge> query </search_knowledge> and it will return the top searched results between <information> and </information>. You can search knowledge and experience as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>"

# ================= Configuration =================
# API Configuration
API_KEY = "sk-xxxxxx"
BASE_URL = "xxxx"

# Data Path Configuration
DATASET_DIR = './data/nq_hotpotqa_train/'  # you can find in https://huggingface.co/datasets/Edaizi/EvolveR-NQ-HotpotQA/tree/main
HOTPOTQA_DIR = './data/nq_hotpotqa_train/hotpot_train_v1.1.json' # you can find in the Training set of HotpotQA: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

# Output Path Configuration
LLamaFactory_DIR = './LLaMA-Factory'
OUTPUT_DIR = os.path.join(LLamaFactory_DIR, 'data')
DATASET_INFO_FILE = os.path.join(OUTPUT_DIR, 'dataset_info.json')
FINAL_OUTPUT_FILE = 'exp_rl_coldstart_multi.json'  # Final output filename
ERROR_FILE = 'exp_rl_coldstart_error.jsonl'  # Error log filename
COLD_START_NAME = 'exp_rl_coldstart_multi'

# Sampling Configuration
SAMPLE_SIZE = 3  # Number of samples
RANDOM_SEED = 50  # Random seed


def read_json(path):
    """Read JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonal(path):
    """Read JSONL file"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_relevant_data(sample):
    """
    Extract query, relevant_document (sentences corresponding to supporting_facts), and answer from HotpotQA sample
    """
    query = sample['question']
    answer = sample['answer']
    context_dict = {title: sentences for title, sentences in sample['context']}
    
    # Extract supporting sentences (supporting facts)
    relevant_sentences = []
    for title, sent_id in sample['supporting_facts']:
        if title in context_dict and 0 <= sent_id < len(context_dict[title]):
            relevant_sentences.append(context_dict[title][sent_id])
        else:
            print(f"⚠️ Cannot find supporting sentence: title={title}, sent_id={sent_id}")
    
    return {
        '_id': sample['_id'],
        "query": query,
        "relevant_document": relevant_sentences,
        "answer": answer
    }


def match_hotpotqa_questions(selected_questions, hotpotqa_data):
    """Match questions in HotpotQA"""
    matched = []
    for q in selected_questions:
        for example in hotpotqa_data:
            if example['question'].strip().lower() == q.strip().lower():
                extract_example = extract_relevant_data(example)
                matched.append(extract_example)
                break  # Stop at the first match
    return matched


def gpt4_inference(prompt, api_key, base_url):
    """GPT-4 inference"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )
    print('gpt4:', response.choices[0].message.content)
    return response.choices[0].message.content


def process_response(response):
    """Process response and extract tag contents
    
    Extract tags according to prompt format:
    - think / Think
    - Search_Experience_Base
    - Think (decide whether knowledge is needed)
    - Search_Knowledge_Base (optional)
    - information (optional)
    - Think (apply rules)
    - Answer
    """
    # Extract all possible tags (new format uses lowercase tags)
    tag_patterns = {
        "think": r"<think>(.*?)</think>",
        "search_experience": r"<search_experience>(.*?)</search_experience>",
        "search_knowledge": r"<search_knowledge>(.*?)</search_knowledge>",
        "information": r"<information>(.*?)</information>",
        "answer": r"<answer>(.*?)</answer>",
    }
    
    tag_contents = {}
    
    # Extract all tags
    for tag_name, pattern in tag_patterns.items():
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            tag_contents[tag_name] = matches
        else:
            tag_contents[tag_name] = []
    
    # Check required tags
    missing_tags = []
    if not tag_contents.get("think"):
        missing_tags.append("think")
    if not tag_contents.get("search_experience"):
        missing_tags.append("search_experience")
    if not tag_contents.get("answer"):
        missing_tags.append("answer")
    
    if missing_tags:
        return None, missing_tags
    
    # Get content of each tag
    think_list = tag_contents.get("think", [])
    search_experience = tag_contents.get("search_experience", [""])[0].strip() if tag_contents.get("search_experience") else ""
    search_knowledge = tag_contents.get("search_knowledge", [""])[0].strip() if tag_contents.get("search_knowledge") else ""
    information = tag_contents.get("information", [""])[0].strip() if tag_contents.get("information") else ""
    answer = tag_contents.get("answer", [""])[0].strip() if tag_contents.get("answer") else ""
    
    # Get the first think as initial reasoning
    initial_think = think_list[0].strip() if len(think_list) > 0 else ""
    
    # Construct multi-turn dialogue messages
    messages = []
    
    # Message 1: Initial reasoning + search experience
    mes1_content = f"<think>{initial_think}</think>"
    if search_experience:
        mes1_content += f"<search_experience>{search_experience}</search_experience>"
    messages.append({
        "role": "assistant",
        "content": mes1_content
    })
    
    # Message 2: Experience returned
    if search_experience:
        messages.append({
            "role": "user",
            "content": f"<experience>{search_experience}</experience>"
        })
    
    # Message 3: Think whether knowledge is needed + search knowledge (if exists)
    think_after_experience = think_list[1].strip() if len(think_list) > 1 else ""
    mes3_content = f"<think>{think_after_experience}</think>" if think_after_experience else ""
    if search_knowledge:
        mes3_content += f"<search_knowledge>{search_knowledge}</search_knowledge>"
    if mes3_content:
        messages.append({
            "role": "assistant",
            "content": mes3_content
        })
    
    # Message 4: Knowledge returned (if exists)
    if information:
        messages.append({
            "role": "user",
            "content": f"<information>{information}</information>"
        })
    
    # Message 5: Final reasoning + answer
    # According to new format, there may be 2-3 think tags: initial, decide whether knowledge is needed, final reasoning
    think_final = think_list[2].strip() if len(think_list) > 2 else (think_list[1].strip() if len(think_list) > 1 and not think_after_experience else "")
    mes5_content = f"<think>{think_final}</think>" if think_final else ""
    if answer:
        mes5_content += f"<answer>{answer}</answer>"
    if mes5_content:
        messages.append({
            "role": "assistant",
            "content": mes5_content
        })
    
    return messages, None


def make_turn_from_response(item):
    """Construct multi-turn dialogue format from response"""
    message = []
    
    idd = item['_id']
    query = item['query']
    response = item['response']
    
    sys_info = {
        'role': 'system',
        "content": prompt_multi
    }
    message.append(sys_info)
    
    query_info = {
        'role': 'user',
        'content': query
    }
    message.append(query_info)
    
    segments, missing_tags = process_response(response)
    if missing_tags:
        return None, missing_tags
    
    message.extend(segments)
    
    info = {
        'id': idd,
        'messages': message
    }
    
    return info, None


if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Build full paths
    train_path = os.path.join(DATASET_DIR, 'train.json')
    final_output_path = os.path.join(OUTPUT_DIR, FINAL_OUTPUT_FILE)
    error_path = os.path.join(OUTPUT_DIR, ERROR_FILE)
    
    # Read data
    hotpotqa_data = read_json(HOTPOTQA_DIR)  # 90447
    train_data = read_jsonal(train_path)
    
    print('train_data len:', len(train_data))  # 169615
    
    # Filter non-NQ data
    non_nq_data = [sample for sample in train_data if sample.get('data_source') != 'nq']
    print("Non-NQ data count:", len(non_nq_data))  # 90447
    
    # Sample data
    random.seed(RANDOM_SEED)
    sampled_data = random.sample(non_nq_data, SAMPLE_SIZE)
    selected_questions = [sample['question'] for sample in sampled_data]
    
    # Match questions in HotpotQA
    matched = match_hotpotqa_questions(selected_questions, hotpotqa_data)
    print(f"\nTotal matched questions: {len(matched)}")
    
    # Process all data and save directly to final output
    result_messages = []
    
    with open(error_path, 'w', encoding='utf-8') as f_error:
        for index, item in tqdm(enumerate(matched), total=len(matched), desc="Processing data"):
            try:
                query = item['query']
                relevant_document = item['relevant_document']
                answer = item['answer']
                
                # Construct prompt
                formatted_prompt = prompt_teacher.format(
                    query=query,
                    relevant_document="\n".join(relevant_document),
                    answer=answer
                )
                
                print(f"[{index}] Calling GPT-4 API...")
                start_time = time.time()
                response = gpt4_inference(formatted_prompt, API_KEY, BASE_URL)
                end_time = time.time()
                
                # Construct result item
                result_item = {
                    "_id": item['_id'],
                    "query": query,
                    "relevant_document": relevant_document,
                    "answer": answer,
                    "prompt": formatted_prompt,
                    "response": response
                }
                
                # Process response and construct multi-turn dialogue format
                message_info, missing_tags = make_turn_from_response(result_item)
                
                if missing_tags:
                    # Write to error file
                    error_record = {
                        'id': item['_id'],
                        'missing_tags': missing_tags,
                        'query': query,
                        'response': response
                    }
                    f_error.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    print(f"[Warning] {item['_id']} missing tags {missing_tags}, recorded to error file, skipping this record.")
                    continue
                
                result_messages.append(message_info)
                
            except Exception as e:
                print(f"Error processing record {index}: {e}")
                error_record = {
                    'id': item.get('_id', 'unknown'),
                    'error': str(e),
                    'query': item.get('query', ''),
                    'response': item.get('response', '')
                }
                f_error.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                continue
    
    # Write final JSON file
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(result_messages, f, ensure_ascii=False, indent=4)
    
    print(f"\nProcessing complete!")
    print(f"- Final output file: {final_output_path}, total {len(result_messages)} records")
    print(f"- Error log file: {error_path}")
    
    # Update dataset_info.json
    # Read existing data first (if file exists), otherwise initialize as empty dict
    if os.path.exists(DATASET_INFO_FILE):
        with open(DATASET_INFO_FILE, 'r', encoding='utf-8') as f:
            data_info = json.load(f)
    else:
        data_info = {}
    
    # Update data
    data_info[COLD_START_NAME] = {
        "file_name": FINAL_OUTPUT_FILE,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "id": "id"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }
    
    # Save updated data
    with open(DATASET_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, ensure_ascii=False, indent=4)
    
    print('Dataset info updated successfully')
