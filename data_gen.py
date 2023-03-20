import openai
from secrets import OPENAI_API_KEY, OPENAI_ORG_ID
import json
import os


def get_gpt_answer_prompt(instruction, context):

    if not context:
        context = "(none)"

    return f"""The following is a part of a conversation between a user and a knowledgable, honest AI assistant. The AI provides accurate, concise information.

Instruction: {instruction}

---

Optional context:

{context}

---

Response:

"""

def get_gpt_answer(instruction, context=None):
    prompt = get_gpt_answer_prompt(instruction, context)
    openai.organization = OPENAI_ORG_ID
    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=512,
        temperature=0.7,
        presence_penalty=1,
        frequency_penalty=1
    )

    # the zero implies k = 1
    completion = response['choices'][0]['text']
    return completion


def get_more_prompts(prompt_file, example_file):
    with open(example_file, "r") as fhand:
        examples = fhand.read().rstrip().split("\n")
    with open(prompt_file, "r") as fhand:
        prompt = fhand.read().rstrip()

    listed = "\n".join([f"{i+1}. {examples[i]}" for i in range(len(examples))])
    prompt = f"{prompt}:\n\n{listed}"
    

    openai.organization = OPENAI_ORG_ID
    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=2048,
        temperature=0.7,
        presence_penalty=.5,
        frequency_penalty=0,
        best_of=2
    )

    # the zero implies k = 1
    completion = response['choices'][0]['text']
    lines = completion.split("\n")
    qs = examples + [x[x.index(".") + 2:] for x in lines if "." in x]
    return qs

if __name__ == '__main__':
    prompts_dir = os.path.join("data_gen_prompts", "coding")

    outfile = os.path.join(prompts_dir, "coding_prompts.json")
    gen_qs_prompt_path = os.path.join(prompts_dir, "gen_coding_qs_prompt.txt")
    example_qs_path = os.path.join(prompts_dir, "example_coding_prompts.txt")

    qs = get_more_prompts(gen_qs_prompt_path, example_qs_path)
    print(qs)
    json.dump(qs, open(outfile, "w"))
    