import openai
from secrets import OPENAI_API_KEY, OPENAI_ORG_ID


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


def get_more_prompts(example_file):
    with open(example_file, "r") as fhand:
        examples = fhand.read().rstrip().split("\n")
    listed = "\n".join([f"{i+1}. {examples[i]}" for i in range(len(examples))])
    prompt = f"Write a list of 20 tasks that a programmer might ask an AI assistant for a quick answer:\n\n{listed}"
    

    openai.organization = OPENAI_ORG_ID
    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=2048,
        temperature=0.7,
        presence_penalty=1,
        frequency_penalty=0,
        best_of=2
    )

    # the zero implies k = 1
    completion = response['choices'][0]['text']
    lines = completion.split("\n")
    qs = [x[x.index(".") + 2:] for x in lines if "." in x]
    return qs

if __name__ == '__main__':
    qs = get_more_prompts("data_gen_prompts/example_coding_prompts.txt")
    print(qs)