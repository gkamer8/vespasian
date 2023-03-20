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
        prompt=instruction,
        max_tokens=512,
        temperature=0.7,
        presence_penalty=1,
        frequency_penalty=1
    )

    # the zero implies k = 1
    completion = response['choices'][0]['text']
    print(completion)


if __name__ == '__main__':
    get_gpt_answer("Please write javascript code that reverses an array")