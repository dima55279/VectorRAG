from openai import OpenAI


def generate_answer(prompt):

    client = OpenAI(
        base_url="https://polza.ai/api/v1",
        api_key=POLZA_API_KEY,
    )

    completion = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct-v0.1",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    return completion.choices[0].message.content