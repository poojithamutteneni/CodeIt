from mistralai import Mistral

api_key = "NA8TwAMiCFePUk47d1VEP0biNJnlOizp"  

model = "mistral-large-latest"

client = Mistral(api_key=api_key)

def generate_code(prompt):

    try:
        response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate clean, modular Python code for " + prompt + ". " +"Strip out '''python from the beginning and ''' in the end"+"Do not give anything else except just the python code. NOTHING ELSE! No comments, no strings nothing else at all."
                    ),
                },
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    user_prompt = "calculate the factorial of a number"
    generated_code = generate_code(user_prompt)
    print("Generated Code:\n", generated_code)
