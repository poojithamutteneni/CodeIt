from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/codegemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/codegemma-2b")

def generate_code(prompt):

    try:
        input_text = (
            "Generate clean, modular Python code for " + prompt +
            ". Strip out '''python from the beginning and ''' in the end. " +
            "Do not include anything else except just the Python code. NOTHING ELSE!"
        )

        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output_ids = model.generate(input_ids, max_length=500)
        generated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_code.strip()
    except Exception as e:
        return f"An error occurred while generating code: {e}"

if __name__ == "__main__":
    user_prompt = "calculate the factorial of a number"
    generated_code = generate_code(user_prompt)
    print("Generated Code:\n", generated_code)
