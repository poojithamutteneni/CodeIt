import openai

openai.api_key = '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'

class Agent:
 
    def __init__(self, role, model="gpt-3.5-turbo"):
        self.role = role
        self.model = model

    def communicate(self, prompt):
   
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a {self.role} agent."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        return response['choices'][0]['message']['content'].strip()

class MultiAgentSystem:
 
    def __init__(self):
        self.designer = Agent(role="designer", model="gpt-3.5-turbo")
        self.coder = Agent(role="coder", model="gpt-4")  # More capable model for coding
        self.tester = Agent(role="tester", model="gpt-3.5-turbo")

    def run(self, task):
       
        design_prompt = f"Design a Python module for the task: '{task}'"
        design = self.designer.communicate(design_prompt)

        code_prompt = (
            "Generate clean, modular Python code for " + task + ". "+ "Strip out '''python from the beginning and ''' in the end" +"Do not give anything else except just the python code. NOTHING ELSE!"
        )
        code = self.coder.communicate(code_prompt)

        test_prompt = f"Generate test cases for the following code:\n{code}"
        test_cases = self.tester.communicate(test_prompt)

        reflection_prompt = f"Review the following code and suggest improvements or optimizations:\n{code}"
        reflection = self.tester.communicate(reflection_prompt)

        return code

if __name__ == "__main__":
    task = "calculate the factorial of a number"

    system = MultiAgentSystem()
    generated_code = system.run(task)

    print("Generated Code:\n", generated_code)
