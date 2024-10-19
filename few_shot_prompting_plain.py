import warnings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from few_shot_examples import examples
from langchain_core.prompts import FewShotPromptTemplate

warnings.filterwarnings('ignore')
_ = load_dotenv()

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

few_examples = examples

prompt = FewShotPromptTemplate(
    examples=few_examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(
    prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
)




