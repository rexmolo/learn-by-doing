import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

#prompt-1 extract information
prompt_extract = ChatPromptTemplate.from_template(
    """
    Extract the technical specification from the following text:\n\n
    {text_input}
    """
)

#prompt-2: transform to JSON
prompt_transform = ChatPromptTemplate.from_template(
    """
    Transform the following technical specification into a JSON object with 'cpu', 'memory', 'storage', 'gpu' as keys:\n\n
    {specification}
    """
)

# --- Build the Chain using LCEL ---
# The StrOutputParser() converts the LLM's message output to a simple string.
# --- 使用 LCEL 构建链 ---
# StrOutputParser() 

extraction_chain = prompt_extract | llm | StrOutputParser()


# The full chain passes the output of the extraction chain into the 'specifications'
# variable for the transformation prompt.
full_chain = (
    {"specification" : extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)

# --- Run the Chain ---
input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."

# Execute the chain with the input text dictionary.
final_result = full_chain.invoke({"text_input": input_text})

print("\n--- Final JSON Output ---")
print(final_result)