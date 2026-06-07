
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from providers import get_chat_model
from pydantic import BaseModel


llm = get_chat_model(temperature=0)

#output parser
class PopularityStructure(BaseModel):
    """A structure to hold the popularity of a topic."""
    topic: str
    popularity: str

parser = PydanticOutputParser(pydantic_object=PopularityStructure)

# prompt
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# chain with the pipe operator
chain = prompt | llm | parser

# invoke the chain
response = chain.invoke({"query": "What is the most popular popcorn flavor?"})
print(response)