# Use this pattern when a task is too complex for a single prompt
# Use the Routing pattern when an agent must decide between multiple distinct workflows, tools, or sub-agents 
# based on the user's input or the current state. It is essential for applications 
# that need to triage or classify incoming requests to handle different types of tasks, 
# such as a customer support bot distinguishing between sales inquiries, technical support, and account management questions
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()


#-- Configuration --
try:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

##-- Define Simulated sub-Agent Handlers(equivalent to ADK sub_agents)--##

def booking_handler(request: str) -> str:
    """Simulates the Booking Agent Handling a request"""
    print("\n--- Delegating to Booking handler  ---")
    return f"Booking Handler processed request: '{request}'. Result: 'Simulated booking action.'"

def info_handler(request: str) -> str:
    """Simulates the Info Agent Handling a request"""
    print("\n--- Delegating to Info handler  ---")
    return f"Info Handler processed request: '{request}'. Result: 'Simulated info action.'"

def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated"""
    print("\n--- Handling Unclear request  ---")
    return f"Coordinator could not delegate request: '{request}'. Result: 'Please clarify your request.'"


##-- Define Coordinator Router Chain(equivalent to ADK coordinator's instruction)
# This chain decides which handler to delegate to

coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", """ Analyze the user's request and determine which specialist handler should process it.
    - If the request is related to booking flights or hotels, output 'booker'.
    - For all other general information questions, output 'info'.
    - If the request is unclear or doesn't fit either category, output 'unclear'.
    ONLY output one word: 'booker', 'info', or 'unclear'."""),
    ("user", "{request}")
])

if llm:
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

##-- Define the Delegation Logic (equivalent to ADK's Auto-flow based on sub_agents)   --##
# Use RunnableBranch to route based on the router chain's output


# Helper to log the decision before calling handler
def log_and_call(handler, x, decision_name):
    print(f"Router Decision: '{x['decision'].strip()}' → Delegating to {decision_name}")
    return handler(x['request'])

# Define the branches for the RunnableBranch
branches = {
    "booker": RunnablePassthrough.assign(output=lambda x: log_and_call(booking_handler, x, "booking_handler")),
    "info": RunnablePassthrough.assign(output=lambda x: log_and_call(info_handler, x, "info_handler")),
    "unclear": RunnablePassthrough.assign(output=lambda x: log_and_call(unclear_handler, x, "unclear_handler")),
}

# Create the RunnableBranch. It takes the output of the rounter chain
# and routes the original input('request') to the corresponding handler

delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'booker', branches["booker"]),
    (lambda x: x['decision'].strip() == 'info', branches["info"]),
    branches["unclear"],
)

# Combine the router chain and the delegation branch into a single runnable
# The rounter chain's output ('decision') is passed along with the original input ('request')
# to the delegation branch

coordinator_agent = {
    "decision": coordinator_router_chain,
    "request": RunnablePassthrough(),
} | delegation_branch | (lambda x: x['output']) # extract the final output


## example useage

def main():
    if not llm:
        print("LLM not initialized. Please check the configuration.")
        return

    print("\n--- Running with a booking request ---")
    request_A = "Book me a flight to London"
    result_A = coordinator_agent.invoke({"request": request_A})
    print(f"\nResult: {result_A}")

    print("\n--- Running with a info request ---")
    request_B = "What is the capital of France?"
    result_B = coordinator_agent.invoke({"request": request_B})
    print(f"\nResult: {result_B}")

    print("\n--- Running with a unclear request ---")
    request_C = "What is the meaning of life?"
    result_C = coordinator_agent.invoke({"request": request_C})
    print(f"\nResult: {result_C}")


if __name__ == "__main__":
    main()



## notes of Google ADK version since I didn't implement it
## InMemoryRunner
## 