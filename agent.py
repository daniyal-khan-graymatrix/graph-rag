from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from tools import combined_pdf_query_tool

# LLM configuration
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# Tools
tools = [combined_pdf_query_tool]

# System message: instruct the assistant to always use tools
system_prompt = SystemMessagePromptTemplate.from_template(
    template=(
        "You are a PDF knowledge assistant designed to help users understand content from government tender PDFs. "
        "You must use the `combined_pdf_query_tool` for every query, regardless of how simple or complex it seems. "
        "Never rely on your own internal knowledge or make assumptions. Always respond using the information retrieved by the tool. "
        "If the tool returns no useful information, respond with: \"I couldn't find any relevant information from the documents.\" "
        "Do not fabricate or summarize beyond what the results provide."
    )
)


# Human message: natural and voice-friendly input style
human_prompt = HumanMessagePromptTemplate.from_template(
    template=(
        "Question: {input}.\n"
        "Please answer naturally and clearly. Avoid special characters, excessive punctuation, or anything that could make speech sound unnatural. "
        "Use only the information provided by the `combined_pdf_query_tool`. If nothing relevant is found, say so clearly. "
        "Do not include document formatting like bullets, numbering, or long pauses—respond as if you’re speaking directly to someone."
    )
)


prompt = ChatPromptTemplate(
    messages=[
        system_prompt,
        human_prompt,
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ],
    input_variables=["input", "agent_scratchpad"]
)

# Create the agent
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    # verbose=True,
    # return_intermediate_steps=True
)
