from agent import agent_executor

if __name__ == "__main__":
    while True:
        user_input = input("Ask your question about the PDF (type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            response = agent_executor.invoke({"input": user_input})
            print("\nAnswer:", response["output"], "\n")
        except Exception as e:
            print(f"Error: {e}")
