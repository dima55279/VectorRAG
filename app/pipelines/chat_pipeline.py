from app.agents.orchestrator import OrchestratorAgent


def run_chat():

    orchestrator = OrchestratorAgent()

    while True:

        q = input("\nQuestion: ")

        if q == "exit":
            break

        result = orchestrator.run_question(q)

        print("\nAnswer:")
        print(result["answer"])

        print("\nDocuments:")
        print(result["documents"])