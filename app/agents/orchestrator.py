from langchain_core.runnables import (
    RunnableLambda,
    RunnableSequence
)

from app.agents.retrieval_agent import RetrievalAgent
from app.agents.reranker_agent import RerankerAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.answer_agent import AnswerAgent
from app.agents.source_tracking_agent import (
    SourceTrackingAgent
)


class OrchestratorAgent:

    def __init__(self):

        self.retrieval_agent = RetrievalAgent()

        self.reranker_agent = RerankerAgent()

        self.validation_agent = ValidationAgent()

        self.answer_agent = AnswerAgent()

        self.source_agent = SourceTrackingAgent()

        self.graph = RunnableSequence(

            RunnableLambda(
                self.retrieval_agent.run
            ),

            RunnableLambda(
                self.reranker_agent.run
            ),

            RunnableLambda(
                self.validation_agent.run
            ),

            RunnableLambda(
                self.source_agent.run
            ),

            RunnableLambda(
                self.answer_agent.run
            )
        )

    def run_question(self, question):

        result = self.graph.invoke({
            "question": question
        })

        return {
            "answer": result["answer"],
            "documents": result["sources"]
        }