from app.core.prompts import RAG_PROMPT
from app.core.ollama_client import generate_answer


class AnswerAgent:

    def run(self, state):

        if not state["valid"]:

            state["answer"] = (
                "Недостаточно данных "
                "в нормативной базе."
            )

            return state

        context = "\n\n".join([
            d["document"]["content"]
            for d in state["retrieved_docs"]
        ])

        prompt = RAG_PROMPT.format(
            context=context,
            question=state["question"]
        )

        answer = generate_answer(prompt)

        state["answer"] = answer

        return state