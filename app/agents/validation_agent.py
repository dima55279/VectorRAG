class ValidationAgent:

    def run(self, state):

        docs = state["retrieved_docs"]

        if len(docs) == 0:

            state["valid"] = False
            return state

        score = docs[0]["score"]

        state["valid"] = score > 0.45

        return state