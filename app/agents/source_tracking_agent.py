class SourceTrackingAgent:

    def run(self, state):

        docs = state["retrieved_docs"]

        sources = list({
            d["document"]["document_name"]
            for d in docs
        })

        state["sources"] = sources

        return state