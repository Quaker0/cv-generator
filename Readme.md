# A LLM setup that generate variations of your CV

Input your CV and Cover Letter and use that information to answer a specific question. Very handy since job applications include answering the same questions phrased slightly differently. Also include the job description to pick out the most transferable skills to highlight.

## Setup

```sh
brew install ollama
ollama pull llama3:8b
```

## Use the generator

Edit the files/ to include your own CV, cover letter and a job description to target.

```sh
make install-dev -- not needed for simple-context-gen
make build -- only for langchain-vector-gen
make run
```

## Multiple generators

- `simple-context-gen`
  The most basic setup, everything in a Modelfile, this one doesn't take a job description but works very well otherwise.

- `langchain-context-gen`
  Uses langchain, sets up more rules but still sends in everything in the context. This one will give the best results.

- `langchain-vector-gen`
  Uses langchain, same as `langchain-context-gen` except it uses a vector database and filters on relevance. This could support infinately large data amounts. It doesn't really matter for our case since we just input a CV and cover letter there which is way below the limit and it will discard information that would have made the text richer.
