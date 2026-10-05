# GraphWalker

An iterative, agentic walk & rephrase approach to graph-based retrieval augmented generation (RAG). The walker determines where to 
look in the graph based on a current position, while the rephraser decides whether sufficient information has been gathered or 
whether more specific subquestions need to be formulated.

Basic setup
-------

1. Clone this project: `git clone...`.
2. Preferably create a virtual environment (`conda create --name gwalker`) and activate it (`conda activate gwalker`).
3. `cd` to the project's root folder and install all required packages: `pip install -r requirements.txt`.
4. Run tests on MetaQA using `python main.py <results-folder> <results-setting/subfolder> <1-hop|2-hop|3-hop> <filtering> <positioning> <model-name> <api-url> <cap>`. 
Here, `<filtering>` and `<positioning>` should be either 0 or 1 depending on whether these settings should be switched off or on. 
`<model-name>` and `<api-url>` refer to the large language model being used as a backbone and the remote api address at which it can be accessed. 
Finally, `<cap>` indicates how many questions in the question set should be evaluated. Setting `<cap>` to -1 indicates that all questions will be evaluated.