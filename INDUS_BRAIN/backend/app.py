import os
import fitz
import chromadb
import networkx as nx
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import google.generativeai as genai
from flask import Flask, request, render_template, redirect
from sentence_transformers import SentenceTransformer

document_text = ""

GEMINI_API_KEY = "YOUR API KEY HERE"

genai.configure(api_key=GEMINI_API_KEY)

model_gemini = genai.GenerativeModel("gemini-3-flash-preview")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("documents")

HTML = """
<h1>INDUS-BRAIN MVP</h1>

<h2>Upload PDF</h2>
<form method="post" action="/upload" enctype="multipart/form-data">
<input type="file" name="pdf">
<input type="submit">
</form>

<h2>Ask Question</h2>
<form method="post" action="/ask">
<input type="text" name="question" style="width:400px">
<input type="submit">
</form>

<h2>Generate Knowledge Graph</h2>
<a href="/graph">Generate Graph</a>

<h2>Compliance Report</h2>
<a href="/compliance">View Report</a>

{% if answer %}
<hr>
<h3>Answer</h3>
<p>{{answer}}</p>
{% endif %}
"""

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, size=1000):
    return [text[i:i+size] for i in range(0, len(text), size)]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files["pdf"]
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)

    global document_text

    text = extract_text(path)
    document_text = text
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        emb = model.encode(chunk).tolist()
        collection.add(
            ids=[f"{f.filename}_{i}"],
            embeddings=[emb],
            documents=[chunk]
        )

    return redirect("/")

@app.route("/ask", methods=["POST"])
def ask():
    q = request.form["question"]
    emb = model.encode(q).tolist()

    results = collection.query(
        query_embeddings=[emb],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
    You are an Industrial Knowledge Assistant.

    Answer only from the provided document context.

    Do NOT use:
    - Markdown
    - *
    - **
    - #
    - Bullet symbols

    Use simple numbered lists.

    Document Context:
    {context[:4000]}

    Question:
    {q}

    Provide a clear and professional answer.
    """

    response = model_gemini.generate_content(prompt)

    answer = response.text

    return render_template("index.html", answer=answer)

@app.route("/graph")
def graph():
    global document_text

    if not document_text:
        return """
        <h2>No document uploaded.</h2>
        <a href="/">Go Back</a>
        """

    try:

        prompt = f"""
        Analyze this industrial document.

        Extract:

        Equipment
        Procedures
        Regulations

        Return ONLY in this exact format:

        Equipment:
        item1
        item2

        Procedures:
        item1
        item2

        Regulations:
        item1
        item2

        Document:
        {document_text[:5000]}
        """

        response = model_gemini.generate_content(prompt)

        extracted = response.text

        equipment = []
        procedures = []
        regulations = []

        current = None

        for line in extracted.splitlines():

            line = line.strip()

            if not line:
                continue

            if line.startswith("Equipment"):
                current = equipment
                continue

            elif line.startswith("Procedures"):
                current = procedures
                continue

            elif line.startswith("Regulations"):
                current = regulations
                continue

            if current is not None:
                current.append(line)

        G = nx.Graph()

        for eq in equipment:

            for proc in procedures:
                G.add_edge(eq, proc)

            for reg in regulations:
                G.add_edge(eq, reg)

        if len(G.nodes()) == 0:
            return """
            <h2>No entities found.</h2>
            <a href="/">Go Back</a>
            """

        plt.switch_backend("Agg")

        plt.figure(figsize=(20, 20))

        pos = nx.spring_layout(
            G,
            k=3,
            iterations=100,
            seed=42
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=4000
        )

        nx.draw_networkx_edges(
            G,
            pos,
            width=2
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=10,
            font_weight="bold"
        )

        plt.axis("off")

        plt.savefig(
            "static_graph.png",
            bbox_inches="tight",
            pad_inches=1
        )

        plt.close("all")

        return """
        <h2>Generated Knowledge Graph</h2>
        <img src="/static_graph.png" width="900">
        <br><br>
        <a href="/">Back</a>
        """

    except Exception as e:

        return f"""
        <h2>Error Generating Graph</h2>
        <pre>{str(e)}</pre>
        <a href="/">Back</a>
        """

@app.route("/static_graph.png")
def graph_image():
    from flask import send_file
    return send_file("static_graph.png")

@app.route("/compliance")
def compliance():
    return """
    <h1>Compliance Report</h1>
    <h3>Score: 85%</h3>
    <ul>
      <li>Missing Fire Drill Record</li>
      <li>Missing Inspection Evidence</li>
    </ul>
    """

if __name__ == "__main__":
    app.run()
