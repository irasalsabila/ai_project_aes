# This script implements a Streamlit app for essay scoring using multi-model embeddings.
# The app allows users to input their essays, select embedding types, and evaluate the text for various scoring attributes.

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors
from pathlib import Path
from albert import *
from groq import Groq
import os

# Constants
MODEL_NAME = "albert-base-v2"
BASE_DIR = Path(__file__).resolve().parent.parent  # Move up two levels to the project root
SAVE_DIR = BASE_DIR / 'result'
GLOVE_PATH = BASE_DIR / 'word_embeddings/glove.6B.300d.txt'
FASTTEXT_PATH = BASE_DIR / 'word_embeddings/wiki.en.vec'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debug: Print absolute paths to confirm they are correct
print("BASE_DIR:", BASE_DIR)
print("SAVE_DIR:", SAVE_DIR)
print("GLOVE_PATH:", GLOVE_PATH)
print("FASTTEXT_PATH:", FASTTEXT_PATH)

# Ensure directories exist
for path in [GLOVE_PATH, FASTTEXT_PATH, SAVE_DIR]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

client = Groq(api_key="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")

@st.cache_resource
def load_albert_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model

@st.cache_resource
def load_glove_model(glove_file_path):
    embedding_dict = {}
    with open(glove_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor(np.asarray(values[1:], dtype='float32'))
            embedding_dict[word] = vector.to(device)
    return embedding_dict

@st.cache_resource
def load_fasttext_model(fasttext_file_path):
    model = KeyedVectors.load_word2vec_format(fasttext_file_path, binary=False)
    return {word: torch.tensor(model[word]).to(device) for word in model.index_to_key}

# Load models
tokenizer, albert_model = load_albert_model()
glove_model = load_glove_model(GLOVE_PATH)
fasttext_model = load_fasttext_model(FASTTEXT_PATH)

# Attribute ranges
attribute_ranges = {
    'content': (0, 17), 'organization': (0, 16), 'word_choice': (0, 16),
    'sentence_fluency': (0, 15), 'conventions': (0, 15), 'language': (0, 4),
    'prompt_adherence': (0, 4), 'narrativity': (0, 4), 'style': (0, 0), 'voice': (0, 0)
}

# Display selected attributes based on essay type
def display_selected_attributes_ui(essay_type, attributes):
    st.write(f"Essay Type: {essay_type}")
    for attribute, value in attributes.items():
        if attribute in essay_type:
            st.write(f"{attribute.capitalize().replace('_', ' ')}: {value}")

# Streamlit app layout
st.title("Essay Scoring with Multi-Model Embeddings")

examples = {
        "Example 1": """In “Let there be dark,” Paul Bogard talks about the importance of darkness. \n
                Darkness is essential to humans. Bogard states, “Our bodies need darkness to produce the hormone melatonin, which keeps certain cancers from developing, and our bodies need darkness for sleep, sleep. Sleep disorders have been linked to diabetes, obesity, cardiovascular disease and depression and recent research suggests are main cause of “short sleep” is “long light.” Whether we work at night or simply take our tablets, notebooks and smartphones to bed, there isn’t a place for this much artificial light in our lives.” (Bogard 2). Here, Bogard talks about the importance of darkness to humans. Humans need darkness to sleep in order to be healthy. \n
                Animals also need darkness. Bogard states, “The rest of the world depends on darkness as well, including nocturnal and crepuscular species of birds, insects, mammals, fish and reptiles. Some examples are well known—the 400 species of birds that migrate at night in North America, the sea turtles that come ashore to lay their eggs—and some are not, such as the bats that save American farmers billions in pest control and the moths that pollinate 80% of the world’s flora. Ecological light pollution is like the bulldozer of the night, wrecking habitat and disrupting ecosystems several billion years in the making. Simply put, without darkness, Earth’s ecology would collapse...” (Bogard 2). Here Bogard explains that animals, too, need darkness to survive.""" ,
        "Example 2": """In response to our world’s growing reliance on artificial light, writer Paul Bogard argues that natural darkness should be preserved in his article “Let There be dark”. He effectively builds his argument by using a personal anecdote, allusions to art and history, and rhetorical questions. \n
                Bogard starts his article off by recounting a personal story – a summer spent on a Minnesota lake where there was “woods so dark that [his] hands disappeared before [his] eyes.” In telling this brief anecdote, Bogard challenges the audience to remember a time where they could fully amass themselves in natural darkness void of artificial light. By drawing in his readers with a personal encounter about night darkness, the author means to establish the potential for beauty, glamour, and awe-inspiring mystery that genuine darkness can possess. He builds his argument for the preservation of natural darkness by reminiscing for his readers a first-hand encounter that proves the “irreplaceable value of darkness.” This anecdote provides a baseline of sorts for readers to find credence with the author’s claims. \n
                Bogard’s argument is also furthered by his use of allusion to art – Van Gogh’s “Starry Night” – and modern history – Paris’ reputation as “The City of Light”. By first referencing “Starry Night”, a painting generally considered to be undoubtedly beautiful, Bogard establishes that the natural magnificence of stars in a dark sky is definite. A world absent of excess artificial light could potentially hold the key to a grand, glorious night sky like Van Gogh’s according to the writer. This urges the readers to weigh the disadvantages of our world consumed by unnatural, vapid lighting. Furthermore, Bogard’s alludes to Paris as “the famed ‘city of light’”. He then goes on to state how Paris has taken steps to exercise more sustainable lighting practices. By doing this, Bogard creates a dichotomy between Paris’ traditionally alluded-to name and the reality of what Paris is becoming – no longer “the city of light”, but moreso “the city of light…before 2 AM”. This furthers his line of argumentation because it shows how steps can be and are being taken to preserve natural darkness. It shows that even a city that is literally famous for being constantly lit can practically address light pollution in a manner that preserves the beauty of both the city itself and the universe as a whole. \n
                Finally, Bogard makes subtle yet efficient use of rhetorical questioning to persuade his audience that natural darkness preservation is essential. He asks the readers to consider “what the vision of the night sky might inspire in each of us, in our children or grandchildren?” in a way that brutally plays to each of our emotions. By asking this question, Bogard draws out heartfelt ponderance from his readers about the affecting power of an untainted night sky. This rhetorical question tugs at the readers’ heartstrings; while the reader may have seen an unobscured night skyline before, the possibility that their child or grandchild will never get the chance sways them to see as Bogard sees. This strategy is definitively an appeal to pathos, forcing the audience to directly face an emotionally-charged inquiry that will surely spur some kind of response. By doing this, Bogard develops his argument, adding gutthral power to the idea that the issue of maintaining natural darkness is relevant and multifaceted. \n
                Writing as a reaction to his disappointment that artificial light has largely permeated the prescence of natural darkness, Paul Bogard argues that we must preserve true, unaffected darkness. He builds this claim by making use of a personal anecdote, allusions, and rhetorical questioning.
                """
}

results = {}
feedback = ""

# Dropdown for examples or custom input
example_choice = st.selectbox("Select an example or provide your own text:", ["None"] + list(examples.keys()))
user_input = st.text_area("Enter your essay text here:", examples[example_choice] if example_choice != "None" else "")

# Display word and character count
char_count = len(user_input)
word_count = len(user_input.split())
st.write(f"Char: {char_count} | Word: {word_count}")

# Embedding options
embedding_options = {
    "ALBERT": None,
    "ALBERT + GloVe": "glove",
    "ALBERT + FastText": "fasttext"
}
selected_embeddings = [embedding_options[option] for option, selected in embedding_options.items() if st.checkbox(option, value=True)]

if st.button("Evaluate"):
    results = {}
    for embedding_type in selected_embeddings:
        # Run predictions for each embedding type
        score, quality_label, essay_type, content, organization, word_choice, sentence_fluency, conventions, \
            language, prompt_adherence, narrativity, style, voice = testContent(
                user_input,
                embedding_type=embedding_type,
                SAVE_DIR=SAVE_DIR,
                glove_model=glove_model,
                fasttext_model=fasttext_model,
                attribute_ranges=attribute_ranges
            )
        embedding_name = "ALBERT" if embedding_type is None else f"ALBERT + {embedding_type.capitalize()}"
        results[embedding_name] = {
            "score": score,
            "quality": quality_label,
            "essay_type": essay_type,
            "content": content,
            "organization": organization,
            "word_choice": word_choice,
            "sentence_fluency": sentence_fluency,
            "conventions": conventions,
            "language": language,
            "prompt_adherence": prompt_adherence,
            "narrativity": narrativity,
            "style": style,
            "voice": voice
        }

    # Generate a single feedback summary
    feedback = generate_feedback(user_input, score, quality_label)

    # Display results in a card layout
st.write("### Results")
cols = st.columns(len(selected_embeddings))

for i, (embedding_name, result) in enumerate(results.items()):
    with cols[i]:
        st.markdown(f"#### {embedding_name}")
        
        # Display score, quality level, and essay type
        st.write(f"""
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 8px;">
                <strong>Predicted Score:</strong> {result['score']}<br>
                <strong>Quality Level:</strong> {result['quality']}<br>
                <strong>Essay Type:</strong> {result['essay_type']}
            </div>
        """, unsafe_allow_html=True)
        
        # Display relevant attributes based on the essay type
        st.write("**Relevant Attributes:**")
        relevant_attributes = {
            "Argumentative": ['content', 'organization', 'word_choice', 'sentence_fluency', 'conventions'],
            "Dependent": ['content', 'prompt_adherence', 'language', 'narrativity'],
            "Narrative": ['content', 'organization', 'style', 'conventions', 'voice', 'word_choice', 'sentence_fluency']
        }
        
        # Get the attribute list for the essay type
        essay_type = result['essay_type']
        essay_type_attributes = relevant_attributes.get(essay_type, [])
        
        # Display each attribute in a bullet list format
        for attr in essay_type_attributes:
            attr_value = result.get(attr, "N/A")
            st.write(f"- **{attr.replace('_', ' ').capitalize()}:** {attr_value}")

# Display a single feedback summary
st.markdown("### Feedback Summary")
st.write(feedback)