import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import pandas as pd
from collections import defaultdict
import re

# Configure page
st.set_page_config(
    page_title="BERT NER Model",
    page_icon="🏷️",
    layout="wide"
)

# Cache the model and tokenizer loading
@st.cache_resource
def load_model_and_tokenizer():
    """Load the fine-tuned BERT model and tokenizer"""
    try:
        # Path to your downloaded model directory
        model_path = "C:\\Users\\youss\\OneDrive\\Desktop\\LLM Tasks\\NER"  # Update this to match your folder name

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Label names from CoNLL-2003
        label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        
        return model, tokenizer, label_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure your model files are in the correct directory.")
        st.info("Current expected path: ./BERT-finetuned-NER")
        st.info("Make sure the folder contains: config.json, model.safetensors, tokenizer files")
        return None, None, None

def predict_entities(text, model, tokenizer, label_names):
    """Predict named entities in the given text"""
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_labels = torch.argmax(predictions, dim=-1)
    
    # Convert tokens back to words and align with predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = predicted_labels[0].numpy()
    
    # Remove special tokens and align with original text
    entities = []
    current_entity = {"text": "", "label": "", "start": 0, "end": 0, "confidence": 0}
    
    word_ids = inputs.word_ids(batch_index=0)
    words = text.split()
    word_idx = 0
    char_idx = 0
    
    for i, (token, pred_label_id, word_id) in enumerate(zip(tokens, predicted_labels, word_ids)):
        if word_id is None:  # Skip special tokens
            continue
            
        label = label_names[pred_label_id]
        confidence = float(predictions[0][i][pred_label_id])
        
        if label.startswith('B-'):  # Beginning of entity
            # Save previous entity if exists
            if current_entity["text"]:
                entities.append(current_entity.copy())
            
            # Start new entity
            entity_type = label[2:]
            current_entity = {
                "text": words[word_id] if word_id < len(words) else token,
                "label": entity_type,
                "start": char_idx,
                "end": char_idx + len(words[word_id]) if word_id < len(words) else char_idx + len(token),
                "confidence": confidence
            }
            
        elif label.startswith('I-') and current_entity["label"] == label[2:]:  # Inside entity
            if word_id < len(words):
                current_entity["text"] += " " + words[word_id]
                current_entity["end"] = char_idx + len(words[word_id])
            current_entity["confidence"] = (current_entity["confidence"] + confidence) / 2
            
        else:  # Outside entity or different entity
            if current_entity["text"]:
                entities.append(current_entity.copy())
                current_entity = {"text": "", "label": "", "start": 0, "end": 0, "confidence": 0}
        
        # Update character index
        if word_id < len(words):
            char_idx = text.find(words[word_id], char_idx) + len(words[word_id])
    
    # Add last entity if exists
    if current_entity["text"]:
        entities.append(current_entity)
    
    return entities

def highlight_entities(text, entities):
    """Create highlighted text with entity labels"""
    if not entities:
        return text
    
    # Define colors for different entity types
    colors = {
        'PER': '#FFB6C1',    # Light pink for persons
        'ORG': '#98FB98',    # Pale green for organizations
        'LOC': '#87CEEB',    # Sky blue for locations
        'MISC': '#DDA0DD'    # Plum for miscellaneous
    }
    
    # Sort entities by start position
    entities = sorted(entities, key=lambda x: x['start'])
    
    highlighted_text = ""
    last_end = 0
    
    for entity in entities:
        # Add text before entity
        highlighted_text += text[last_end:entity['start']]
        
        # Add highlighted entity
        color = colors.get(entity['label'], '#FFFF00')
        confidence_str = f"{entity['confidence']:.2f}"
        highlighted_text += f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 1px;" title="Confidence: {confidence_str}">{entity["text"]} <sub>{entity["label"]}</sub></span>'
        
        last_end = entity['end']
    
    # Add remaining text
    highlighted_text += text[last_end:]
    
    return highlighted_text

def main():
    st.title("🏷️ BERT Named Entity Recognition")
    st.markdown("This app uses a fine-tuned BERT model to identify named entities in text.")
    
    # Load model and tokenizer
    model, tokenizer, label_names = load_model_and_tokenizer()
    
    if model is None:
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("Entity Types")
        st.markdown("""
        - **PER**: Person names
        - **ORG**: Organizations
        - **LOC**: Locations
        - **MISC**: Miscellaneous entities
        """)
        
        st.header("Model Information")
        st.markdown("""
        - **Base Model**: BERT-base-cased
        - **Dataset**: CoNLL-2003
        - **Task**: Named Entity Recognition
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Enter Text for Analysis")
        
        # Sample texts
        sample_texts = [
            "Barack Obama was born in Honolulu, Hawaii and served as President of the United States.",
            "Apple Inc. is headquartered in Cupertino, California and was founded by Steve Jobs.",
            "The United Nations headquarters is located in New York City.",
            "Google was founded by Larry Page and Sergey Brin at Stanford University."
        ]
        
        selected_sample = st.selectbox("Choose a sample text (optional):", [""] + sample_texts)
        
        # Text input
        input_text = st.text_area(
            "Text to analyze:",
            value=selected_sample,
            height=150,
            placeholder="Enter your text here..."
        )
        
        if st.button("Analyze Text", type="primary"):
            if input_text.strip():
                with st.spinner("Analyzing text..."):
                    entities = predict_entities(input_text, model, tokenizer, label_names)
                
                st.header("Results")
                
                # Display highlighted text
                st.subheader("Highlighted Text")
                highlighted = highlight_entities(input_text, entities)
                st.markdown(highlighted, unsafe_allow_html=True)
                
                # Display entities table
                if entities:
                    st.subheader("Detected Entities")
                    df = pd.DataFrame(entities)
                    df['confidence'] = df['confidence'].round(3)
                    st.dataframe(df, use_container_width=True)
                    
                    # Entity statistics
                    st.subheader("Entity Statistics")
                    entity_counts = defaultdict(int)
                    for entity in entities:
                        entity_counts[entity['label']] += 1
                    
                    stats_df = pd.DataFrame(
                        list(entity_counts.items()), 
                        columns=['Entity Type', 'Count']
                    )
                    st.bar_chart(stats_df.set_index('Entity Type'))
                else:
                    st.info("No entities detected in the text.")
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.header("Legend")
        st.markdown("""
        <div style="margin-bottom: 10px;">
            <span style="background-color: #FFB6C1; padding: 2px 4px; border-radius: 3px;">Person (PER)</span>
        </div>
        <div style="margin-bottom: 10px;">
            <span style="background-color: #98FB98; padding: 2px 4px; border-radius: 3px;">Organization (ORG)</span>
        </div>
        <div style="margin-bottom: 10px;">
            <span style="background-color: #87CEEB; padding: 2px 4px; border-radius: 3px;">Location (LOC)</span>
        </div>
        <div style="margin-bottom: 10px;">
            <span style="background-color: #DDA0DD; padding: 2px 4px; border-radius: 3px;">Miscellaneous (MISC)</span>
        </div>
        """, unsafe_allow_html=True)
        
        if 'entities' in locals() and entities:
            st.header("Quick Stats")
            total_entities = len(entities)
            avg_confidence = np.mean([e['confidence'] for e in entities])
            st.metric("Total Entities", total_entities)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

if __name__ == "__main__":
    main()