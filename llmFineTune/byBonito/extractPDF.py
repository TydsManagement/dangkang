import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Open the PDF file
    text = ""
    for page in doc:  # Iterate through each page
        text += page.get_text()  # Extract text and append it to the text variable
    return text

pdf_path = 'Emerging_Tech_Bitcoin_Crypto.pdf'  # Specify the path to your PDF document
text = extract_text_from_pdf(pdf_path)  # Call the function with the path to your PDF
print("###extracted text:",text)
##split text to sentence

import spacy

nlp = spacy.load("en_core_web_sm")  # Load the English language model

def split_into_sentences(text):
    doc = nlp(text)  # Process the text with SpaCy
    sentences = [sent.text.strip() for sent in doc.sents]  # Extract sentences and strip whitespace
    return sentences

sentences = split_into_sentences(text) # Split the extracted text into sentences
print("##splited sentence:",sentences)
