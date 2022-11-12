import os
import glob
import gradio as gr
from predict_cheque_parser import parse_cheque_with_donut

##Create list of examples to be loaded
example_list = glob.glob("examples/cheque_parser/*")
example_list = list(map(lambda el:[el], example_list))

demo = gr.Blocks()

with demo:
    
    gr.Markdown("# **<p align='center'>ChequeEasy: Banking with Transformers </p>**")
    gr.Markdown("ChequeEasy is a project that aims to simplify the process of approval of cheques and making it easier for both bank officials and customers. \
    This project leverages Donut model proposed in the paper <a href=\"https://arxiv.org/abs/2111.15664/\"> OCR-free Document Understanding Transformer </a> for the parsing of the required data from cheques. \
    Donut is based on a very simple transformer encoder and decoder architecture. It's main USP is that it is an OCR-free approach to Visual Document Understanding (VDU) and can perform tasks like document classification, information extraction as well as VQA. \
    OCR based techniques come with several limitations such as requiring use of additional downstream models, lack of understanding about document structure, requiring use of hand crafted rules for information extraction,etc. \
    Donut helps you get rid of all of these OCR specific limitations. The model for the project has been trained using a subset of this  <a href=\"https://www.kaggle.com/datasets/medali1992/cheque-images/\"> kaggle dataset </a>. The original dataset contains images of cheques of 10 different banks. \
    A filtered version of this dataset containing images of cheques from 4 banks that are more commonly found in the Indian Banking Sector was created with ground truth prepared in the format required for fine-tuning Donut. This <a href=\"https://huggingface.co/datasets/shivi/cheques_sample_data/\"> dataset </a> is available on the Hugging Face Hub for download.")

    
    with gr.Tabs():
        
        with gr.TabItem("Cheque Parser"):
            gr.Markdown("This module is used to extract details filled by a bank customer from cheques. \
            - At present the model is trained to extract details like - Payee Name, Amount in words, Amount in Figures, Bank Name and Cheque Date.  \
            - This model can be further trained to parse additional details like MICR Code, Cheque Number, Account Number, etc.   \
            - Additionally, the app compares if the extracted legal & courtesy amount are matching which is an important check done during approval process of cheques. \
            - It also checks if the cheque is stale. A cheque is considered stale if it is presented to the bank 3 months after the date mentioned on the cheque.")
            
            with gr.Box():
                gr.Markdown("**Upload Cheque**")
                input_image_parse = gr.Image(type='filepath', label="Input Cheque")
            with gr.Box():
                gr.Markdown("**Parsed Cheque Data**")
        
                payee_name = gr.Textbox(label="Payee Name")
                amt_in_words = gr.Textbox(label="Legal Amount")
                amt_in_figures = gr.Textbox(label="Courtesy Amount")
                cheque_date = gr.Textbox(label="Cheque Date")
                bank_name = gr.Textbox(label="Bank Name")
                
                amts_matching = gr.Checkbox(label="Legal & Courtesy Amount Matching")
                stale_check = gr.Checkbox(label="Stale Cheque")
            
            with gr.Box():
                gr.Markdown("**Predict**")
                with gr.Row():
                    parse_cheque = gr.Button("Call Donut 🍩")
            
            with gr.Column():
                gr.Examples(example_list, [input_image_parse], 
                            [payee_name,amt_in_words,amt_in_figures,cheque_date],parse_cheque_with_donut,cache_examples=False)
        
    
    parse_cheque.click(parse_cheque_with_donut, inputs=input_image_parse, outputs=[payee_name,amt_in_words,amt_in_figures,bank_name,cheque_date,amts_matching,stale_check])
                                    
    gr.Markdown('\n Solution built by: <a href=\"https://twitter.com/singhshiviii/\">Shivalika Singh</a>')
    
demo.launch(share=True,debug=True)