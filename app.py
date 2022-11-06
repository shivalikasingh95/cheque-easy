import os
import glob
import gradio as gr
from predict_cheque_parser import parse_cheque_with_donut

##Create list of examples to be loaded
example_list = glob.glob("examples/cheque_parser/*")
faulty_cheques_list = glob.glob("examples/cheque_analyze/*")
example_list = list(map(lambda el:[el], example_list))
faulty_cheques_list = list(map(lambda el:[el], faulty_cheques_list))

demo = gr.Blocks(css="#warning {color: red}")

with demo:
    
    gr.Markdown("# **<p align='center'>ChequeEasy: Banking with Transformers </p>**")
    gr.Markdown("This space demonstrates the use of Donut proposed in this <a href=\"https://arxiv.org/abs/2111.15664/\">paper </a>")
    
    with gr.Tabs():
        
        with gr.TabItem("Cheque Parser"):
            gr.Markdown("The module is used to extract details filled by a bank customer from cheques. At present the model is trained to extract details like - payee_name, amount_in_words, amount_in_figures. This model can be further trained to parse additional details like micr_code, cheque_number, account_number, etc")
            with gr.Box():
                gr.Markdown("**Upload Cheque**")
                input_image_parse = gr.Image(type='filepath', label="Input Cheque")
            with gr.Box():
                gr.Markdown("**Parsed Cheque Data**")
        
                payee_name = gr.Textbox(label="Payee Name")
                amt_in_words = gr.Textbox(label="Courtesy Amount",elem_id="warning")
                amt_in_figures = gr.Textbox(label="Legal Amount")
                cheque_date = gr.Textbox(label="Cheque Date")

                # micr_code = gr.Textbox(label="MICR code")
                # cheque_number = gr.Textbox(label="Cheque Number")
                # account_number = gr.Textbox(label="Account Number")
                
                amts_matching = gr.Checkbox(label="Legal & Courtesy Amount Matching")
                stale_check = gr.Checkbox(label="Stale Cheque")
            
            with gr.Box():
                gr.Markdown("**Predict**")
                with gr.Row():
                    parse_cheque = gr.Button("Call Donut 🍩")
            
            with gr.Column():
                gr.Examples(example_list, [input_image_parse], 
                            [payee_name,amt_in_words,amt_in_figures,cheque_date],parse_cheque_with_donut,cache_examples=False)
                                    # micr_code,cheque_number,account_number,
                                    # amts_matching, stale_check]#,cache_examples=True)
            
            
        with gr.TabItem("Quality Analyzer"):
            gr.Markdown("The module is used to detect any mistakes made by bank customers while filling out the cheque or while taking a snapshot of the cheque. At present the model is trained to find mistakes like -'object blocking cheque', 'overwriting in cheque'. ")
            with gr.Box():
                gr.Markdown("**Upload Cheque**")
                input_image_detect = gr.Image(type='filepath',label="Input Cheque", show_label=True)
            
            with gr.Box(): # with gr.Column():
                gr.Markdown("**Cheque Quality Results:**")
                output_detections = gr.Image(label="Analyzed Cheque Image", show_label=True)
                output_text = gr.Textbox()
            
            with gr.Box():
                gr.Markdown("**Predict**")
                with gr.Row():
                    analyze_cheque = gr.Button("Call YOLOS 🤙")
            
            gr.Markdown("**Examples:**")

            with gr.Column():
                gr.Examples(faulty_cheques_list, input_image_detect, [output_detections, output_text])#, predict, cache_examples=True)
        
    
    parse_cheque.click(parse_cheque_with_donut, inputs=input_image_parse, outputs=[payee_name,amt_in_words,amt_in_figures,cheque_date,amts_matching,stale_check])
                                    # micr_code,cheque_number,account_number,
                                    # amts_matching, stale_check])
    # analyze_cheque.click(predict, inputs=input_image_detect, outputs=[output_detections, output_text])
    
    gr.Markdown('\n Solution built by: <a href=\"https://www.linkedin.com/in/shivalika-singh/\">Shivalika Singh</a>')
    
demo.launch(share=True, debug=True)
