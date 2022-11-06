from transformers import DonutProcessor, VisionEncoderDecoderModel
from word2number import w2n
from dateutil import relativedelta
from datetime import datetime
from word2number import w2n
from textblob import Word
from PIL import Image
import torch
import re

CHEQUE_PARSER_MODEL = "shivi/donut-base-cheque"
TASK_PROMPT = "<s_cord-v2>"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_donut_model_and_processor():
    donut_processor = DonutProcessor.from_pretrained(CHEQUE_PARSER_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(CHEQUE_PARSER_MODEL)
    model.to(device)
    return donut_processor, model

def prepare_data_using_processor(donut_processor: DonutProcessor, image):
    ## Pass image through donut processor's feature extractor and retrieve image tensor
    
    pixel_values = donut_processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    ## Pass task prompt for document (cheque) parsing task to donut processor's tokenizer and retrieve the input_ids
    decoder_input_ids = donut_processor.tokenizer(TASK_PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"]
    decoder_input_ids = decoder_input_ids.to(device)

    return pixel_values, decoder_input_ids

def load_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return image

def parse_cheque_with_donut(input_image_path: str):

    image = load_image(input_image_path)
    print("type image:", type(image))

    donut_processor, model = load_donut_model_and_processor()

    cheque_image_tensor, input_for_decoder = prepare_data_using_processor(donut_processor,image)
    
    outputs = model.generate(cheque_image_tensor,
                                decoder_input_ids=input_for_decoder,
                                max_length=model.decoder.config.max_position_embeddings,
                                early_stopping=True,
                                pad_token_id=donut_processor.tokenizer.pad_token_id,
                                eos_token_id=donut_processor.tokenizer.eos_token_id,
                                use_cache=True,
                                num_beams=1,
                                bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],
                                return_dict_in_generate=True,
                                output_scores=True,)

    decoded_output_sequence = donut_processor.batch_decode(outputs.sequences)[0]
    
    extracted_cheque_details = decoded_output_sequence.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")
    ## remove task prompt from token sequence
    cleaned_cheque_details = re.sub(r"<.*?>", "", extracted_cheque_details, count=1).strip()  
    ## generate ordered json sequence from output token sequence
    cheque_details_json = donut_processor.token2json(cleaned_cheque_details)
    print("cheque_details_json:",cheque_details_json['cheque_details'])
    
    ## extract required fields from predicted json

    amt_in_words  = cheque_details_json['cheque_details'][0]['amt_in_words']
    amt_in_figures = cheque_details_json['cheque_details'][1]['amt_in_figures']
    macthing_amts = match_legal_and_courstesy_amount(amt_in_words,amt_in_figures)
    
    payee_name = cheque_details_json['cheque_details'][2]['payee_name']
    cheque_date = '06/05/2022'
    stale_cheque = check_if_cheque_is_stale(cheque_date)

    return payee_name,amt_in_words,amt_in_figures,cheque_date,macthing_amts,stale_cheque

def spell_correction(amt_in_words: str) -> str:
    corrected_amt_in_words =''
    words = amt_in_words.split()
    words = [word.lower() for word in words]
    for word in words:
        word = Word(word)
        corrected_word = word.correct()+' '
        corrected_amt_in_words += corrected_word
    return corrected_amt_in_words

def match_legal_and_courstesy_amount(legal_amount: str ,courtesy_amount: str) -> bool:
    macthing_amts = False
    corrected_amt_in_words = spell_correction(legal_amount)
    print("corrected_amt_in_words:",corrected_amt_in_words)
    numeric_legal_amt = w2n.word_to_num(corrected_amt_in_words)
    print("numeric_legal_amt:",numeric_legal_amt)
    if int(numeric_legal_amt) == int(courtesy_amount):
        macthing_amts = True
    return macthing_amts

def check_if_cheque_is_stale(cheque_issue_date: str) -> bool:
    stale_check = False
    current_date = datetime.now().strftime('%d/%m/%Y')
    current_date_ = datetime.strptime(current_date, "%d/%m/%Y")
    cheque_issue_date_ = datetime.strptime(cheque_issue_date, "%d/%m/%Y")
    relative_diff = relativedelta.relativedelta(current_date_, cheque_issue_date_)
    months_difference = (relative_diff.years * 12) + relative_diff.months
    print("months_difference:",months_difference)
    if months_difference > 3:
        stale_check = True
    return stale_check

